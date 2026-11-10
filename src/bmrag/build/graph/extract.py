"""Graph stage "extract": one multimodal LLM call per PDF returns the paper's cases and rules.

Writes kg_cases/<paper>.json per paper, then kg_cases/ALL_papers.json ({"papers": [...]}) and
kg_cases/ALL_cases.json, which the entities stage reads. A paper whose record exists is reused
unless --force is given.

When the profile scopes ids to papers (record_ids.paper_scoped_ids), each paper in ALL_papers.json
also gets "source_pdf": the PDF's file name and SHA-256, written by the pipeline, not the model.
The entities stage keys ids on that hash. The per-paper files are left as the model returned them,
so a reused record gets the field again on every run and paper_v1 output is unchanged.

A reply that is not valid JSON is first sent back to the same model for a syntax-only repair;
if that fails, the request is repeated with the parse error appended, up to max_attempts.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from bmrag.build import llm_client
from bmrag.build.graph import extraction_prompt as prompts
from bmrag.build.graph.extraction_records import normalize_paper_record, paper_cases, paper_slug, parse_json_reply
from bmrag.build.graph.paper_content import PaperContent, read_paper
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.manifest import sha256_file
from bmrag.build.profiles import ExtractionSettings
from bmrag.build.stages import BuildContext

log = logging.getLogger(__name__)
STAGE = "extract"


class Extractor:
    """Sends papers to one model (OpenAI or Anthropic, chosen by the model name)."""

    def __init__(self, settings: ExtractionSettings, usage: llm_client.UsageLog):
        self.settings = settings
        self.usage = usage
        self.anthropic = llm_client.is_anthropic(settings.model)
        self.client = llm_client.anthropic_client() if self.anthropic else llm_client.openai_client()

    def extract(self, pdf_path: Path) -> dict[str, Any]:
        """The normalized paper record, or {} when every attempt failed."""
        content = read_paper(pdf_path, self.settings)
        paper_type = prompts.infer_paper_type(content.text, pdf_path.name)
        log.info(
            "  text=%s chars | tables=%s chars | image_pages=%s | detected=%s",
            f"{len(content.text):,}",
            f"{len(content.tables_markdown):,}",
            [p for p, _ in content.page_images],
            paper_type,
        )
        system = prompts.system_prompt(self.settings.prompt, paper_type)
        model = self.settings.model
        follow_up: list[dict] = []  # parse errors from earlier attempts, sent with the next one
        last_error: Exception | None = None

        for attempt in range(1, self.settings.max_attempts + 1):
            raw = None
            try:
                raw = self._call(system, content, pdf_path.name, follow_up)
                record = normalize_paper_record(parse_json_reply(raw), pdf_path, paper_type, model)
                log.info("  %s -> extracted %d case record(s)", model, len(paper_cases(record)))
                return record
            except ValueError as exc:  # includes json.JSONDecodeError
                last_error = exc
                log.warning(
                    "  %s attempt %d/%d: JSON parse error - %s", model, attempt, self.settings.max_attempts, exc
                )
                if raw:
                    log.warning("  %s raw preview: %s", model, raw[:600].replace("\n", "\\n"))
                if not self.anthropic and raw:
                    repaired = self._repair(raw, exc, pdf_path, paper_type)
                    if repaired is not None:
                        return repaired
                if attempt < self.settings.max_attempts:
                    message = f"Your previous response failed JSON parsing: {exc}. Return ONLY a valid JSON object."
                    if self.anthropic:
                        follow_up.append({"type": "text", "text": f"\n\n{message}"})
                    else:
                        follow_up.append({"role": "assistant", "content": raw or ""})
                        follow_up.append({"role": "user", "content": message})
                    time.sleep(self.settings.retry_delay_seconds * attempt)
            except Exception as exc:
                last_error = exc
                log.warning("  %s attempt %d/%d: API error - %s", model, attempt, self.settings.max_attempts, exc)
                if attempt < self.settings.max_attempts:
                    time.sleep(self.settings.retry_delay_seconds * attempt)

        log.error("  %s failed after %d attempts: %s", model, self.settings.max_attempts, last_error)
        return {}

    def _call(self, system: str, content: PaperContent, filename: str, follow_up: list[dict]) -> str:
        settings = self.settings
        if self.anthropic:
            response = self.client.messages.create(
                model=settings.model,
                max_tokens=settings.max_tokens,
                temperature=settings.temperature,
                system=system,
                messages=[{"role": "user", "content": prompts.anthropic_user_content(content, filename) + follow_up}],
            )
            self.usage.add_anthropic(STAGE, settings.model, getattr(response, "usage", None))
            return response.content[0].text
        # Earlier parse errors go before the paper, as in the run that built the paper's graph.
        messages = [
            {"role": "system", "content": system},
            *follow_up,
            {"role": "user", "content": prompts.openai_user_content(content, filename)},
        ]
        kwargs = llm_client.chat_kwargs(
            settings.model,
            messages,
            max_tokens=settings.max_tokens,
            temperature=settings.temperature,
            min_gpt5_completion_tokens=settings.min_gpt5_completion_tokens,
        )
        return llm_client.complete(self.client, kwargs, self.usage, STAGE).choices[0].message.content

    def _repair(self, raw: str, error: Exception, pdf_path: Path, paper_type: str) -> dict[str, Any] | None:
        """Ask the model to fix only the JSON syntax; None if that fails too."""
        messages = [
            {"role": "system", "content": prompts.JSON_REPAIR_SYSTEM},
            {"role": "user", "content": prompts.json_repair_prompt(raw, error)},
        ]
        kwargs = llm_client.chat_kwargs(
            self.settings.model,
            messages,
            max_tokens=self.settings.max_tokens,
            temperature=0,
            min_gpt5_completion_tokens=self.settings.min_gpt5_completion_tokens,
        )
        try:
            reply = llm_client.complete(self.client, kwargs, self.usage, STAGE).choices[0].message.content
            record = normalize_paper_record(parse_json_reply(reply), pdf_path, paper_type, self.settings.model)
        except Exception as exc:
            log.warning("  %s JSON repair failed: %s", self.settings.model, exc)
            return None
        record.setdefault("extraction_warnings", [])
        record["extraction_warnings"].append(
            f"Initial model output required JSON syntax repair after parse error: {error}"
        )
        log.info("  %s -> repaired JSON and extracted %d case record(s)", self.settings.model, len(paper_cases(record)))
        return record


def run(ctx: BuildContext) -> dict[str, Any]:
    settings = ctx.profile.extraction
    out_dir = ctx.layout.kg_cases
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = ctx.selected_pdfs()
    log.info("Extracting %d paper(s) with %s into %s", len(pdfs), settings.model, out_dir)

    extractor: Extractor | None = None
    papers: list[dict] = []
    stats = {"papers": len(pdfs), "extracted": 0, "reused": 0, "failed": 0, "cases": 0, "rules": 0}
    for number, pdf_path in enumerate(pdfs, 1):
        out_path = out_dir / f"{paper_slug(pdf_path)}.json"
        if out_path.exists() and not ctx.force:
            log.info("[%d/%d] Reusing %s", number, len(pdfs), out_path.name)
            record = read_json(out_path)
            stats["reused"] += 1
        else:
            log.info("[%d/%d] Processing: %s", number, len(pdfs), pdf_path.name)
            extractor = extractor or Extractor(settings, ctx.record.usage)
            started = time.perf_counter()
            record = extractor.extract(pdf_path)
            if not paper_cases(record):
                log.error("  No cases extracted from %s", pdf_path.name)
                stats["failed"] += 1
                continue
            write_json(out_path, record)
            stats["extracted"] += 1
            log.info(
                "  Saved %d cases to %s (%.1fs)", len(paper_cases(record)), out_path.name, time.perf_counter() - started
            )
            if number < len(pdfs):
                time.sleep(settings.pause_between_papers_seconds)
        if ctx.profile.record_ids.paper_scoped_ids:
            record = {**record, "source_pdf": {"file": pdf_path.name, "sha256": sha256_file(pdf_path)}}
        papers.append(record)
        stats["cases"] += len(paper_cases(record))
        stats["rules"] += len(record.get("kg_ready_rules") or [])

    write_json(ctx.layout.extracted_cases, [case for paper in papers for case in paper_cases(paper)])
    write_json(ctx.layout.extracted_papers, {"papers": papers})
    if stats["failed"]:
        ctx.fail_or_warn(f"extract: no cases from {stats['failed']} of {len(pdfs)} paper(s)")
    log.info("Extracted %(extracted)d, reused %(reused)d, failed %(failed)d; %(cases)d cases, %(rules)d rules", stats)
    return stats
