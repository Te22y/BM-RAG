"""Unit tests for the build pipeline (bmrag.build). Fast; no API, database or real paper needed.

Tests that need PyMuPDF or unstructured skip themselves when the package is missing.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib.util
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from bmrag.build import __main__ as cli
from bmrag.build import llm_client, stages
from bmrag.build.graph import entities, extract, extraction_prompt, neo4j_import, quality, standardize, to_cypher
from bmrag.build.graph.extraction_records import normalize_paper_record, paper_slug, parse_json_reply
from bmrag.build.graph.paper_content import PaperContent
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.layout import DataLayout
from bmrag.build.manifest import RunRecord
from bmrag.build.profiles import PAPER_V1, PAPER_V1_UNIQUE_IDS, get_profile, with_overrides
from bmrag.build.stores import records as store_records
from tests import build_fixtures as fx

HAS_UNSTRUCTURED = importlib.util.find_spec("unstructured") is not None


def make_ctx(root: Path, **options) -> stages.BuildContext:
    layout = DataLayout(root)
    record = RunRecord(layout, PAPER_V1, overrides=[], options={})
    return stages.BuildContext(layout=layout, profile=options.pop("profile", PAPER_V1), record=record, **options)


def fake_completion(content: str, model: str = "gpt-5-mini"):
    usage = types.SimpleNamespace(
        prompt_tokens=100,
        completion_tokens=20,
        prompt_tokens_details=types.SimpleNamespace(cached_tokens=10),
        completion_tokens_details=types.SimpleNamespace(reasoning_tokens=5),
    )
    message = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(model=model, usage=usage, choices=[types.SimpleNamespace(message=message)])


class FakeOpenAI:
    """Replies in order; an Exception in the list is raised instead of returned."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests: list[dict] = []
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return fake_completion(reply, kwargs["model"])


class ProfileTests(unittest.TestCase):
    def test_paper_v1_keeps_the_settings_of_the_paper_build(self):
        self.assertEqual(PAPER_V1.extraction.model, "gpt-5-mini")
        self.assertEqual((PAPER_V1.extraction.max_tokens, PAPER_V1.extraction.temperature), (24000, 0.1))
        self.assertEqual(PAPER_V1.extraction.prompt, "kg_case_extraction_research")
        self.assertEqual(PAPER_V1.summaries.model, "gpt-5-mini")
        self.assertEqual(PAPER_V1.chunking.strategies[0][0], "hi_res (full)")
        self.assertEqual(
            (
                PAPER_V1.chunking.max_characters,
                PAPER_V1.chunking.combine_text_under_n_chars,
                PAPER_V1.chunking.new_after_n_chars,
            ),
            (10000, 2000, 6000),
        )
        self.assertEqual(PAPER_V1.reference_cleaning.end_threshold, 4)
        self.assertEqual(PAPER_V1.image_filter.min_page_ratio, 0.01)
        self.assertFalse(PAPER_V1.standardize.use_llm)

    def test_overrides_are_typed_and_rename_the_profile(self):
        changed = with_overrides(
            PAPER_V1, ["standardize.use_llm=true", "summaries.batch_size=4", "extraction.model=claude-sonnet-4-6"]
        )
        self.assertTrue(changed.standardize.use_llm)
        self.assertEqual(changed.summaries.batch_size, 4)
        self.assertEqual(changed.extraction.model, "claude-sonnet-4-6")
        self.assertEqual(changed.name, "paper_v1+overrides")
        self.assertFalse(PAPER_V1.standardize.use_llm)
        self.assertIs(with_overrides(PAPER_V1, []), PAPER_V1)

    def test_bad_overrides_are_rejected(self):
        for item in ("use_llm=true", "nosuch.field=1", "standardize.nosuch=1", "standardize.use_llm=maybe"):
            with self.subTest(item=item), self.assertRaises(ValueError):
                with_overrides(PAPER_V1, [item])


class StageAndCliTests(unittest.TestCase):
    def test_targets_resolve_in_pipeline_order(self):
        self.assertEqual([s.name for s in stages.resolve(["all"])], list(stages.STAGE_NAMES))
        self.assertEqual([s.name for s in stages.resolve(["neo4j", "extract"])], ["extract", "neo4j"])
        self.assertTrue(all(s.group == stages.GRAPH for s in stages.resolve(["graph"])))
        with self.assertRaises(SystemExit):
            stages.resolve(["nosuch"])

    def test_every_stage_module_has_run(self):
        for stage in stages.STAGES:
            if stage.group == stages.PAPER_INDEX and not HAS_UNSTRUCTURED:
                continue
            with self.subTest(stage=stage.name):
                module = importlib.import_module(stage.module)
                self.assertTrue(callable(module.run))

    def test_cli_lists_stages_and_needs_a_target(self):
        with mock.patch("builtins.print") as printed:
            self.assertEqual(cli.main(["--list"]), 0)
        self.assertIn("standardize", printed.call_args.args[0])
        with mock.patch("sys.stderr"):
            self.assertEqual(cli.main([]), 2)

    def test_cli_defaults_to_judge_routable_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout = DataLayout(Path(tmp))
            paper = {**fx.SYNTHETIC_RECORD, "source_pdf": {"file": "a.pdf", "sha256": "abcd1234" + "0" * 56}}
            write_json(layout.extracted_papers, {"papers": [paper]})
            rule_ids = {}
            for args in ([], ["--profile", "paper_v1"]):
                self.assertEqual(cli.main(["entities", "--data", tmp, *args]), 0)
                records = read_json(layout.entity_records)
                rule_ids[tuple(args)] = next(r["rule_id"] for r in records if r["record_type"] == "rule")
            self.assertEqual(rule_ids[()], "rule_R1@abcd1234")
            self.assertEqual(rule_ids[("--profile", "paper_v1")], "R1")

    def test_pdf_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            ctx.layout.papers.mkdir()
            for name in ("b.pdf", "A.PDF", "notes.txt"):
                (ctx.layout.papers / name).write_bytes(b"%PDF-1.4")
            self.assertEqual([p.name for p in ctx.selected_pdfs()], ["A.PDF", "b.pdf"])
            ctx.limit = 1
            self.assertEqual([p.name for p in ctx.selected_pdfs()], ["A.PDF"])
            ctx.paper_names = ["missing.pdf"]
            with self.assertRaises(SystemExit):
                ctx.selected_pdfs()
            self.assertEqual(len(ctx.record.data["inputs"]), 2)

    def test_fail_or_warn(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            ctx.fail_or_warn("problem")
            self.assertEqual(ctx.record.data["warnings"], ["problem"])
            ctx.strict = True
            with self.assertRaises(SystemExit):
                ctx.fail_or_warn("problem")


class LlmClientTests(unittest.TestCase):
    def test_chat_kwargs_per_model_family(self):
        messages = [{"role": "user", "content": "x"}]
        gpt5 = llm_client.chat_kwargs(
            "gpt-5-mini", messages, max_tokens=4096, temperature=0.1, min_gpt5_completion_tokens=16000
        )
        self.assertEqual(gpt5["max_completion_tokens"], 16000)
        self.assertNotIn("temperature", gpt5)
        other = llm_client.chat_kwargs(
            "gpt-4o-mini", messages, max_tokens=4096, temperature=0, min_gpt5_completion_tokens=4096
        )
        self.assertEqual((other["max_tokens"], other["temperature"]), (4096, 0))
        self.assertEqual(other["response_format"], {"type": "json_object"})

    def test_complete_renames_a_rejected_argument_and_records_usage(self):
        client = FakeOpenAI([RuntimeError("Unsupported parameter: 'max_tokens'"), "{}"])
        usage = llm_client.UsageLog()
        kwargs = {"model": "gpt-5-mini", "messages": [], "max_tokens": 10}
        llm_client.complete(client, kwargs, usage, "stage")
        self.assertEqual(client.requests[1]["max_completion_tokens"], 10)
        self.assertNotIn("max_tokens", client.requests[1])
        self.assertEqual(kwargs["max_tokens"], 10)  # the caller's dict is left alone
        row = usage.summary()["chat"][0]
        self.assertEqual((row["calls"], row["input_tokens"], row["cached_tokens"]), (1, 100, 10))
        self.assertAlmostEqual(row["estimated_usd"], (90 * 0.25 + 10 * 0.025 + 20 * 2.0) / 1e6, places=6)

    def test_complete_reraises_other_errors(self):
        client = FakeOpenAI([RuntimeError("rate limited")])
        with self.assertRaises(RuntimeError):
            llm_client.complete(client, {"model": "gpt-5-mini", "messages": []})

    def test_unpriced_models_are_listed(self):
        usage = llm_client.UsageLog()
        usage.add_chat("s", "gpt-4o-mini", 10, 0, 1)
        self.assertEqual(usage.summary()["unpriced_models"], ["gpt-4o-mini"])
        self.assertIsNone(usage.summary()["estimated_chat_usd"])


class ExtractionPromptTests(unittest.TestCase):
    def test_markdown_only_rules_are_dropped(self):
        prompt = "Keep this criterion.\nDo not output JSON.\n# Case summary\nAlso keep this."
        self.assertEqual(extraction_prompt.sanitize_task_prompt(prompt), "Keep this criterion.\nAlso keep this.")

    def test_paper_type_hint(self):
        hint = extraction_prompt.infer_paper_type
        self.assertEqual(hint("", "A numerical study of walls.pdf"), "original")
        self.assertEqual(hint("", "A review of PCM walls.pdf"), "review")
        self.assertEqual(hint("Abstract: a meta-analysis of studies", "x.pdf"), "meta_analysis")
        review_abstract = "Abstract\nThis review covers a literature review of walls.\nKeywords: PCM"
        self.assertEqual(hint(review_abstract, "PCM walls.pdf"), "review")
        self.assertEqual(hint("Abstract\nThis study presents a model.\n1. Introduction", "PCM walls.pdf"), "original")

    def test_system_prompt_wraps_the_criteria(self):
        text = extraction_prompt.system_prompt(PAPER_V1.extraction.prompt, "review")
        self.assertIn("Detected paper type hint from filename/text: review.", text)
        self.assertIn('"kg_ready_rules": [', text)
        self.assertTrue(text.startswith("You are running PCM KG Step0."))

    def test_user_content_layout(self):
        content = PaperContent("full text", "| a |", [(2, "QUJD")])
        parts = extraction_prompt.openai_user_content(content, "p.pdf")
        self.assertTrue(parts[0]["text"].startswith("SOURCE FILE: p.pdf\n\n=== FULL PAPER TEXT ===\nfull text"))
        self.assertIn("=== EXTRACTED TABLES ===", parts[0]["text"])
        self.assertEqual(parts[1]["image_url"]["url"], "data:image/png;base64,QUJD")
        self.assertEqual(parts[2]["text"], "[Image above: Page 2 of p.pdf]")
        self.assertEqual(parts[-1]["text"], extraction_prompt.FINAL_INSTRUCTION)
        anthropic = extraction_prompt.anthropic_user_content(content, "p.pdf")
        self.assertEqual(anthropic[1]["source"]["data"], "QUJD")


class ExtractionRecordTests(unittest.TestCase):
    def test_parse_reply(self):
        self.assertEqual(parse_json_reply('```json\n{"cases": []}\n```'), {"cases": []})
        self.assertEqual(parse_json_reply('["a", "b"]'), {"cases": ["a", "b"]})
        # Kept from the paper build: braces win over brackets, so an array of objects yields its object span.
        self.assertEqual(parse_json_reply('[{"case_id": "c"}]'), {"case_id": "c"})
        for bad in ("no json here", '{"cases": [}'):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                parse_json_reply(bad)

    def test_normalize_fills_provenance(self):
        pdf = Path("My paper (2020).pdf")
        record = normalize_paper_record({"cases": [{}, {"case_id": "kept"}]}, pdf, "original", "gpt-5-mini")
        self.assertEqual(paper_slug(pdf), "My_paper__2020_")
        self.assertEqual([c["case_id"] for c in record["cases"]], ["My_paper__2020__case_1", "kept"])
        self.assertEqual(record["paper_type"], "research_paper")
        self.assertEqual(record["paper_metadata"]["source_file"], "My paper (2020).pdf")
        self.assertEqual(record["cases"][0]["extraction_model"], "gpt-5-mini")


class ExtractorTests(unittest.TestCase):
    def _extractor(self, client: FakeOpenAI) -> extract.Extractor:
        with mock.patch.object(llm_client, "openai_client", return_value=client):
            return extract.Extractor(PAPER_V1.extraction, llm_client.UsageLog())

    def _run(self, replies) -> tuple[dict, FakeOpenAI]:
        client = FakeOpenAI(replies)
        extractor = self._extractor(client)
        content = PaperContent("Abstract\nThis study presents a wall.\n", "", [])
        with mock.patch.object(extract, "read_paper", return_value=content), mock.patch.object(extract.time, "sleep"):
            return extractor.extract(Path(fx.SYNTHETIC_PDF_NAME)), client

    def test_valid_reply(self):
        record, client = self._run([fx.SYNTHETIC_REPLY])
        self.assertEqual(len(record["cases"]), 2)
        request = client.requests[0]
        self.assertEqual(request["max_completion_tokens"], 24000)
        self.assertEqual([m["role"] for m in request["messages"]], ["system", "user"])

    def test_invalid_json_is_repaired_by_the_model(self):
        record, client = self._run(['{"cases": [', fx.SYNTHETIC_REPLY])
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(client.requests[1]["messages"][0]["content"], extraction_prompt.JSON_REPAIR_SYSTEM)
        self.assertNotIn("max_tokens", client.requests[1])
        self.assertIn("required JSON syntax repair", record["extraction_warnings"][-1])

    def test_failed_repair_retries_with_the_error_before_the_paper(self):
        record, client = self._run(['{"cases": [', "still not json", fx.SYNTHETIC_REPLY])
        self.assertEqual(len(record["cases"]), 2)
        roles = [m["role"] for m in client.requests[2]["messages"]]
        self.assertEqual(roles, ["system", "assistant", "user", "user"])

    def test_api_errors_give_up_after_max_attempts(self):
        record, client = self._run([RuntimeError("boom")] * 3)
        self.assertEqual(record, {})
        self.assertEqual(len(client.requests), 3)

    def test_existing_records_are_reused_without_a_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            ctx.layout.papers.mkdir()
            pdf = ctx.layout.papers / fx.SYNTHETIC_PDF_NAME
            pdf.write_bytes(b"%PDF-1.4")
            write_json(ctx.layout.kg_cases / f"{paper_slug(pdf)}.json", fx.SYNTHETIC_RECORD)
            with mock.patch.object(extract, "Extractor", side_effect=AssertionError("no LLM expected")):
                stats = extract.run(ctx)
            self.assertEqual((stats["reused"], stats["cases"], stats["rules"]), (1, 2, 1))
            self.assertEqual(
                read_json(ctx.layout.extracted_papers)["papers"][0]["paper_metadata"]["paper_id"],
                "synthetic_pcm_wall_2026",
            )
            self.assertEqual(len(read_json(ctx.layout.extracted_cases)), 2)
            self.assertNotIn("source_pdf", read_json(ctx.layout.extracted_papers)["papers"][0])

    def test_unique_ids_profile_records_the_pdf_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), profile=PAPER_V1_UNIQUE_IDS)
            ctx.layout.papers.mkdir()
            pdf = ctx.layout.papers / fx.SYNTHETIC_PDF_NAME
            pdf.write_bytes(b"%PDF-1.4")
            per_paper = ctx.layout.kg_cases / f"{paper_slug(pdf)}.json"
            write_json(per_paper, fx.SYNTHETIC_RECORD)
            with mock.patch.object(extract, "Extractor", side_effect=AssertionError("no LLM expected")):
                extract.run(ctx)
            source = read_json(ctx.layout.extracted_papers)["papers"][0]["source_pdf"]
            self.assertEqual(source, {"file": fx.SYNTHETIC_PDF_NAME, "sha256": hashlib.sha256(b"%PDF-1.4").hexdigest()})
            self.assertNotIn("source_pdf", read_json(per_paper))  # the model's record is left as it was


def write_extracted(ctx: stages.BuildContext, papers: list[dict]) -> None:
    write_json(ctx.layout.extracted_papers, {"papers": papers})


class GraphStageTests(unittest.TestCase):
    def test_synthetic_paper_through_entities_to_neo4j_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), strict=True)
            write_extracted(ctx, [fx.SYNTHETIC_RECORD])
            self.assertEqual(entities.run(ctx), {"cases": 2, "rules": 1, "duplicate_ids": 0})
            self.assertEqual(quality.run(ctx)["error"], 0)
            self.assertEqual(to_cypher.run(ctx), {"records": 3})
            self.assertEqual(standardize.run(ctx)["errors"], 0)
            stats = neo4j_import.run(ctx)
            self.assertIsNone(stats["imported_into"])
            report = read_json(ctx.layout.neo4j_report)
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["error_count"], 0)
            cypher = ctx.layout.standardized_cypher.read_text(encoding="utf-8")
            self.assertIn("MERGE (record:Case {case_id: 'case_a_interior'})", cypher)

            records = read_json(ctx.layout.standardized_records)
            case_store = store_records.build_case_store(records)
            rule_store = store_records.build_rule_store(records)
            self.assertEqual(sorted(case_store), ["case_a_interior", "case_b_exterior"])
            self.assertEqual(list(rule_store), ["R1"])
            self.assertIn("climate", case_store["case_a_interior"]["retrieval_text"])

    def test_duplicate_ids_across_papers_are_reported(self):
        other = copy.deepcopy(fx.SYNTHETIC_RECORD)
        other["paper_metadata"] = {"paper_id": "other", "paper_title": "Other", "source_file": "Other.pdf"}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            write_extracted(ctx, [fx.SYNTHETIC_RECORD, other])
            stats = entities.run(ctx)
            self.assertEqual(stats["duplicate_ids"], 3)  # two cases and the rule
            self.assertIn("case:case_a_interior", ctx.record.data["warnings"][0])
            ctx.strict = True
            with self.assertRaises(SystemExit):
                entities.run(ctx)

    def test_paper_scoped_ids_keep_papers_apart(self):
        first = {**fx.SYNTHETIC_RECORD, "source_pdf": {"file": "a.pdf", "sha256": "aaaa1111" + "0" * 56}}
        # same model output, even the same paper_id, but another PDF
        other = {**fx.SYNTHETIC_RECORD, "source_pdf": {"file": "b.pdf", "sha256": "bbbb2222" + "0" * 56}}
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), strict=True, profile=PAPER_V1_UNIQUE_IDS)
            write_extracted(ctx, [first, other])
            self.assertEqual(entities.run(ctx), {"cases": 4, "rules": 2, "duplicate_ids": 0})
            records = read_json(ctx.layout.entity_records)
            key = "aaaa1111"
            first_rule = next(r for r in records if r["record_type"] == "rule")
            self.assertEqual(first_rule["rule_id"], f"rule_R1@{key}")
            self.assertEqual(first_rule["kg_rule"]["local_rule_id"], "R1")
            # the rule still points at its own paper's cases, under their new ids
            self.assertTrue(first_rule["kg_rule"]["evidence_case_ids"])
            self.assertTrue(all(cid.endswith(f"@{key}") for cid in first_rule["kg_rule"]["evidence_case_ids"]))
            to_cypher.run(ctx)
            standardize.run(ctx)
            rule_store = store_records.build_rule_store(read_json(ctx.layout.standardized_records))
            self.assertEqual(len(rule_store), 2)

    def test_renamed_ids_are_ones_the_judge_routes(self):
        from bmrag.agents import judge

        paper = {
            "paper_metadata": {"paper_id": "p", "paper_title": "A Paper"},
            "source_pdf": {"file": "p.pdf", "sha256": "0123456789abcdef"},
            "cases": [{"case_id": "case_01"}, {"case_id": "R1"}, {"case_id": "baseline_no_PCM"}, {}],
            "kg_ready_rules": [
                {"rule_id": "R1", "evidence_case_ids": ["case_01", "R1", "case_99"]},
                {"rule_id": "Rule2"},
                {},
            ],
        }
        settings = PAPER_V1_UNIQUE_IDS.record_ids
        renamed = entities.rename_ids(paper, Path("p.json"), settings)
        key = "01234567"
        case_ids = [c["case_id"] for c in renamed["cases"]]
        rule_ids = [r["rule_id"] for r in renamed["kg_ready_rules"]]
        self.assertEqual(
            case_ids, [f"case_01@{key}", f"case_R1@{key}", f"case_baseline_no_PCM@{key}", f"case_a_paper_case_4@{key}"]
        )
        self.assertEqual(rule_ids, [f"rule_R1@{key}", f"Rule2@{key}", f"rule_a_paper_rule_3@{key}"])
        self.assertTrue(all(judge._classify_evidence_id(cid) == "kg_case" for cid in case_ids))
        self.assertTrue(all(judge._classify_evidence_id(rid) == "kg_rule" for rid in rule_ids))
        # links follow their cases; an id naming no case of this paper is left as it is
        self.assertEqual(renamed["kg_ready_rules"][0]["evidence_case_ids"], [case_ids[0], case_ids[1], "case_99"])
        self.assertEqual(renamed["cases"][1]["local_case_id"], "R1")
        self.assertEqual(renamed["kg_ready_rules"][0]["local_rule_id"], "R1")
        self.assertEqual(paper["cases"][1]["case_id"], "R1")  # the input is not changed

    def test_typed_prefixes_without_paper_key(self):
        paper = {"cases": [{"case_id": "R1"}], "kg_ready_rules": [{"rule_id": "rule_1"}]}
        settings = dataclasses.replace(PAPER_V1.record_ids, typed_prefixes=True)
        renamed = entities.rename_ids(paper, Path("p.json"), settings)
        self.assertEqual(renamed["cases"][0]["case_id"], "case_R1")
        self.assertEqual(renamed["kg_ready_rules"][0]["rule_id"], "rule_1")

    def test_scoped_ids_need_the_pdf_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), profile=PAPER_V1_UNIQUE_IDS)
            write_extracted(ctx, [fx.SYNTHETIC_RECORD])  # extracted without the profile: no source_pdf
            with self.assertRaisesRegex(SystemExit, "no source_pdf hash"):
                entities.run(ctx)

    def test_id_prefix_patterns_match_the_judge(self):
        from bmrag.agents import judge

        self.assertEqual([p.pattern for p in judge._CASE_ID_PATTERNS], [entities.CASE_ID_START.pattern])
        self.assertEqual(
            "|".join(p.pattern.lstrip("^") for p in judge._RULE_ID_PATTERNS),
            entities.RULE_ID_START.pattern[len("^(?:") : -1],
        )

    def test_paper_v1_keeps_model_ids(self):
        self.assertEqual(get_profile("paper_v1").record_ids, entities.RecordIdSettings())
        unique = get_profile("paper_v1_unique_ids").record_ids
        self.assertTrue(unique.paper_scoped_ids and unique.typed_prefixes)

    def test_rule_integrity_findings_warn_unless_strict(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            write_extracted(ctx, [fx.SYNTHETIC_RECORD])
            entities.run(ctx)
            to_cypher.run(ctx)
            standardize.run(ctx)
            # a FAVORS rule whose target the extraction did not make a WallPosition, as in the paper's data
            records = read_json(ctx.layout.standardized_records)
            rule = next(r for r in records if r["record_type"] == "rule")
            rule["kg_rule"]["relation"] = "favors"
            rule["cypher"] = rule["cypher"].replace(":WallPosition", ":PerformanceOutcome")
            write_json(ctx.layout.standardized_records, records)

            stats = neo4j_import.run(ctx)  # validates only, and goes on
            self.assertIsNone(stats["imported_into"])
            self.assertIn("rule-integrity", ctx.record.data["warnings"][-1])
            self.assertEqual(read_json(ctx.layout.neo4j_report)["error_count"], 1)
            ctx.strict = True
            with self.assertRaises(SystemExit):
                neo4j_import.run(ctx)

    def test_destructive_words_inside_strings_are_not_cypher(self):
        context = {"source_file": "p.pdf", "case_id": "c"}
        text = "MERGE (n:PerformanceOutcome {name: 'initialized early to remove initial effects; delete nothing'})"
        self.assertEqual(neo4j_import.validate_statement(text, context, 0), [])
        errors = neo4j_import.validate_statement("MATCH (n:Case) REMOVE n.name", context, 0)
        self.assertEqual([e["error"] for e in errors], ["Potentially destructive Cypher is not allowed"])

    def test_invalid_cypher_stops_the_neo4j_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            ctx.layout.kg_canonical.mkdir(parents=True)
            ctx.layout.standardized_cypher.write_text("MATCH (n) DETACH DELETE n;", encoding="utf-8")
            with self.assertRaises(SystemExit):
                neo4j_import.run(ctx)


class FakeDriver:
    def __init__(self, nodes: int):
        self.nodes = nodes
        self.queries: list[str] = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute_query(self, query, **kwargs):
        return types.SimpleNamespace(records=[{"n": self.nodes}])

    def session(self, database):
        driver = self

        class Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def run(self, query):
                driver.queries.append(query)
                return types.SimpleNamespace(consume=lambda: None)

            def begin_transaction(self):
                session = self

                class Tx:
                    def __enter__(self):
                        return self

                    def __exit__(self, *exc):
                        return False

                    def run(self, query):
                        return session.run(query)

                    def commit(self):
                        pass

                return Tx()

        return Session()


class Neo4jImportTests(unittest.TestCase):
    def _import(self, nodes: int, **target):
        driver = FakeDriver(nodes)
        neo4j_target = stages.Neo4jTarget(database="trial", uri="neo4j://x", user="neo4j", **target)
        with mock.patch("neo4j.GraphDatabase.driver", return_value=driver):
            result = neo4j_import.import_to_neo4j(
                neo4j_target,
                "pw",
                [{"query": "MERGE (n:Paper {paper_id: 'p'})"}],
                ["CREATE CONSTRAINT c IF NOT EXISTS FOR (n:Paper) REQUIRE n.x IS UNIQUE"],
            )
        return result, driver

    def test_non_empty_database_is_refused(self):
        with self.assertRaises(SystemExit):
            self._import(5)

    def test_empty_database_is_imported(self):
        result, driver = self._import(0)
        self.assertEqual(result["nodes_before"], 0)
        self.assertNotIn("MATCH (n) DETACH DELETE n", driver.queries)
        self.assertIn("MERGE (n:Paper {paper_id: 'p'})", driver.queries)

    def test_clear_and_allow_non_empty(self):
        _, driver = self._import(5, clear=True)
        self.assertEqual(driver.queries[0], "MATCH (n) DETACH DELETE n")
        _, driver = self._import(5, allow_non_empty=True)
        self.assertNotIn("MATCH (n) DETACH DELETE n", driver.queries)

    def test_password_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp), neo4j=stages.Neo4jTarget("trial", "neo4j://x", "neo4j"))
            write_extracted(ctx, [fx.SYNTHETIC_RECORD])
            for stage in (entities, to_cypher, standardize):
                stage.run(ctx)
            with mock.patch.dict("os.environ", {"NEO4J_PASSWORD": ""}), self.assertRaises(SystemExit):
                neo4j_import.run(ctx)


class RunRecordTests(unittest.TestCase):
    def test_record_holds_profile_stages_and_usage_but_no_key(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-secret-123"}):
            ctx = make_ctx(Path(tmp))
            write_extracted(ctx, [fx.SYNTHETIC_RECORD])
            path = stages.run_stages(stages.resolve(["entities", "cypher"]), ctx)
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            self.assertNotIn("sk-secret-123", text)
            self.assertEqual([s["stage"] for s in data["stages"]], ["entities", "cypher"])
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["profile"]["extraction"]["model"], "gpt-5-mini")
            self.assertIn("python", data["environment"])
            self.assertEqual(
                {o["file"] for o in data["outputs"]},
                {"kg_entities/ALL_papers_structured.json", "kg_cypher/ALL_papers_structured_kg.json"},
            )

    def test_failed_stage_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            ctx = make_ctx(Path(tmp))
            with self.assertRaises(SystemExit):
                stages.run_stages(stages.resolve(["entities"]), ctx)
            data = read_json(next(ctx.layout.build_runs.glob("*.json")))
            self.assertEqual((data["status"], data["stages"][0]["status"]), ("failed", "failed"))


@unittest.skipUnless(HAS_UNSTRUCTURED, "needs unstructured")
class ReferenceCleaningTests(unittest.TestCase):
    def _chunk(self, *elements):
        from unstructured.documents.elements import CompositeElement

        chunk = CompositeElement(text="\n\n".join(e.text for e in elements))
        chunk.metadata.orig_elements = list(elements)
        return chunk

    def test_drop_rules(self):
        from unstructured.documents.elements import NarrativeText, Title

        from bmrag.build.paper_index.clean import clean_chunks

        settings = PAPER_V1.reference_cleaning
        body = self._chunk(Title("1. Introduction"), NarrativeText("Walls store heat. See the references below."))
        mid_ack = self._chunk(
            Title("Results"), NarrativeText("A result."), Title("Acknowledgements"), NarrativeText("Thanks.")
        )
        refs = self._chunk(Title("References"), NarrativeText("[1] A."))
        cleaned = clean_chunks([body, mid_ack, refs], settings)
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned[1].text, "Results\n\nA result.")
        # "References" far from the end is a normal heading
        early = clean_chunks([refs, body, body, body, body, body], settings)
        self.assertEqual(len(early), 6)

    def test_image_filter_reasons(self):
        from bmrag.build.paper_index.separate import filter_reason

        limits = PAPER_V1.image_filter
        size = {"width": 40.0, "height": 100.0, "page_ratio": 0.2, "aspect_ratio": 0.4}
        self.assertEqual(filter_reason(size, limits), "width=40.0 < 50.0")
        self.assertEqual(filter_reason(None, limits), "no_coordinates")
        self.assertEqual(filter_reason(dict(size, width=400.0, aspect_ratio=4.0), limits), "")
        self.assertEqual(dataclasses.asdict(limits)["max_aspect_ratio"], 10.0)


REPO = Path(__file__).resolve().parents[1]
# Code held to the style rules in pyproject.toml; the rest of the repository predates them.
STYLED_PATHS = [
    "src/bmrag/build",
    "tests/build_baseline.py",
    "tests/build_fixtures.py",
    "tests/build_standin.py",
    "tests/test_build_units.py",
    "tests/test_build_pipeline.py",
    "tests/golden/record_build_baseline.py",
]


@unittest.skipUnless(importlib.util.find_spec("ruff"), "needs ruff (pip install .[dev])")
class CodeStyleTests(unittest.TestCase):
    def test_build_code_passes_ruff(self):
        import subprocess
        import sys

        for command in (["check"], ["format", "--check"]):
            with self.subTest(command=command):
                result = subprocess.run(
                    [sys.executable, "-m", "ruff", *command, *STYLED_PATHS], cwd=REPO, capture_output=True, text=True
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
