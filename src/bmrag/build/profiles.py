"""Build profiles: every setting that changes what the build produces, in one place.

PAPER_V1 holds the settings the existing stores and graph were built with. The regression
tests in tests/test_build_pipeline.py check that it still produces byte-identical output, so
change a setting by adding a new profile, never by editing PAPER_V1.

Settings that only change how a run proceeds (worker count, which papers, resuming) are
command-line options instead; see bmrag.build.__main__.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


def _hi_res_strategies() -> tuple[tuple[str, dict[str, Any]], ...]:
    return (
        (
            "hi_res (full)",
            {
                "strategy": "hi_res",
                "infer_table_structure": True,
                "extract_image_block_types": ["Image"],
                "extract_image_block_to_payload": True,
            },
        ),
        ("fast (no tables)", {"strategy": "fast"}),
        ("ocr_only", {"strategy": "ocr_only"}),
    )


@dataclass(frozen=True)
class ChunkingSettings:
    """Paper index step 1: unstructured.partition_pdf. Strategies are tried in order until one succeeds."""

    strategies: tuple[tuple[str, dict[str, Any]], ...] = field(default_factory=_hi_res_strategies)
    chunking_strategy: str = "by_title"
    max_characters: int = 10000
    combine_text_under_n_chars: int = 2000
    new_after_n_chars: int = 6000
    math_threads_per_worker: int = 8

    def partition_options(self) -> dict[str, Any]:
        return {
            "chunking_strategy": self.chunking_strategy,
            "max_characters": self.max_characters,
            "combine_text_under_n_chars": self.combine_text_under_n_chars,
            "new_after_n_chars": self.new_after_n_chars,
        }


@dataclass(frozen=True)
class ReferenceCleaningSettings:
    """Paper index step 2. A references heading only counts within the last end_threshold chunks."""

    end_threshold: int = 4
    always_drop_sections: tuple[str, ...] = (
        "acknowledgment",
        "acknowledgement",
        "acknowledgements",
        "bibliography",
        "credit authorship contribution statement",
        "author contribution",
        "declaration of competing interest",
        "conflict of interest",
        "data availability",
        "appendix",
        "supplementary material",
    )
    end_only_sections: tuple[str, ...] = ("reference", "references")
    max_header_words: int = 5


@dataclass(frozen=True)
class ImageFilterSettings:
    """Paper index step 3. Images failing any limit are kept in the manifest but not embedded."""

    min_page_ratio: float = 0.01  # share of the page area
    min_width: float = 50.0  # coordinate units
    min_height: float = 50.0
    max_aspect_ratio: float = 10.0  # width/height above this, or below its inverse, is filtered


@dataclass(frozen=True)
class SummarySettings:
    """Paper index step 4: one LLM summary per element, then embeddings of the summaries."""

    model: str = "gpt-5-mini"
    text_prompt: str = "paper_summary_text"
    image_prompt: str = "paper_summary_image"
    concurrency: int = 50
    batch_size: int = 10
    text_max_retries: int | None = 0  # None keeps the client default
    image_max_retries: int | None = None
    embedding_batch_size: int = 500


@dataclass(frozen=True)
class ExtractionSettings:
    """Graph step 0: one multimodal LLM call per paper returns its cases and rules."""

    model: str = "gpt-5-mini"
    prompt: str = "kg_case_extraction_research"
    temperature: float = 0.1  # not sent to gpt-5 models, which reject it
    max_tokens: int = 24000
    min_gpt5_completion_tokens: int = 16000
    max_attempts: int = 3
    retry_delay_seconds: float = 5.0  # multiplied by the attempt number
    pause_between_papers_seconds: float = 2.0
    text_char_limit: int = 120_000  # longer texts lose their references section
    image_scale: float = 1.5  # page images at about 150 DPI
    image_page_keywords: tuple[str, ...] = (
        "fig",
        "figure",
        "wall",
        "layer",
        "configuration",
        "cross-section",
        "cross section",
        "scheme",
        "schematic",
        "setup",
        "model",
        "structure",
        "composition",
        "assembly",
        "detail",
        "diagram",
        "section",
    )


@dataclass(frozen=True)
class StandardizeSettings:
    """Graph step 6. Entity names are always aligned by rules; use_llm adds an LLM pass."""

    use_llm: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_max_entities: int = 600
    llm_batch_size: int = 80
    llm_max_tokens: int = 4096
    min_gpt5_completion_tokens: int = 4096


@dataclass(frozen=True)
class RecordIdSettings:
    """Graph stage "entities": how case and rule ids are formed.

    The extraction model numbers cases and rules per paper, so two papers can both return "rule_1".
    The stores and the graph key records by id alone and keep only one of them. paper_scoped_ids
    appends the first 8 hex digits of the paper's PDF SHA-256, "rule_1@3f2a9c1e", so every id is
    unique and stays the same when the PDF is renamed or extracted again.

    The judge decides from the start of a cited id whether it names a case ("case...") or a rule
    ("rule..." or "Rule<digit>"), and does not check ids that match neither. typed_prefixes puts
    "case_" or "rule_" in front of any id that lacks it, so "R1" becomes "case_R1" or "rule_R1".
    """

    paper_scoped_ids: bool = False
    typed_prefixes: bool = False


@dataclass(frozen=True)
class BuildProfile:
    name: str
    description: str
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    reference_cleaning: ReferenceCleaningSettings = field(default_factory=ReferenceCleaningSettings)
    image_filter: ImageFilterSettings = field(default_factory=ImageFilterSettings)
    summaries: SummarySettings = field(default_factory=SummarySettings)
    extraction: ExtractionSettings = field(default_factory=ExtractionSettings)
    standardize: StandardizeSettings = field(default_factory=StandardizeSettings)
    record_ids: RecordIdSettings = field(default_factory=RecordIdSettings)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


PAPER_V1 = BuildProfile(
    name="paper_v1",
    description="Settings of the stores and graph used in the paper (built 2026; one extraction model).",
)

PAPER_V1_UNIQUE_IDS = dataclasses.replace(
    PAPER_V1,
    name="paper_v1_unique_ids",
    description=(
        "paper_v1 with case and rule ids that are unique across papers and that the judge can "
        "classify. Gives a different graph and stores from the paper's."
    ),
    record_ids=RecordIdSettings(paper_scoped_ids=True, typed_prefixes=True),
)

PROFILES: dict[str, BuildProfile] = {profile.name: profile for profile in (PAPER_V1, PAPER_V1_UNIQUE_IDS)}

# What `python -m bmrag.build` uses without --profile. Data built from new PDFs cannot reproduce the
# paper's graph anyway, and with paper_v1 the judge cannot check ids such as "C1" or "R1" (found in
# the 2026-11-10 smoke run), so new builds get ids it can route. Choose paper_v1 to compare with the paper.
DEFAULT_PROFILE = PAPER_V1_UNIQUE_IDS.name


def get_profile(name: str) -> BuildProfile:
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown build profile {name!r}; known: {', '.join(PROFILES)}") from None


def with_overrides(profile: BuildProfile, overrides: list[str]) -> BuildProfile:
    """Apply "section.field=value" overrides, e.g. "standardize.use_llm=true".

    The result is renamed "<name>+overrides" so a run record never presents it as the original profile.
    """
    if not overrides:
        return profile
    sections: dict[str, dict[str, Any]] = {}
    for item in overrides:
        key, sep, raw = item.partition("=")
        section, dot, name = key.strip().partition(".")
        if not sep or not dot:
            raise ValueError(f"Override must look like section.field=value: {item!r}")
        current = getattr(profile, section, None)
        if not dataclasses.is_dataclass(current):
            raise ValueError(f"Unknown profile section {section!r}")
        fields = {f.name: f for f in dataclasses.fields(current)}
        if name not in fields:
            raise ValueError(f"Unknown setting {section}.{name}")
        sections.setdefault(section, {})[name] = _coerce(raw.strip(), getattr(current, name), key)
    changed = {
        section: dataclasses.replace(getattr(profile, section), **values) for section, values in sections.items()
    }
    return dataclasses.replace(profile, name=f"{profile.name}+overrides", **changed)


def _coerce(raw: str, current: Any, key: str) -> Any:
    if isinstance(current, bool):
        if raw.lower() not in {"true", "false", "1", "0", "yes", "no"}:
            raise ValueError(f"{key} needs true or false, got {raw!r}")
        return raw.lower() in {"true", "1", "yes"}
    if isinstance(current, int) or current is None and raw.isdigit():
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, str):
        return raw
    raise ValueError(f"{key} cannot be overridden from the command line")
