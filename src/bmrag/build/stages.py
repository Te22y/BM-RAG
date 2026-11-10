"""The build stages, in the order they run, and what each one needs.

A stage is a function run(ctx) -> dict in its own module; the dict goes into the run record.
Modules are imported only when their stage runs, so the graph stages work without the
paper-index packages (unstructured, torch) and the other way round.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bmrag.build.layout import DataLayout
from bmrag.build.manifest import RunRecord
from bmrag.build.profiles import BuildProfile

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Neo4jTarget:
    """Where the neo4j stage imports. Without one it only validates. The password comes from the environment."""

    database: str
    uri: str
    user: str
    clear: bool = False
    allow_non_empty: bool = False


@dataclass
class BuildContext:
    layout: DataLayout
    profile: BuildProfile
    record: RunRecord
    force: bool = False  # redo work whose output already exists
    strict: bool = False  # turn validation warnings into failures
    workers: int = 1  # PDF chunking processes
    paper_names: list[str] = field(default_factory=list)  # empty means every PDF in papers/
    limit: int | None = None
    neo4j: Neo4jTarget | None = None

    def selected_pdfs(self) -> list[Path]:
        """The PDFs this run processes, sorted by name."""
        folder = self.layout.papers
        if self.paper_names:
            pdfs = [folder / name for name in self.paper_names]
            missing = [p.name for p in pdfs if not p.is_file()]
            if missing:
                raise SystemExit(f"Not found in {folder}: {', '.join(missing)}")
        else:
            pdfs = (
                [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"] if folder.is_dir() else []
            )
        pdfs = sorted(pdfs)[: self.limit] if self.limit else sorted(pdfs)
        if not pdfs:
            raise SystemExit(f"No PDF files in {folder}")
        self.record.add_inputs(pdfs)
        return pdfs

    def fail_or_warn(self, message: str) -> None:
        """A validation problem: fatal with --strict, otherwise recorded as a warning."""
        if self.strict:
            raise SystemExit(message)
        self.record.warn(message)


@dataclass(frozen=True)
class Stage:
    name: str
    group: str
    summary: str
    module: str  # module with run(ctx) -> dict
    outputs: Callable[[DataLayout], list[Path]]  # main files, hashed into the run record

    def run(self, ctx: BuildContext) -> dict[str, Any]:
        return importlib.import_module(self.module).run(ctx)


PAPER_INDEX = "paper-index"
GRAPH = "graph"

STAGES: tuple[Stage, ...] = (
    Stage(
        "chunk",
        PAPER_INDEX,
        "partition each PDF and split it into title-based chunks",
        "bmrag.build.paper_index.chunk",
        lambda d: sorted(d.paper_chunks.glob("*.pkl")),
    ),
    Stage(
        "clean",
        PAPER_INDEX,
        "drop references and back matter",
        "bmrag.build.paper_index.clean",
        lambda d: sorted(d.paper_chunks_cleaned.glob("*.pkl")),
    ),
    Stage(
        "separate",
        PAPER_INDEX,
        "split text, tables and images; filter small images",
        "bmrag.build.paper_index.separate",
        lambda d: [d.paper_manifest],
    ),
    Stage(
        "summarize",
        PAPER_INDEX,
        "summarize every element with an LLM and embed the summaries",
        "bmrag.build.paper_index.summarize",
        lambda d: [d.paper_manifest],
    ),
    Stage(
        "extract",
        GRAPH,
        "extract cases and rules from each PDF with one LLM call",
        "bmrag.build.graph.extract",
        lambda d: [d.extracted_papers, d.extracted_cases],
    ),
    Stage(
        "entities",
        GRAPH,
        "normalize cases and rules into entities and relations",
        "bmrag.build.graph.entities",
        lambda d: [d.entity_records],
    ),
    Stage(
        "quality",
        GRAPH,
        "check required fields, labels and relations",
        "bmrag.build.graph.quality",
        lambda d: [d.quality_report],
    ),
    Stage(
        "cypher",
        GRAPH,
        "write the Cypher MERGE statements for every record",
        "bmrag.build.graph.to_cypher",
        lambda d: [d.cypher_records],
    ),
    Stage(
        "standardize",
        GRAPH,
        "align entity names and enforce the PCM ontology",
        "bmrag.build.graph.standardize",
        lambda d: [d.standardized_records, d.standardized_cypher, d.canonical_map, d.standardize_report],
    ),
    Stage(
        "kg-index",
        GRAPH,
        "embed cases and rules into the knowledge-graph index",
        "bmrag.build.stores.kg_index",
        lambda d: [d.case_store, d.rule_store],
    ),
    Stage(
        "neo4j",
        GRAPH,
        "validate the Cypher and, with --neo4j-database, import it",
        "bmrag.build.graph.neo4j_import",
        lambda d: [d.neo4j_report],
    ),
)
STAGE_NAMES = tuple(s.name for s in STAGES)
GROUPS = {PAPER_INDEX: [s for s in STAGES if s.group == PAPER_INDEX], GRAPH: [s for s in STAGES if s.group == GRAPH]}


def resolve(targets: list[str]) -> list[Stage]:
    """Stages for targets such as ["all"], ["graph"] or ["extract", "entities"], in pipeline order."""
    wanted: set[str] = set()
    for target in targets:
        if target == "all":
            wanted.update(STAGE_NAMES)
        elif target in GROUPS:
            wanted.update(s.name for s in GROUPS[target])
        elif target in STAGE_NAMES:
            wanted.add(target)
        else:
            raise SystemExit(
                f"Unknown target {target!r}; use all, {PAPER_INDEX}, {GRAPH} or a stage: {', '.join(STAGE_NAMES)}"
            )
    return [s for s in STAGES if s.name in wanted]


def run_stages(stages: list[Stage], ctx: BuildContext) -> Path:
    """Run the stages in order and write the run record, also when a stage fails."""
    outputs: list[Path] = []
    try:
        for stage in stages:
            with ctx.record.stage(stage.name) as entry:
                entry.update(stage.run(ctx) or {})
            outputs.extend(stage.outputs(ctx.layout))
    finally:
        path = ctx.record.write(outputs)
        log.info("Run record: %s", path)
    return path
