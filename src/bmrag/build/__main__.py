"""Command line for the build pipeline.

Examples (from the repository root):
  python -m bmrag.build all
  python -m bmrag.build paper-index --workers 4
  python -m bmrag.build graph --paper "Some paper.pdf"
  python -m bmrag.build neo4j --neo4j-database bmragtrial
  python -m bmrag.build graph --set standardize.use_llm=true
  python -m bmrag.build --list

Each run writes a record to <data>/build_runs/. See docs/build_pipeline.md.
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing
import os
import sys
from pathlib import Path

from bmrag.build.layout import DataLayout
from bmrag.build.manifest import RunRecord
from bmrag.build.profiles import DEFAULT_PROFILE, PROFILES, get_profile, with_overrides
from bmrag.build.stages import GRAPH, PAPER_INDEX, STAGES, BuildContext, Neo4jTarget, resolve, run_stages
from bmrag.config import get_paths, load_environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m bmrag.build",
        description="Build the paper index, the knowledge graph and its index.",
        epilog="Targets: all, paper-index, graph, or stage names in any order; stages always run in pipeline order.",
    )
    parser.add_argument("targets", nargs="*", help=f"all | {PAPER_INDEX} | {GRAPH} | stage names")
    parser.add_argument("--list", action="store_true", help="list the stages and exit")
    parser.add_argument("--data", type=Path, default=None, help="data folder (default: BMRAG_DATA_DIR or data/)")
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE,
        choices=sorted(PROFILES),
        help=f"build profile (default {DEFAULT_PROFILE}; paper_v1 has the paper's settings)",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.FIELD=VALUE",
        help="override one profile setting; recorded in the run record",
    )
    parser.add_argument(
        "--paper",
        dest="papers",
        action="append",
        default=[],
        metavar="FILE",
        help="only this PDF in <data>/papers (repeatable)",
    )
    parser.add_argument("--limit", type=int, default=None, help="only the first N PDFs by name")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, max(1, multiprocessing.cpu_count() - 1)),
        help="processes for PDF chunking",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="redo work whose output exists (chunks, cleaned chunks, summaries, extractions, KG vectors)",
    )
    parser.add_argument("--strict", action="store_true", help="fail on validation problems instead of warning")
    neo4j = parser.add_argument_group("Neo4j import (stage neo4j)")
    neo4j.add_argument(
        "--neo4j-database", default=None, help="database to import into; without it the stage only validates"
    )
    neo4j.add_argument("--neo4j-uri", default=None, help="default: NEO4J_URI or neo4j://127.0.0.1:7687")
    neo4j.add_argument("--neo4j-user", default=None, help="default: NEO4J_USER or neo4j; password: NEO4J_PASSWORD")
    neo4j.add_argument(
        "--clear-database", action="store_true", help="delete everything in the target database before importing"
    )
    neo4j.add_argument(
        "--allow-non-empty",
        action="store_true",
        help="import into a database that already has nodes (MERGE updates them)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    return parser.parse_args(argv)


def list_stages() -> str:
    width = max(len(s.name) for s in STAGES)
    return "\n".join(f"{s.group:<12} {s.name:<{width}}  {s.summary}" for s in STAGES)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list:
        print(list_stages())
        return 0
    if not args.targets:
        print("Name what to build, e.g. `python -m bmrag.build all`. Stages:\n" + list_stages(), file=sys.stderr)
        return 2
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore", "openai", "chromadb", "pdfminer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    load_environment()
    # Paper text must not leave for a tracing service, whichever spelling .env uses to enable it.
    for variable in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING_V2", "LANGSMITH_TRACING"):
        os.environ[variable] = "false"

    stages = resolve(args.targets)
    layout = DataLayout(args.data.resolve() if args.data else get_paths().data_dir)
    profile = with_overrides(get_profile(args.profile), args.overrides)
    options = {
        "targets": args.targets,
        "papers": args.papers,
        "limit": args.limit,
        "workers": args.workers,
        "force": args.force,
        "strict": args.strict,
        "neo4j_database": args.neo4j_database,
        "clear_database": args.clear_database,
        "allow_non_empty": args.allow_non_empty,
    }
    record = RunRecord(layout, profile, overrides=args.overrides, options=options)
    target = None
    if args.neo4j_database:
        target = Neo4jTarget(
            database=args.neo4j_database,
            uri=args.neo4j_uri or os.getenv("NEO4J_URI", "neo4j://127.0.0.1:7687"),
            user=args.neo4j_user or os.getenv("NEO4J_USER", "neo4j"),
            clear=args.clear_database,
            allow_non_empty=args.allow_non_empty,
        )
    ctx = BuildContext(
        layout=layout,
        profile=profile,
        record=record,
        force=args.force,
        strict=args.strict,
        workers=args.workers,
        paper_names=args.papers,
        limit=args.limit,
        neo4j=target,
    )
    run_stages(stages, ctx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
