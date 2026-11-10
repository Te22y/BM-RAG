"""Record the build pipeline baseline from the code before the DP1-DP4 refactor (2026-11-07).

Works only with the old module layout (bmrag.indexing.*, bmrag.knowledge_graph.step*). It is
kept so the baseline can be audited: check out the commit before the refactor and run it.
Makes no network call. Writes tests/golden/build_baseline.json (synthetic paper) and, when the
local trial paper and its recorded extraction exist, tests/golden/build_baseline_local.json.

Usage (from the repo root):
  .venv/Scripts/python.exe -X utf8 tests/golden/record_build_baseline.py [--check]

--check runs everything again and reports differences instead of writing, to show the
baseline is reproducible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tests import build_baseline as bb  # noqa: E402
from tests import build_fixtures as fx  # noqa: E402

GOLDEN = REPO / "tests" / "golden"
WORK = REPO / "runs" / "build_baseline_work"
LOCAL_PDF = (
    REPO
    / "data"
    / "papers"
    / "A numerical study of external building walls containing phase change materials (PCM).pdf"
)
LOCAL_REPLY = (
    REPO
    / "runs"
    / "build_trial_20261107"
    / "data"
    / "kg_cases"
    / "model_compare"
    / "A_numerical_study_of_external_building_walls_containing_phas.gpt-5-mini.raw.json"
)

OLD_STEPS = [
    bb.Step("A1_chunk", "bmrag.indexing.pdf_chunking", ("--workers", "1")),
    bb.Step("A2_clean", "bmrag.indexing.reference_cleaning"),
    bb.Step("A3_separate", "bmrag.indexing.element_separation"),
    bb.Step("A4_summarize", "bmrag.indexing.paper_summaries", stand_in=True),
    bb.Step("B0_extract", "bmrag.knowledge_graph.step0_extract_cases", stand_in=True),
    bb.Step(
        "B2_entities",
        "bmrag.knowledge_graph.step2_extract_entities_relations",
        ("--input", "{data}/kg_cases/ALL_papers.json"),
    ),
    # The old quality check skips ALL_* files, so it reads a per-paper copy made in folder mode.
    bb.Step(
        "B2_entities_folder",
        "bmrag.knowledge_graph.step2_extract_entities_relations",
        ("--input", "{data}/kg_cases", "--output", "{data}/kg_entities_folder_mode"),
    ),
    bb.Step(
        "B2_quality",
        "bmrag.knowledge_graph.step2_check_quality",
        (
            "--input",
            "{data}/kg_entities_folder_mode",
            "--output",
            "{data}/kg_entities_folder_mode/step2_quality_report.json",
        ),
    ),
    bb.Step(
        "B5_cypher",
        "bmrag.knowledge_graph.step5_generate_cypher",
        ("--input", "{data}/kg_entities/ALL_papers_structured.json"),
    ),
    bb.Step("B6_standardize", "bmrag.knowledge_graph.step6_standardize"),
    bb.Step(
        "B6_standardize_llm",
        "bmrag.knowledge_graph.step6_standardize",
        ("--use-llm", "--output", "{data}/kg_canonical_llm"),
        stand_in=True,
    ),
    bb.Step("B_kg_index", "bmrag.indexing.build_kg_index", stand_in=True),
    bb.Step("B7_dry_run", "bmrag.knowledge_graph.step7_import_neo4j", ("--dry-run", "--database", "bmragtest")),
]


def cases() -> list[tuple[bb.Case, Path]]:
    synthetic_pdf = fx.write_synthetic_pdf(WORK / "inputs" / fx.SYNTHETIC_PDF_NAME)
    found = [
        (
            bb.Case("synthetic", {fx.SYNTHETIC_PDF_NAME: synthetic_pdf}, {fx.SYNTHETIC_PDF_NAME: fx.SYNTHETIC_REPLY}),
            GOLDEN / "build_baseline.json",
        )
    ]
    if LOCAL_PDF.exists() and LOCAL_REPLY.exists():
        found.append(
            (
                bb.Case(
                    "local_paper",
                    {LOCAL_PDF.name: LOCAL_PDF},
                    {LOCAL_PDF.name: LOCAL_REPLY.read_text(encoding="utf-8")},
                ),
                GOLDEN / "build_baseline_local.json",
            )
        )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="compare with the stored baseline instead of writing")
    args = parser.parse_args()
    if not bb.build_packages_available() or not bb.tool_path():
        raise SystemExit("Needs the build packages (requirements-build.lock.txt), Poppler and Tesseract")
    for case, target in cases():
        print(f"Running {case.name} ...", flush=True)
        result = bb.run_case(case, OLD_STEPS, WORK)
        if args.check:
            problems = bb.compare(json.loads(target.read_text(encoding="utf-8")), result)
            print(f"{case.name}: {'identical' if not problems else 'DIFFERENT'}")
            for line in problems:
                print("  " + line)
        else:
            target.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"{case.name}: {len(result['files'])} files, {sum(map(len, result['requests'].values()))} "
                f"requests -> {target.relative_to(REPO)}"
            )


if __name__ == "__main__":
    main()
