"""The refactored build pipeline must reproduce the baseline recorded from the code before it.

tests/golden/record_build_baseline.py recorded, with the old modules, every output file's digest,
the Chroma contents and the digest of every LLM request, for a synthetic paper and (locally) one
real paper. This test runs the same work through `python -m bmrag.build` and compares.

Slow (several minutes per paper) and needs the build packages, Poppler and Tesseract, so it only
runs when BMRAG_BUILD_TESTS=1. The real-paper case also needs the local data/ and runs/ folders.
No API is called: OpenAI is replaced by tests/build_standin.py.
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
from pathlib import Path

from tests import build_baseline as bb
from tests.golden import record_build_baseline as recorder

ENABLED = os.environ.get("BMRAG_BUILD_TESTS") == "1"
# The new pipeline also writes a run record, which the old one did not.
NEW_FILES = ["build_runs/*.json"]


def _keep_llm_variant(data: Path) -> None:
    """The old LLM-aligned standardize wrote to kg_canonical_llm/; the new one writes kg_canonical/."""
    shutil.move(data / "kg_canonical", data / "kg_canonical_llm")


def _cli(*args: str) -> tuple[str, ...]:
    # paper_v1 explicitly: the baseline was recorded from the code behind the paper
    return (*args, "--profile", "paper_v1", "--data", "{data}")


NEW_STEPS = [
    bb.Step("A1_chunk", "bmrag.build", _cli("chunk", "--workers", "1")),
    bb.Step("A2_clean", "bmrag.build", _cli("clean")),
    bb.Step("A3_separate", "bmrag.build", _cli("separate")),
    bb.Step("A4_summarize", "bmrag.build", _cli("summarize"), stand_in=True),
    bb.Step("B0_extract", "bmrag.build", _cli("extract"), stand_in=True),
    bb.Step("B2_entities", "bmrag.build", _cli("entities")),
    bb.Step("B2_quality", "bmrag.build", _cli("quality")),
    bb.Step("B5_cypher", "bmrag.build", _cli("cypher")),
    bb.Step(
        "B6_standardize_llm",
        "bmrag.build",
        _cli("standardize", "--set", "standardize.use_llm=true"),
        stand_in=True,
        after=_keep_llm_variant,
    ),
    bb.Step("B6_standardize", "bmrag.build", _cli("standardize")),
    bb.Step("B_kg_index", "bmrag.build", _cli("kg-index"), stand_in=True),
    bb.Step("B7_dry_run", "bmrag.build", _cli("neo4j")),
]


@unittest.skipUnless(ENABLED, "set BMRAG_BUILD_TESTS=1 to run the build pipeline regression test")
class BuildPipelineMatchesBaseline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not bb.build_packages_available() or not bb.tool_path():
            raise unittest.SkipTest("needs the build packages, Poppler and Tesseract")

    def _check(self, name: str) -> None:
        found = {case.name: (case, target) for case, target in recorder.cases()}
        if name not in found:
            self.skipTest(f"{name}: local data not available")
        case, target = found[name]
        expected = json.loads(target.read_text(encoding="utf-8"))
        actual = bb.run_case(case, NEW_STEPS, recorder.WORK.parent / "build_pipeline_test")
        self.assertEqual(bb.compare(expected, actual, NEW_FILES), [])

    def test_synthetic_paper(self) -> None:
        self._check("synthetic")

    def test_local_paper(self) -> None:
        self._check("local_paper")


if __name__ == "__main__":
    unittest.main()
