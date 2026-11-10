"""Repository rules that are easy to break by accident."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CODE_SUFFIXES = {".py", ".yaml", ".toml"}
FORBIDDEN_PATHS = re.compile(r"D:[\\/]+DXD_Projects|03_PCM_codes|05_Agents|06_Ablation|04_Energyplus|02_codes")
# The recorder documents the old layout on purpose; these tests name the patterns
ALLOWED = {"tests/golden/record_baseline_2514c33.py", "tests/test_repository.py"}
CHINESE = re.compile(f"[{chr(0x4e00)}-{chr(0x9fff)}]")


def _tracked_files() -> list[Path]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                            capture_output=True, text=True, encoding="utf-8").stdout.splitlines()
    return [REPO / name for name in listed if (REPO / name).exists()]


class SourceRules(unittest.TestCase):
    def test_no_machine_paths_or_old_folders_in_code(self):
        offenders = []
        for path in _tracked_files():
            relative = path.relative_to(REPO).as_posix()
            if path.suffix in CODE_SUFFIXES and relative not in ALLOWED:
                if FORBIDDEN_PATHS.search(path.read_text(encoding="utf-8", errors="replace")):
                    offenders.append(relative)
        self.assertEqual(offenders, [])

    def test_comments_and_docstrings_are_in_english(self):
        offenders = []
        for path in _tracked_files():
            relative = path.relative_to(REPO).as_posix()
            if path.suffix in CODE_SUFFIXES:
                if CHINESE.search(path.read_text(encoding="utf-8", errors="replace")):
                    offenders.append(relative)
        self.assertEqual(offenders, [])


class ReplayDemo(unittest.TestCase):
    def test_replay_imports_with_the_standard_library_only(self):
        """-I and -S hide site-packages: the public replay must run from a bare Python."""
        code = ("import sys; sys.path.insert(0, 'web'); import server; "
                "print(len(server.load_scenarios()), len(server.Handler._sample_index()))")
        result = subprocess.run([sys.executable, "-I", "-S", "-X", "utf8", "-c", code], cwd=REPO,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        scenarios, samples = result.stdout.split()
        self.assertGreater(int(scenarios), 0)
        self.assertGreater(int(samples), 0)

    def test_samples_carry_what_the_page_needs(self):
        for path in [*(REPO / "web" / "samples").glob("*.json"), *(REPO / "web" / "demo_samples").glob("*.json")]:
            report = json.loads(path.read_text(encoding="utf-8"))
            self.assertLessEqual({"meta", "input", "attempts", "final", "events", "usage", "evidence"}, set(report))

    def test_public_recorded_samples_carry_no_paper_text(self):
        for path in (REPO / "web" / "demo_samples").glob("*.json"):
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["meta"].get("synthetic"):
                continue
            self.assertIn("(recorded run)", report["meta"]["title"])
            self.assertTrue(all(item["text"] is None for item in report["evidence"].values()), path.name)

    def test_demo_samples_are_invented_and_reproducible(self):
        import build_demo_samples as demo
        reports = demo.build_all()
        self.assertGreaterEqual(len(reports), 2)
        for sample_id, report in reports.items():
            stored = (REPO / "web" / "demo_samples" / f"{sample_id}.json").read_text(encoding="utf-8")
            self.assertEqual(demo.render(report), stored)
            self.assertTrue(report["meta"]["synthetic"])
            self.assertIn("synthetic", report["meta"]["title"])
            self.assertTrue(report["final"]["joint_rationale"].startswith("Invented example"))
            for item in report["evidence"].values():
                self.assertTrue(item["paper"] is None or item["paper"].startswith("Invented"), item)


if __name__ == "__main__":
    unittest.main()
