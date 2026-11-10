"""Test suite. Run from the repo root: python -X utf8 -m unittest discover -s tests -t .

No test calls an API or needs a database. Tests that read the local stores skip themselves when the stores are absent.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for folder in ("src", "tools", "experiments/ablation", "tests/golden", "web"):
    path = str(REPO / folder)
    if path not in sys.path:
        sys.path.insert(0, path)
