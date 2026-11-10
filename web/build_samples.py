"""Build replay samples from a `trace_pipeline.py --mode full` run directory.

Without --public, every city goes to web/samples/ (private). With --public, the chosen
cities go to web/demo_samples/ with paper text removed from the evidence: ids, paper
titles and the system's own reasoning stay.

Usage (from the repo root):
  python web/build_samples.py runs/<timestamp>_full
  python web/build_samples.py runs/<timestamp>_full --public --cities chicago,newyork
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from report import build_report  # noqa: E402

SAMPLES = HERE / "samples"
PUBLIC_SAMPLES = HERE / "demo_samples"
TITLES = {"beijing": "Beijing", "chicago": "Chicago", "newyork": "New York",
          "sandiego": "San Diego", "singapore": "Singapore"}


def _add_agent_summaries(events: list[dict], attempts: list[dict]) -> None:
    """Runs recorded before agent_end carried verdicts get them from the attempts."""
    by_attempt = {a["attempt"]: a for a in attempts}
    for e in events:
        attempt = by_attempt.get(e.get("attempt"))
        if e["kind"] != "agent_end" or attempt is None:
            continue
        if e["agent"] == "reasoner":
            r = attempt["reasoner"]
            e.update(pos=r["pcm_interface_pos"], tm=r["pcm_tm_c"], confidence=r["confidence"])
        elif e["agent"] == "judge":
            j = attempt["judge"]
            e.update(verdict=j["verdict"], n_fatal=len(j["fatal_issues"]), n_soft=len(j["soft_issues"]))


def build_sample(run: dict[str, Any], recorded: str, public: bool) -> dict[str, Any]:
    city = run["city"]
    pipeline = run["pipeline_result"]
    events = run["events"]
    _add_agent_summaries(events, pipeline["reasoner_attempts"])
    title = TITLES.get(city, city.title())
    meta = {"id": city, "title": f"{title} (recorded run)" if public else title, "zone": run["zone"],
            "recorded": recorded, "source": "replay"}
    report = build_report(meta, pipeline, events, run["llm"], city)
    if public:
        for item in report["evidence"].values():
            item["text"] = None   # summaries of paper content stay private
    return report


def render(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--public", action="store_true", help="remove paper text and write to web/demo_samples/")
    ap.add_argument("--cities", help="comma-separated subset")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    out_dir = args.out or (PUBLIC_SAMPLES if args.public else SAMPLES)
    wanted = set(args.cities.split(",")) if args.cities else None
    out_dir.mkdir(exist_ok=True)
    for path in sorted(args.run_dir.glob("*.json")):
        if path.name == "summary.json" or (wanted and path.stem not in wanted):
            continue
        report = build_sample(json.loads(path.read_text(encoding="utf-8")), args.run_dir.name[:8], args.public)
        out = out_dir / f"{report['meta']['id']}.json"
        out.write_text(render(report), encoding="utf-8")
        s = report["score"] or {}
        print(f"{out.stem:12s} {out.stat().st_size // 1024:4d} KB  pred pos{s.get('pos')}/{s.get('tm')}  "
              f"optimum pos{s.get('gt_pos')}/{s.get('gt_tm')}  capture_loss={s.get('capture_loss')}")


if __name__ == "__main__":
    main()
