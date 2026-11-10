"""Score predictions without the original EnergyPlus summary.csv.

Ground truth comes from the `ground_truth` field embedded in the problem sets.
Energy per (pos, Tm) cell is harvested from that field's top-5 list and from the
scored metrics in historical result files, then checked for consistency. The metric
formulas mirror lookup_scoring.score_one. A capture loss exists only for cells with
known energy. Standard library only: the replay demo scores with this module.

Read-only for project files. Writes runs/gt_scoring/.

Usage (from the repo root):
  python -m bmrag.evaluation.ground_truth [--run runs/<dir>_full]
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

from bmrag.config import REPO_ROOT, get_paths

PROBLEM_SETS = REPO_ROOT / "experiments" / "problem_sets"
PROBLEM_SET = PROBLEM_SETS / "problem_set_olddry.json"
PROBLEM_SET_18 = PROBLEM_SETS / "problem_set_18city.json"
# Cities of the 12-city ablation runs (ablation_summary_20260605_070256.json)
CITIES_12 = ["beijing", "chicago", "newyork", "phoenix", "cairo", "riyadh",
             "london", "seattle", "vancouver", "atlanta", "shanghai", "tokyo"]
GRID_SIZE = 55
FLOAT_TOL = 0.01


def load_truth() -> dict[str, dict[str, Any]]:
    """Olddry set (cairo/riyadh, tied Tm) merged with the 18-city set's top-5 lists."""
    truth = {r["city"]: dict(r["ground_truth"]) for r in
             json.loads(PROBLEM_SET.read_text(encoding="utf-8"))}
    for r in json.loads(PROBLEM_SET_18.read_text(encoding="utf-8")):
        gt = r["ground_truth"]
        if r["city"] not in truth:
            truth[r["city"]] = dict(gt)
        elif (gt["gt_pos"], gt["gt_tm"]) == (truth[r["city"]]["gt_pos"], truth[r["city"]]["gt_tm"]):
            truth[r["city"]] = {**gt, **truth[r["city"]]}
    for gt in truth.values():
        # olddry records omit the baseline; it follows from the optimum's savings
        gt.setdefault("baseline_total_kWh", gt["gt_total_kWh"] / (1 - gt["gt_savings_pct"] / 100))
    return truth


def history_files() -> list[str]:
    runs = get_paths().runs_dir
    files = glob.glob(str(runs / "ablation" / "exp*_run*.json"))
    files += glob.glob(str(runs / "eval_reports" / "eval_report_*.json"))
    files += glob.glob(str(runs / "diagnostic_runs" / "*" / "summary.json"))
    return sorted(files)


def _walk(obj: Any, city: str | None, out: list[tuple[str, dict[str, Any]]]) -> None:
    if isinstance(obj, dict):
        city = obj.get("city", city)
        if city and obj.get("status") == "ok" and obj.get("e_pred_kWh") is not None:
            out.append((city, obj))
        for value in obj.values():
            _walk(value, city, out)
    elif isinstance(obj, list):
        for value in obj:
            _walk(value, city, out)


def harvest(truth: dict[str, dict[str, Any]]) -> tuple[dict, dict, list]:
    """Return ({city: {(pos, tm): kWh}}, {city: tied Tm set}, conflicts)."""
    cells: dict[str, dict[tuple[int, float], float]] = defaultdict(dict)
    tied: dict[str, set[float]] = defaultdict(set)
    conflicts: list[dict[str, Any]] = []

    def put(city: str, key: tuple[int, float], kwh: float, source: str) -> None:
        old = cells[city].get(key)
        if old is not None and abs(old - kwh) > FLOAT_TOL:
            conflicts.append({"city": city, "cell": key, "kept": old, "other": kwh, "source": source})
            return
        cells[city].setdefault(key, kwh)

    for city, gt in truth.items():
        for row in gt.get("response_surface_top5") or []:
            put(city, (int(row["pos"]), float(row["tm"])), row["total_kWh"], "ground_truth")

    for path in history_files():
        found: list[tuple[str, dict[str, Any]]] = []
        _walk(json.loads(Path(path).read_text(encoding="utf-8")), None, found)
        for city, m in found:
            if city not in truth:
                continue
            gt = truth[city]
            # Skip records scored against another ground-truth version
            if m.get("gt_pos") != gt["gt_pos"] or m.get("gt_tm") != gt["gt_tm"]:
                continue
            put(city, (int(m["used_pos"]), float(m["used_tm"])), m["e_pred_kWh"], Path(path).name)
            tied[city].update(m.get("tied_tm_set") or [])
    return cells, tied, conflicts


def score(city: str, pos: Any, tm: Any, truth: dict, cells: dict, tied: dict) -> dict[str, Any]:
    gt = truth[city]
    if pos is None or tm is None:
        return {"status": "missing_prediction"}
    tm_set = gt.get("tied_tm_set") or sorted(tied.get(city) or {gt["gt_tm"]})
    pos_dist, tm_dist = abs(pos - gt["gt_pos"]), min(abs(tm - t) for t in tm_set)
    out: dict[str, Any] = {
        "status": "ok", "pos": pos, "tm": tm, "gt_pos": gt["gt_pos"], "gt_tm": gt["gt_tm"],
        "exact_pos_hit": pos_dist == 0, "near_pos_hit": pos_dist <= 1,
        "tm_hit_2c": tm_dist <= 2, "tm_hit_4c": tm_dist <= 4,
        "joint_strict": pos_dist == 0 and tm_dist <= 2,
        "joint_loose": pos_dist <= 1 and tm_dist <= 4,
        "capture_loss": None,
    }
    kwh = cells.get(city, {}).get((int(pos), float(tm)))
    if kwh is not None:
        base = gt["baseline_total_kWh"]
        s_pred = (base - kwh) / base * 100
        capture = s_pred / gt["gt_savings_pct"] if gt["gt_savings_pct"] else 0.0
        out["capture_loss"] = max(0.0, 1.0 - capture)
    return out


def verify(truth: dict, cells: dict, tied: dict) -> dict[str, Any]:
    """Rescore every historical record and compare with its stored metrics."""
    checked = mismatched = 0
    examples = []
    for path in glob.glob(str(get_paths().runs_dir / "ablation" / "exp*_run*.json")):
        for rec in json.loads(Path(path).read_text(encoding="utf-8"))["per_city"]:
            m = rec.get("metrics") or {}
            if m.get("status") != "ok" or rec["city"] not in truth:
                continue
            mine = score(rec["city"], m["used_pos"], m["used_tm"], truth, cells, tied)
            checked += 1
            bad = [k for k in ("exact_pos_hit", "near_pos_hit", "tm_hit_2c", "tm_hit_4c",
                               "joint_strict", "joint_loose") if mine[k] != m[k]]
            if abs((mine["capture_loss"] or 0) - m["capture_loss"]) > 1e-6:
                bad.append("capture_loss")
            if bad:
                mismatched += 1
                if len(examples) < 5:
                    examples.append({"file": Path(path).name, "city": rec["city"], "fields": bad})
    return {"records": checked, "mismatched": mismatched, "examples": examples}


def experiment_table() -> dict[str, dict[str, Any]]:
    """Per-experiment means over runs, for the 18 and the 12 city sets, from stored metrics."""
    runs: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for path in sorted(glob.glob(str(get_paths().runs_dir / "ablation" / "exp*_run*.json"))):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        runs[data["experiment"]].append(data["per_city"])

    def mean_over(run_list: list, cities: list[str] | None, field: str) -> float | None:
        values = []
        for per_city in run_list:
            ok = [c["metrics"][field] for c in per_city
                  if (c.get("metrics") or {}).get("status") == "ok"
                  and (cities is None or c["city"] in cities)]
            if ok:
                values.append(sum(ok) / len(ok))
        return round(st.mean(values), 4) if values else None

    fields = ("capture_loss", "is_acceptable", "exact_pos_hit", "near_pos_hit",
              "tm_hit_2c", "joint_strict", "joint_loose")
    return {name: {"n_runs": len(rl), "n_cities": len(rl[0]),
                   "all_cities": {f: mean_over(rl, None, f) for f in fields},
                   "cities_12": {f: mean_over(rl, CITIES_12, f) for f in fields}}
            for name, rl in sorted(runs.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", type=Path, help="trace_pipeline full-run directory to score")
    args = ap.parse_args()

    truth = load_truth()
    cells, tied, conflicts = harvest(truth)
    coverage = {c: len(cells.get(c, {})) for c in truth}
    missing_12 = {c: GRID_SIZE - coverage[c] for c in CITIES_12}
    report: dict[str, Any] = {
        "coverage_of_55": coverage,
        "cells_missing_12_cities": sum(missing_12.values()),
        "cell_conflicts": conflicts,
        "verification": verify(truth, cells, tied),
        "experiments": experiment_table(),
    }
    if args.run:
        rows = []
        for path in sorted(args.run.glob("*.json")):
            if path.name == "summary.json":
                continue
            r = json.loads(path.read_text(encoding="utf-8"))
            fp = r["final_prediction"]
            if r["city"] in truth:
                rows.append({"city": r["city"],
                             **score(r["city"], fp["pcm_interface_pos"], fp["pcm_tm_c"], truth, cells, tied)})
        report["scored_run"] = rows

    out = get_paths().runs_dir / "gt_scoring"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (out / "known_cells.json").write_text(json.dumps(
        {c: [{"pos": k[0], "tm": k[1], "total_kWh": v} for k, v in sorted(d.items())]
         for c, d in cells.items()}, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()}, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
