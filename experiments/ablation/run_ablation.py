"""Run experiments x repetitions x cities, score each prediction by EnergyPlus lookup, and
write one JSON per run plus an aggregate summary to runs/ablation/.

Usage (from the repo root):
    python experiments/ablation/run_ablation.py --experiments all --runs 5
    python experiments/ablation/run_ablation.py --experiments exp1_bmrag,exp3_naive_rag --runs 2
    python experiments/ablation/run_ablation.py --cities miami,beijing --runs 1

Llama experiments are SKIPPED by default; pass --include-llama to run them.
If they raise auth/permission errors they are recorded as status=error
rather than crashing the whole driver.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from common import load_problem_set, score  # also puts src/ on the import path
from runners import EXPERIMENTS, safe_run

from bmrag.config import get_paths
from bmrag.retrieval.graph import close_driver

RESULTS_DIR = get_paths().runs_dir / "ablation"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LLAMA_EXPERIMENTS = {"exp2_only_llm_llama70b", "exp6_bmrag_llama"}
DEFAULT_EXPERIMENT_ORDER = [
    "exp2_only_llm_gpt5",
    "exp3_naive_rag",
    "exp4_graph_rag",
    "exp5_bmrag_no_kg",
    "exp1_bmrag",
    "exp2_only_llm_llama70b",
    "exp6_bmrag_llama",
]


# ══════════════════════════════════════════════════════════════════════════════
# Per-run execution
# ══════════════════════════════════════════════════════════════════════════════
def run_one_experiment(exp_name: str, run_id: int, records: list[dict[str, Any]]
                       ) -> dict[str, Any]:
    """One pass of an experiment over all cities. Saves per-run JSON."""
    fn = EXPERIMENTS[exp_name]
    t0 = time.time()
    per_city: list[dict[str, Any]] = []

    for i, rec in enumerate(records, 1):
        city = rec["city"]
        print(f"  [{exp_name}|run{run_id}|{i}/{len(records)}] {city}", flush=True)
        try:
            r = safe_run(fn, rec)
        except Exception as e:
            r = {"status": "error", "error": str(e),
                 "trace": traceback.format_exc(limit=8),
                 "prediction": {"pos": None, "tm": None}}

        pred = r.get("prediction") or {}
        metrics = score(city, pred.get("pos"), pred.get("tm"))
        per_city.append({
            "city":       city,
            "zone":       rec["zone"],
            "prediction": pred,
            "metrics":    metrics,
            "status":     r.get("status"),
            "error":      r.get("error"),
            "extra":      {k: v for k, v in r.items()
                            if k not in {"status","error","trace","prediction"}},
        })
        m = metrics
        if m.get("status") == "ok":
            print(f"     pred=(pos{pred.get('pos')},Tm{pred.get('tm')})  "
                  f"loss={m['capture_loss']:.2%}  "
                  f"strict={m['joint_strict']} loose={m['joint_loose']}",
                  flush=True)
        else:
            print(f"     status={m.get('status')} / {r.get('error','')[:80]}",
                  flush=True)

    elapsed = round(time.time() - t0, 1)
    out = {
        "experiment":   exp_name,
        "run_id":       run_id,
        "elapsed_sec":  elapsed,
        "n_cities":     len(records),
        "per_city":     per_city,
    }
    fp = RESULTS_DIR / f"{exp_name}_run{run_id}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str),
                  encoding="utf-8")
    print(f"  {exp_name} run{run_id} done ({elapsed:.0f}s): {fp.name}")
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Aggregation
# ══════════════════════════════════════════════════════════════════════════════
def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.mean(xs) if xs else None

def _stdev(xs):
    xs = [x for x in xs if x is not None]
    return statistics.stdev(xs) if len(xs) > 1 else 0.0


def aggregate_experiment(exp_name: str, runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate N runs of the same experiment. Compute per-run metric means,
    then mean ± std across runs."""
    if not runs:
        return {"experiment": exp_name, "note": "no runs"}

    # Per-run means
    per_run_summary = []
    for run in runs:
        ok = [c for c in run["per_city"] if (c.get("metrics",{}).get("status")=="ok")]
        if not ok:
            per_run_summary.append({"run_id": run["run_id"], "n_ok": 0})
            continue
        loss   = [c["metrics"]["capture_loss"]  for c in ok]
        strict = [c["metrics"]["joint_strict"]  for c in ok]
        loose  = [c["metrics"]["joint_loose"]   for c in ok]
        epos   = [c["metrics"]["exact_pos_hit"] for c in ok]
        npos   = [c["metrics"]["near_pos_hit"]  for c in ok]
        tmh2   = [c["metrics"]["tm_hit_2c"]     for c in ok]
        acc    = [c["metrics"]["is_acceptable"] for c in ok]
        per_run_summary.append({
            "run_id":         run["run_id"],
            "n_ok":           len(ok),
            "capture_loss":   _mean(loss),
            "joint_strict":   _mean(strict),
            "joint_loose":    _mean(loose),
            "exact_pos_hit":  _mean(epos),
            "near_pos_hit":   _mean(npos),
            "tm_hit_2c":      _mean(tmh2),
            "is_acceptable":  _mean(acc),
            "elapsed_sec":    run["elapsed_sec"],
        })

    def col(key):
        return [s.get(key) for s in per_run_summary if s.get("n_ok",0) > 0]

    agg = {
        "experiment":          exp_name,
        "n_runs":              len(runs),
        "n_runs_with_data":    sum(1 for s in per_run_summary if s.get("n_ok",0) > 0),
        "per_run":             per_run_summary,
        "across_runs": {
            "capture_loss_mean":     _mean(col("capture_loss")),
            "capture_loss_std":      _stdev(col("capture_loss")),
            "joint_strict_mean":     _mean(col("joint_strict")),
            "joint_strict_std":      _stdev(col("joint_strict")),
            "joint_loose_mean":      _mean(col("joint_loose")),
            "joint_loose_std":       _stdev(col("joint_loose")),
            "exact_pos_hit_mean":    _mean(col("exact_pos_hit")),
            "near_pos_hit_mean":     _mean(col("near_pos_hit")),
            "tm_hit_2c_mean":        _mean(col("tm_hit_2c")),
            "is_acceptable_mean":    _mean(col("is_acceptable")),
            "elapsed_sec_mean":      _mean(col("elapsed_sec")),
        },
    }

    # Per-city stability (mean prediction across runs)
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        for c in run["per_city"]:
            by_city[c["city"]].append(c)
    city_table = []
    for city, rows in sorted(by_city.items()):
        ok_rows = [r for r in rows if r.get("metrics",{}).get("status")=="ok"]
        if not ok_rows:
            city_table.append({"city": city, "n_ok": 0})
            continue
        city_table.append({
            "city":              city,
            "zone":              rows[0]["zone"],
            "n_ok":              len(ok_rows),
            "capture_loss_mean": _mean([r["metrics"]["capture_loss"] for r in ok_rows]),
            "capture_loss_std":  _stdev([r["metrics"]["capture_loss"] for r in ok_rows]),
            "joint_strict_rate": _mean([r["metrics"]["joint_strict"] for r in ok_rows]),
            "pos_mode":          _mode([r["prediction"]["pos"] for r in ok_rows]),
            "tm_mode":           _mode([r["prediction"]["tm"]  for r in ok_rows]),
        })
    agg["per_city"] = city_table
    return agg


def _mode(values):
    """Return most frequent value (ties broken arbitrarily)."""
    from collections import Counter
    cnt = Counter([v for v in values if v is not None])
    if not cnt: return None
    return cnt.most_common(1)[0][0]


# ══════════════════════════════════════════════════════════════════════════════
# CLI / main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", default="all",
                    help="comma-separated subset or 'all'")
    ap.add_argument("--runs",    type=int, default=5)
    ap.add_argument("--cities",  default=None,
                    help="comma-separated city subset (for quick tests)")
    ap.add_argument("--zones",   default=None,
                    help="comma-separated climate zones to include (e.g. dry,oceanic)")
    ap.add_argument("--include-llama", action="store_true",
                    help="run Llama experiments (skipped by default to avoid HF auth issues)")
    ap.add_argument("--problem-set", default=None,
                    help="path to alternative problem-set JSON (default: experiments/problem_sets/problem_set_18city.json)")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip per-run JSON files that already exist")
    args = ap.parse_args()

    if args.experiments == "all":
        exps = list(DEFAULT_EXPERIMENT_ORDER)
    else:
        exps = [e.strip() for e in args.experiments.split(",") if e.strip()]

    if not args.include_llama:
        exps = [e for e in exps if e not in LLAMA_EXPERIMENTS]

    records = load_problem_set(Path(args.problem_set) if args.problem_set else None)
    if args.cities:
        want = set(c.strip() for c in args.cities.split(","))
        records = [r for r in records if r["city"] in want]
    if args.zones:
        zwant = set(z.strip() for z in args.zones.split(","))
        records = [r for r in records if r.get("zone") in zwant]

    print(f"Experiments: {exps}")
    print(f"Runs/exp:    {args.runs}")
    print(f"Cities/run:  {len(records)}")
    print(f"Total runs:  {len(exps) * args.runs}")
    print("="*72)

    all_runs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    t_overall = time.time()
    for exp in exps:
        if exp not in EXPERIMENTS:
            print(f"[skip] unknown experiment: {exp}")
            continue
        print(f"\n### {exp} ###")
        for run_id in range(1, args.runs + 1):
            out_path = RESULTS_DIR / f"{exp}_run{run_id}.json"
            if args.skip_existing and out_path.exists():
                print(f"  {exp} run{run_id} already exists, loading")
                data = json.loads(out_path.read_text(encoding="utf-8"))
                all_runs[exp].append(data)
                continue
            try:
                out = run_one_experiment(exp, run_id, records)
                all_runs[exp].append(out)
            except Exception as e:
                print(f"  {exp} run{run_id} crashed: {e}")
                traceback.print_exc()

    # Cross-run aggregate per experiment
    summary = {"runs_per_experiment": args.runs,
               "n_cities":            len(records),
               "experiments":         {}}
    for exp, runs in all_runs.items():
        summary["experiments"][exp] = aggregate_experiment(exp, runs)

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = RESULTS_DIR / f"ablation_summary_{ts}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str),
                            encoding="utf-8")

    # Pretty print headline table
    print("\n" + "="*78)
    print("ABLATION SUMMARY")
    print("="*78)
    print(f"{'experiment':28s} {'loss%':>8s} {'±std':>6s} {'strict%':>8s} "
          f"{'loose%':>7s} {'accept%':>8s} {'time_s':>8s}")
    print("-"*78)
    for exp, agg in summary["experiments"].items():
        a = agg.get("across_runs", {})
        def _fmt_pct(x): return f"{x*100:6.2f}" if x is not None else "  n/a "
        print(f"{exp:28s} "
              f"{_fmt_pct(a.get('capture_loss_mean')):>8s} "
              f"{_fmt_pct(a.get('capture_loss_std')):>6s} "
              f"{_fmt_pct(a.get('joint_strict_mean')):>8s} "
              f"{_fmt_pct(a.get('joint_loose_mean')):>7s} "
              f"{_fmt_pct(a.get('is_acceptable_mean')):>8s} "
              f"{a.get('elapsed_sec_mean') or 0:>8.1f}")
    print("="*78)
    print(f"\nTotal wall time: {time.time()-t_overall:.0f}s")
    print(f"Summary: {summary_path}")

    close_driver()


if __name__ == "__main__":
    main()
