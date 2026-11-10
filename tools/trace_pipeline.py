"""Record what the pipeline does for review and optimisation. Changes nothing it runs.

Per city it records the agent timeline, every retrieval call, every LLM call with
tokens and cost, and which conditional stages ran.

Modes:
  scan  Analyst and retrieval for first-attempt and retry settings, stopping before
        the Reasoner LLM call. Paper search only records its query. Also runs the
        Neo4j templates with the Analyst's keywords. About US$0.006 per city.
  full  The complete pipeline, as an evaluation runs it. About US$0.15-0.40 per city.

Usage (from the repo root):
  .venv/Scripts/python.exe -X utf8 tools/trace_pipeline.py --mode scan --cities all
  .venv/Scripts/python.exe -X utf8 tools/trace_pipeline.py --mode full --cities beijing,miami
Outputs go to runs/<timestamp>_<mode>/ (git-ignored).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from bmrag import events, pipeline  # noqa: E402
from bmrag.agents import reasoner  # noqa: E402
from bmrag.agents.analyst import analyze  # noqa: E402
from bmrag.config import get_paths  # noqa: E402
from bmrag.evaluation import lookup_scoring  # noqa: E402
from bmrag.profiles import Profile, get_profile  # noqa: E402
from bmrag.retrieval import graph  # noqa: E402
from bmrag.schemas import AnalystInput  # noqa: E402

PROBLEM_SET = REPO / "experiments" / "problem_sets" / "problem_set_18city.json"
# Local sweep results used for scoring in full mode
LOCAL_SUMMARY_CSV = REPO / "simulation" / "energyplus" / "summary.csv"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, set):
        return sorted(value)
    return str(value)


def load_lookup(run_dir: Path) -> dict[str, Any]:
    """Lookup table from the local summary.csv, which lacks the latent_utilization column.

    A copy with that column left empty goes to the run directory; the original is not changed.
    """
    with LOCAL_SUMMARY_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    patched = run_dir / "summary_for_lookup.csv"
    fields = list(rows[0].keys()) + (["latent_utilization"] if "latent_utilization" not in rows[0] else [])
    with patched.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"latent_utilization": "", **row})
    return lookup_scoring.build_lookup(patched)


def _stage_flags(trace_events: list[dict[str, Any]], attempt: int) -> dict[str, Any]:
    attempt_events = [e for e in trace_events if e.get("attempt") == attempt and e.get("agent") == "reasoner"]
    calls = [e for e in attempt_events if e["kind"] in ("retrieval", "paper_chunks_stubbed")]
    fallback = next((e for e in calls if e.get("fn") == "retrieve_cases_with_fallback"), {})
    return {
        "case_fallback_rewrite": bool(fallback.get("fallback_used")),
        "neo4j_climate_n": next((e["n"] for e in calls if e.get("template") == "cases_by_climate"), None),
        "neo4j_rules_ran": any(e.get("template") == "rules_by_climate_objective" for e in calls),
        "paper_chunks_ran": any(e.get("fn") == "retrieve_paper_chunks" or e["kind"] == "paper_chunks_stubbed"
                                for e in calls),
        "reasoner_prompt_chars": next((e["prompt_chars"] for e in reversed(attempt_events)
                                       if e["kind"] in ("llm", "reasoner_prompt")
                                       and e.get("caller", "reasoner") == "reasoner" and e.get("prompt_chars")), None),
    }


def _record_paper_query(query: str, top_k: int = 3) -> list:
    events.emit("paper_chunks_stubbed", query=query, top_k=top_k)
    return []


def run_scan(record: dict[str, Any], profile: Profile | None = None) -> dict[str, Any]:
    profile = profile or get_profile()
    trace = events.Trace()
    out: dict[str, Any] = {"city": record["city"], "zone": record["zone"]}
    with events.capture(trace):
        with events.agent_span("analyst"):
            analyst_out = analyze(AnalystInput(**record["analyst_input"]), profile)
        out["analyst"] = analyst_out.to_dict()
        flags = []
        for top_k_cases, top_k_rules in ((profile.first_top_k_cases, profile.first_top_k_rules),
                                         (profile.retry_top_k_cases, profile.retry_top_k_rules)):
            with events.agent_span("reasoner"):
                evidence, _ = reasoner.collect_evidence(analyst_out, profile, top_k_cases, top_k_rules,
                                                        paper_search=_record_paper_query)
                events.emit("reasoner_prompt", prompt_chars=len(reasoner.build_prompt(analyst_out, evidence)))
            flags.append(_stage_flags(trace.events, trace.context["attempt"]))
        out["stages_attempt1"], out["stages_retry"] = flags
        out["template_check"] = _check_templates(analyst_out)
    out["events"] = trace.events
    out["llm"] = events.summarize_llm_usage(trace.events)
    return out


def _check_templates(analyst_out: Any) -> dict[str, Any]:
    """Run every retrieval template with the Analyst's keywords (direct and free)."""
    climate = analyst_out.climate_kw or None
    probes = {
        "cases_by_climate": ("cases_by_climate", {"climate_kw": analyst_out.climate_kw or "", "limit": 8}),
        "rules_by_climate_objective": ("rules_by_climate_objective", {
            "climate_kw": climate, "objective_kw": analyst_out.objective_kw or None, "limit": 8}),
        "rules_by_climate_only": ("rules_by_climate_objective", {"climate_kw": climate, "limit": 8}),
        "cases_by_wall_materials": ("cases_by_wall_materials", {"materials": analyst_out.materials or []}),
        "cases_by_pcm_and_climate": ("cases_by_pcm_and_climate", {"climate_kw": climate}),
    }
    results = {}
    for label, (template, params) in probes.items():
        started = time.time()
        try:
            rows = graph.run_template(template, **params)
            results[label] = {"n": len(rows), "elapsed_sec": round(time.time() - started, 3)}
        except Exception as exc:
            results[label] = {"error": repr(exc)[:300]}
    return results


def run_full(record: dict[str, Any], lookup: dict[str, Any], profile: Profile | None = None) -> dict[str, Any]:
    trace = events.Trace()
    with events.capture(trace):
        result = pipeline.run(AnalystInput(**record["analyst_input"]), profile or get_profile())
    res = result.to_dict()
    final = res["final_prediction"]
    metrics = (lookup_scoring.score_one(record["city"], final.get("pcm_interface_pos"), final.get("pcm_tm_c"), lookup)
               if record["city"] in lookup else {"status": "no_ground_truth_in_local_summary_csv"})
    attempts = []
    for attempt in res["reasoner_attempts"]:
        number = attempt["attempt"]
        agent_ends = [e for e in trace.events if e["kind"] == "agent_end" and e.get("attempt") == number]
        attempts.append({
            "attempt": number,
            "pred": [attempt["reasoner"]["pcm_interface_pos"], attempt["reasoner"]["pcm_tm_c"]],
            "confidence": attempt["reasoner"]["confidence"],
            "n_evidence_cited": len(attempt["reasoner"]["evidence_ids"]),
            "stages": attempt["reasoner"]["diagnostics"].get("stages"),
            **_stage_flags(trace.events, number),
            "judge_verdict": attempt["judge"]["verdict"],
            "judge_fatal": attempt["judge"]["fatal_issues"],
            "judge_soft": attempt["judge"]["soft_issues"],
            "reasoner_sec": next((e["elapsed_sec"] for e in agent_ends if e["agent"] == "reasoner"), None),
            "judge_sec": next((e["elapsed_sec"] for e in agent_ends if e["agent"] == "judge"), None),
        })
    analyst_sec = next((e["elapsed_sec"] for e in trace.events
                        if e["kind"] == "agent_end" and e.get("agent") == "analyst"), None)
    return {"city": record["city"], "zone": record["zone"], "verdict": res["verdict"],
            "elapsed_sec": res["elapsed_seconds"], "analyst_sec": analyst_sec,
            "final_prediction": final, "metrics": metrics, "attempts": attempts,
            "llm": events.summarize_llm_usage(trace.events), "events": trace.events, "pipeline_result": res}


def _summary_row(mode: str, run: dict[str, Any]) -> dict[str, Any]:
    usage = run["llm"]
    row: dict[str, Any] = {"city": run["city"], "zone": run["zone"], "cost_usd": usage["cost_usd_total"],
                           "llm_calls": sum(b["calls"] for b in usage["by_caller"].values())}
    if mode == "scan":
        for tag in ("attempt1", "retry"):
            stages = run[f"stages_{tag}"]
            row |= {f"{tag}_fallback": stages["case_fallback_rewrite"],
                    f"{tag}_neo4j_climate_n": stages["neo4j_climate_n"],
                    f"{tag}_neo4j_rules": stages["neo4j_rules_ran"],
                    f"{tag}_paper": stages["paper_chunks_ran"],
                    f"{tag}_prompt_chars": stages["reasoner_prompt_chars"]}
        row |= {f"tpl_{k}": v.get("n", "ERR") for k, v in run["template_check"].items()}
        row["climate_kw"] = run["analyst"].get("climate_kw")
        row["objective_kw"] = run["analyst"].get("objective_kw")
    else:
        metrics = run["metrics"]
        first = run["attempts"][0]
        row |= {"verdict": run["verdict"], "attempts": len(run["attempts"]), "elapsed_sec": run["elapsed_sec"],
                "pred_pos": run["final_prediction"]["pcm_interface_pos"],
                "pred_tm": run["final_prediction"]["pcm_tm_c"],
                "gt_pos": metrics.get("gt_pos"), "gt_tm": metrics.get("gt_tm"),
                "capture_loss": metrics.get("capture_loss"), "exact_pos_hit": metrics.get("exact_pos_hit"),
                "joint_strict": metrics.get("joint_strict"), "a1_fallback": first["case_fallback_rewrite"],
                "a1_neo4j_rules": first["neo4j_rules_ran"], "a1_paper": first["paper_chunks_ran"],
                "a1_prompt_chars": first["reasoner_prompt_chars"],
                "fatal_issues": sum(len(a["judge_fatal"]) for a in run["attempts"]),
                "soft_issues": sum(len(a["judge_soft"]) for a in run["attempts"])}
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--mode", choices=("scan", "full"), required=True)
    parser.add_argument("--cities", default="beijing", help="comma list or 'all'")
    parser.add_argument("--out-dir", type=Path, default=get_paths().runs_dir)
    args = parser.parse_args()

    records = {r["city"]: r for r in json.loads(PROBLEM_SET.read_text(encoding="utf-8"))}
    cities = list(records) if args.cities == "all" else [c.strip() for c in args.cities.split(",") if c.strip()]
    run_dir = args.out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{args.mode}"
    run_dir.mkdir(parents=True, exist_ok=True)
    lookup = load_lookup(run_dir) if args.mode == "full" else {}

    rows = []
    try:
        for city in cities:
            print(f"[{args.mode}] {city}", flush=True)
            try:
                run = run_scan(records[city]) if args.mode == "scan" else run_full(records[city], lookup)
            except Exception as exc:
                print(f"  ERROR {city}: {exc!r}", flush=True)
                rows.append({"city": city, "error": repr(exc)[:300]})
                continue
            (run_dir / f"{city}.json").write_text(json.dumps(run, ensure_ascii=False, indent=2, default=_jsonable),
                                                  encoding="utf-8")
            row = _summary_row(args.mode, run)
            rows.append(row)
            print("  " + json.dumps(row, ensure_ascii=False, default=str), flush=True)
    finally:
        graph.close_driver()
        (run_dir / "summary.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str),
                                              encoding="utf-8")
        columns = list(dict.fromkeys(k for r in rows for k in r))
        with (run_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        print(f"run dir: {run_dir}")


if __name__ == "__main__":
    main()
