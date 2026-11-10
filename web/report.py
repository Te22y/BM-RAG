"""Turn a traced BM-RAG run into the JSON report the demo page renders.

Standard library only. Replay mode must work from a fresh clone, where the
knowledge-graph stores and the agent dependencies are not available.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmrag.config import get_paths  # noqa: E402
from bmrag.evaluation.ground_truth import load_truth, score  # noqa: E402

SUMMARY_CHARS = 280
QUERY_CHARS = 240

_EVENT_FIELDS = ("t", "kind", "agent", "attempt", "fn", "template", "n", "elapsed_sec", "caller",
                 "model", "input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens",
                 "cost_usd", "fallback_used", "pos", "tm", "confidence", "verdict", "n_fatal",
                 "n_soft", "error")


def slim_event(event: dict[str, Any]) -> dict[str, Any]:
    out = {k: event[k] for k in _EVENT_FIELDS if event.get(k) is not None}
    for key in ("query", "rewritten_query"):
        if event.get(key):
            out[key] = str(event[key])[:QUERY_CHARS]
    if event.get("ids"):
        out["ids"] = event["ids"][:6]
    return out


def _load_store(name: str) -> dict[str, Any]:
    path = get_paths().kg_index / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _load_cells(truth: dict[str, Any]) -> dict[str, dict[tuple[int, float], float]]:
    cells: dict[str, dict[tuple[int, float], float]] = {}
    for city, gt in truth.items():
        for row in gt.get("response_surface_top5") or []:
            cells.setdefault(city, {})[(int(row["pos"]), float(row["tm"]))] = row["total_kWh"]
    known_cells = get_paths().runs_dir / "gt_scoring" / "known_cells.json"
    if known_cells.exists():
        for city, rows in json.loads(known_cells.read_text(encoding="utf-8")).items():
            for row in rows:
                cells.setdefault(city, {})[(int(row["pos"]), float(row["tm"]))] = row["total_kWh"]
    return cells


def _paper_title(source_file: Any) -> str | None:
    return str(source_file).removesuffix(".pdf") if source_file else None


def _clip(text: Any) -> str | None:
    if not text:
        return None
    text = str(text)
    return text if len(text) <= SUMMARY_CHARS else text[:SUMMARY_CHARS].rsplit(" ", 1)[0] + "…"


def build_evidence(attempts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """One entry per cited id, from the Judge's audit plus the stores when available."""
    cases, rules = _load_store("case_store.json"), _load_store("rule_store.json")
    evidence: dict[str, dict[str, Any]] = {}
    for attempt in attempts:
        for eid, audit in attempt["judge"]["evidence_audit"].items():
            kind = audit.get("kind")
            if kind == "paper_chunk":
                kind_label = "paper"
            elif kind == "neo4j_row":
                kind_label = "graph"
            elif eid in cases and "rule_relation" not in audit:
                kind_label = "case"
            elif eid in rules or "rule_relation" in audit:
                kind_label = "rule"
            else:
                kind_label = "unknown"
            bundle = (cases if kind_label == "case" else rules).get(eid) or {}
            profile = bundle.get("case_profile") or bundle.get("rule_profile") or {}
            evidence[eid] = {
                "type": kind_label,
                "verified": bool(audit.get("exists")),
                "paper": _paper_title(bundle.get("source_file")),
                "climate": audit.get("climate"),
                "favored": audit.get("favored_positions") or [],
                "relation": audit.get("rule_relation"),
                "text": _clip(audit.get("evidence_sentence") or profile.get("case_summary")),
            }
    return evidence


def score_prediction(city: str | None, final: dict[str, Any]) -> tuple[dict | None, dict | None]:
    truth = load_truth()
    if not city or city not in truth:
        return None, None
    gt = truth[city]
    result = score(city, final.get("pcm_interface_pos"), final.get("pcm_tm_c"), truth, _load_cells(truth), {})
    public_truth = {"gt_pos": gt["gt_pos"], "gt_tm": gt["gt_tm"],
                    "tied_tm_set": gt.get("tied_tm_set") or [gt["gt_tm"]],
                    "savings_pct": gt.get("gt_savings_pct"),
                    "baseline_kWh": gt.get("baseline_total_kWh"),
                    "optimum_kWh": gt.get("gt_total_kWh")}
    return result, public_truth


def _slim_attempt(attempt: dict[str, Any]) -> dict[str, Any]:
    r, j = attempt["reasoner"], attempt["judge"]
    return {
        "attempt": attempt["attempt"],
        "pos": r["pcm_interface_pos"], "tm": r["pcm_tm_c"], "position": r["position"],
        "confidence": r["confidence"], "evidence_ids": r["evidence_ids"],
        "verdict": j["verdict"], "fatal_issues": j["fatal_issues"], "soft_issues": j["soft_issues"],
    }


def build_report(meta: dict[str, Any], pipeline: dict[str, Any], events: list[dict[str, Any]],
                 usage: dict[str, Any], city: str | None) -> dict[str, Any]:
    final = pipeline["final_prediction"]
    result, truth = score_prediction(city, final)
    analyst = pipeline["analyst"]
    return {
        "meta": meta,
        "input": pipeline["user_input"],
        "analyst": {k: analyst.get(k) for k in ("climate_kw", "objective_kw", "materials", "composite_query", "notes")},
        "attempts": [_slim_attempt(a) for a in pipeline["reasoner_attempts"]],
        "final": final,
        "verdict": pipeline["verdict"],
        "elapsed_seconds": pipeline["elapsed_seconds"],
        "events": [slim_event(e) for e in events],
        "usage": usage,
        "score": result,
        "truth": truth,
        "evidence": build_evidence(pipeline["reasoner_attempts"]),
    }
