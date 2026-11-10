"""Synthetic pipeline scenarios shared by the baseline recorder and the tests.

Everything here is invented: no text comes from the literature, so the recorded
baseline can be stored in the repository. Retrieval results depend on the query
string and top_k, so a changed query or parameter changes the recorded prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

TASK = {
    "climate": "testville",
    "building_type": (
        "small office, exterior wall; HVAC dual-setpoint thermostat - occupied: cooling setpoint = 23.89°C, "
        "heating setpoint = 21.11°C; setback: cooling = 29.44°C, heating = 15.56°C."
    ),
    "wall_construction": "stucco | gypsum | insulation R=1.07 | gypsum",
    "pcm_parameters": "organic PCM, latent heat=200 kJ/kg, thickness=20 mm, phase-transition band = 4°C",
    "goal": "minimize total annual energy",
    "orientation": None,
    "season": None,
    "optimization_target": "min_total_energy",
}

ANALYST_REPLY = {
    "climate_query": "synthetic climate query",
    "construction_query": "synthetic construction query",
    "pcm_query": "synthetic pcm query",
    "composite_query": "synthetic composite query",
    "materials": ["Stucco", "Gypsum"],
    "objective_kw": "Energy Saving",
    "climate_kw": "testville",
    "pcm_tm_c": None,
    "pcm_thickness_mm": 20,
    "wall_layer_count": 4,
    "notes": ["synthetic note"],
}

MECHANISM_REPLY = "rewritten synthetic query"


def reasoner_reply(pos: int, tm: float, evidence_ids: list[str]) -> str:
    return json.dumps({
        "position": f"synthetic position {pos}",
        "pcm_interface_pos": pos,
        "pcm_tm_c": tm,
        "tm_reasoning": ["synthetic Tm step"],
        "joint_rationale": "synthetic rationale",
        "reasoning_chain": ["synthetic step"],
        "evidence_ids": evidence_ids,
        "confidence": 0.5,
        "alternatives": [],
    })


POOL_SIZE = 40

SCENARIOS: dict[str, dict[str, Any]] = {
    # pos4 inside the Judge's accepted Tm range: passes at once; enough evidence to skip stages 4 and 5
    "pass_first_attempt": {
        "replies": [reasoner_reply(4, 20.0, ["case_syn_0", "rule_syn_0"])],
        "optimal_cases": POOL_SIZE, "n_rules": POOL_SIZE, "tight_scores": False, "sources": None, "max_retries": 2,
    },
    # pos3 always gets the generic cycling reminder: runs out of retries
    "soft_retry_exhausted": {
        "replies": [reasoner_reply(3, 26.0, ["case_syn_0"]), reasoner_reply(3, 24.0, ["case_syn_1"]),
                    reasoner_reply(3, 24.0, ["case_syn_1"])],
        "optimal_cases": POOL_SIZE, "n_rules": POOL_SIZE, "tight_scores": False, "sources": None, "max_retries": 2,
    },
    # A missing citation is fatal, then the retry passes
    "fatal_then_pass": {
        "replies": [reasoner_reply(4, 20.0, ["case_missing"]), reasoner_reply(4, 20.0, ["case_syn_0"])],
        "optimal_cases": POOL_SIZE, "n_rules": POOL_SIZE, "tight_scores": False, "sources": None, "max_retries": 2,
    },
    # Tight scores trigger the query rewrite; thin evidence triggers stages 4 and 5
    "rewrite_and_all_stages": {
        "replies": [reasoner_reply(4, 22.0, ["case_syn_0", "rule_syn_0", "abcdef0123456789__text__0001",
                                             "cases_by_climate_row_0", "R11"])],
        "optimal_cases": 1, "n_rules": 2, "tight_scores": True, "sources": None, "max_retries": 2,
    },
    # Same restriction as the "no KG" ablation
    "paper_only": {
        "replies": [reasoner_reply(4, 18.0, ["abcdef0123456789__text__0001"])],
        "optimal_cases": 1, "n_rules": 2, "tight_scores": False, "sources": ["paper_chunks"], "max_retries": 2,
    },
    # Off-grid prediction and an opposite-end contradiction are fatal; max_retries=0 stops at once
    "off_grid_no_retries": {
        "replies": [reasoner_reply(7, 21.0, ["case_syn_0"])],
        "optimal_cases": POOL_SIZE, "n_rules": POOL_SIZE, "tight_scores": False, "sources": None, "max_retries": 0,
    },
}

MISSING_CASES = {"case_missing"}
PAPER_DOC_IDS = [f"abcdef0123456789__text__{i:04d}" for i in range(1, 7)]


def _rotation(query: str, size: int) -> int:
    return int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16) % size


def _scores(top_k: int, tight: bool) -> list[float]:
    step = 0.001 if tight else 0.02
    return [round(0.65 - i * step, 6) for i in range(top_k)]


def case_bundle(index: int, score: float | None, optimal: bool) -> dict[str, Any]:
    return {
        "case_id": f"case_syn_{index}",
        "score": score,
        "source_file": f"synthetic paper {index % 3}.pdf",
        "case_profile": {
            "case_summary": f"Synthetic summary of case {index}.",
            "conditional_context": {
                "climate": f"synthetic climate {index}",
                "wall_assembly": "synthetic wall",
                "pcm_position": "interior" if index % 2 else "exterior",
                "optimal_pcm_position": "interior" if optimal else None,
                "performance_metrics": f"{index} percent",
            },
            "raw_case": {"climate_location": f"synthetic location {index}"},
        },
        "internal_kg": {"relations_text": [f"Case {index} --REL{j}--> Thing {j}" for j in range(25)]},
    }


def rule_bundle(index: int, score: float | None = None) -> dict[str, Any]:
    return {
        "rule_id": f"rule_syn_{index}",
        "score": score,
        "source_file": f"synthetic paper {index % 3}.pdf",
        "rule_profile": {
            "case_summary": f"Synthetic rule context {index}.",
            "kg_rule": {"relation": "FAVORS", "subject": f"subject {index}", "object": "interior",
                        "condition": "synthetic", "evidence_sentence": f"Synthetic sentence {index}."},
        },
        "internal_kg": {"relations_text": [f"Rule {index} --REL{j}--> Thing {j}" for j in range(20)]},
    }


def search_cases(query: str, top_k: int, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    first = _rotation(query, POOL_SIZE)
    scores = _scores(top_k, scenario["tight_scores"])
    return [case_bundle(i, scores[rank], i < scenario["optimal_cases"])
            for rank, i in enumerate((first + r) % POOL_SIZE for r in range(top_k))]


def search_rules(query: str, top_k: int, scenario: dict[str, Any]) -> list[dict[str, Any]]:
    if scenario["n_rules"] < POOL_SIZE:
        return [rule_bundle(i, 0.6) for i in range(min(top_k, scenario["n_rules"]))][:1]
    first = _rotation(query, POOL_SIZE)
    scores = _scores(top_k, False)
    return [rule_bundle((first + r) % POOL_SIZE, scores[r]) for r in range(top_k)]


def graph_rows(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Rows a Neo4j session would return for one of the Cypher templates."""
    if "c.case_id IN $case_ids" in cypher:
        return [{"case_id": cid, "paper": "synthetic paper", "favored_positions": ["interior side of the wall"],
                 "tested_positions": ["exterior", "interior"], "climate": "synthetic climate",
                 "supporting_rules": []}
                for cid in params["case_ids"] if cid not in MISSING_CASES]
    if "KGRule" in cypher and "supporting_cases" in cypher:
        return [{"rule_id": "rule_syn_5", "evidence": "Synthetic sentence.", "favored_position": "interior",
                 "climates": ["testville"], "objectives": ["energy saving"], "supporting_cases": 2}]
    if "STUDIED_IN" in cypher and "favored_positions" in cypher:
        return [{"paper": "synthetic paper", "climate": "testville", "favored_positions": ["interior"],
                 "tested_positions": [], "outcomes": [], "pcm_materials": []}]
    return []


def reply_to(prompt: str, replies: list[str]) -> str:
    """The canned LLM reply for a prompt, chosen by which agent wrote the prompt."""
    if prompt.startswith("You are the ANALYST"):
        return json.dumps(ANALYST_REPLY)
    if prompt.startswith("You rewrite a building-PCM"):
        return MECHANISM_REPLY
    if prompt.startswith("You are an expert relevance"):
        count = len(re.findall(r"^\[\d+\] ", prompt.split("Documents:\n", 1)[1], flags=re.MULTILINE))
        return json.dumps([{"index": i, "score": round(0.9 - i * 0.05, 2)} for i in range(count)])
    if prompt.startswith("You are the REASONER"):
        return replies.pop(0) if len(replies) > 1 else replies[0]
    # Ablation baselines without agents
    return json.dumps({"pcm_interface_pos": 4, "pcm_tm_c": 20.3, "rationale": "synthetic rationale"})


SCORING_CASES = [(4, 18.0), (4, 19.0), (3, 26.0), (0, 32.0), (None, 20.0), (4, None), (9, 20.0), (2, 14.0)]
SYNTHETIC_LOOKUP_CSV = """\
city,zone,status,is_psm,is_baseline,position,melt_temp,total_kWh,pct_savings,baseline_delta_total_kWh,discomfort_hours,latent_utilization
alpha,cold,ok,false,true,,,1000,,,,
alpha,cold,ok,false,false,4,18,900,10,100,5,0.5
alpha,cold,ok,false,false,4,20,900.005,9.99,99,5,0.5
alpha,cold,ok,false,false,3,26,950,5,50,7,
alpha,cold,ok,false,false,0,32,990,1,10,,0.1
alpha,cold,ok,true,false,4,18,800,20,200,5,0.5
alpha,cold,failed,false,false,4,22,700,30,300,5,0.5
"""
