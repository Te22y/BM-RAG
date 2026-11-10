"""Reasoner agent: gathers evidence stage by stage, then predicts position and melting temperature.

Retrieval runs cheapest first. Stages 1-3 always run; stage 4 (graph rules) and
stage 5 (papers) run only when the evidence so far is thin. Which stages may run
and when they trigger come from the profile.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Optional

from bmrag import events, llm
from bmrag.profiles import (PAPER_V1, STAGE_GRAPH_CLIMATE, STAGE_GRAPH_RULES, STAGE_KG_CASES, STAGE_KG_RULES,
                            STAGE_PAPERS, Profile)
from bmrag.prompts import load_prompt
from bmrag.retrieval import graph, kg_index, papers
from bmrag.schemas import AnalystOutput, ReasonerOutput, RetrievalResult

POSITION_SEMANTICS = (
    "pos0 = outside surface, before stucco\n"
    "pos1 = stucco / gypsum interface\n"
    "pos2 = gypsum / insulation interface\n"
    "pos3 = MIDPOINT of insulation (insulation NoMass layer split R/2; PCM at centre) ← NEW\n"
    "pos4 = insulation / inner-gypsum interface  (= old pos3 from 6-Tm grid)"
)

# Climate-agnostic queries that surface position-to-indoor coupling evidence climate queries miss
MECHANISM_RULE_QUERY = (
    "PCM proximity heat source interior coupling latent heat flux reduction "
    "wall position effectiveness R-C-R distribution insulation downstream"
)
MECHANISM_CASE_QUERY = (
    "PCM placement comparison interior exterior middle wall position "
    "effective heat flux reduction parametric"
)

MAX_CASE_RELATIONS = 20
MAX_RULE_RELATIONS = 15
PaperSearch = Callable[..., list[RetrievalResult]]


# ── Evidence cards ───────────────────────────────────────────────────────────
def _pack_case(result: RetrievalResult) -> dict[str, Any]:
    payload = result.payload
    profile = payload.get("case_profile") or {}
    context = profile.get("conditional_context") or {}
    internal_kg = payload.get("internal_kg") or {}
    return {
        "evidence_id": result.evidence_id,
        "source": result.source,
        "score": result.score,
        "climate": context.get("climate") or (profile.get("raw_case") or {}).get("climate_location"),
        "wall_assembly": context.get("wall_assembly"),
        "pcm_position": context.get("pcm_position"),
        "optimal_pcm_position": context.get("optimal_pcm_position"),
        "performance_metrics": context.get("performance_metrics"),
        "case_summary": profile.get("case_summary"),
        "kg_relations": (internal_kg.get("relations_text") or [])[:MAX_CASE_RELATIONS],
    }


def _pack_rule(result: RetrievalResult) -> dict[str, Any]:
    payload = result.payload
    profile = payload.get("rule_profile") or {}
    internal_kg = payload.get("internal_kg") or {}
    return {
        "evidence_id": result.evidence_id,
        "source": result.source,
        "score": result.score,
        "rule": profile.get("kg_rule") or {},
        "case_summary": profile.get("case_summary"),
        "kg_relations": (internal_kg.get("relations_text") or [])[:MAX_RULE_RELATIONS],
    }


def _pack_graph_row(result: RetrievalResult) -> dict[str, Any]:
    return {
        "evidence_id": result.evidence_id,
        "source": result.source,
        "template": result.diagnostics.get("template"),
        "row": result.payload,
    }


def _pack_paper(result: RetrievalResult) -> dict[str, Any]:
    payload = result.payload
    return {
        "evidence_id": result.evidence_id,
        "source": result.source,
        "score": result.score,
        "element_type": payload.get("element_type"),
        "source_file": payload.get("source_file"),
        "summary": payload.get("summary"),
    }


def _union_by_id(*result_lists: list[RetrievalResult]) -> list[RetrievalResult]:
    """Merge result lists, keeping the first occurrence of each evidence id."""
    seen: set[str] = set()
    merged = []
    for results in result_lists:
        for result in results:
            if result.evidence_id not in seen:
                seen.add(result.evidence_id)
                merged.append(result)
    return merged


# ── Traced retrieval calls ───────────────────────────────────────────────────
def _search_cases(query: str, top_k: int) -> list[RetrievalResult]:
    return events.traced_retrieval("retrieve_cases", {"query": query, "top_k": top_k},
                                   lambda: kg_index.retrieve_cases(query, top_k=top_k))


def _search_rules(query: str, top_k: int) -> list[RetrievalResult]:
    return events.traced_retrieval("retrieve_rules", {"query": query, "top_k": top_k},
                                   lambda: kg_index.retrieve_rules(query, top_k=top_k))


def _query_graph(template: str, **params: Any) -> list[RetrievalResult]:
    return events.traced_retrieval("neo4j_template", {"template": template, "params": params},
                                   lambda: graph.run_template(template, **params))


def _write_paper_query(analyst_out: AnalystOutput, cases: list[RetrievalResult], rules: list[RetrievalResult],
                       profile: Profile) -> str:
    """The query for paper search: the Analyst's, or one written to fill gaps in the evidence."""
    if not profile.context_paper_query:
        return analyst_out.pcm_query
    try:
        raw = analyst_out.raw_input or {}
        prompt = load_prompt("paper_query").format(
            climate=raw.get("climate", ""),
            wall=(raw.get("wall_construction") or "")[:200],
            pcm=(raw.get("pcm_parameters") or "")[:200],
            goal=raw.get("goal", ""),
            n_cases=len(cases),
            n_rules=len(rules),
            sample_cases=", ".join(c.evidence_id for c in cases[:5]) or "(none)",
            sample_rules=", ".join(r.evidence_id for r in rules[:5]) or "(none)",
        )
        chat_model = llm.create_chat_model(profile.analyst_model, temperature=0.2, max_retries=1)
        reply = llm.complete(chat_model, prompt, caller="stage5_query")
        for line in reply.splitlines():
            line = line.strip().strip('"').strip("'")
            if line:
                return line
    except Exception:
        pass
    return analyst_out.pcm_query


def collect_evidence(analyst_out: AnalystOutput, profile: Profile, top_k_cases: int, top_k_rules: int,
                     paper_search: Optional[PaperSearch] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the retrieval stages. Returns the evidence cards and a per-stage record.

    paper_search replaces the paid paper search, for scans that only inspect triggers.
    """
    stages = profile.retrieval_stages
    diagnostics: dict[str, Any] = {"stages": []}
    evidence: list[dict[str, Any]] = []
    cases: list[RetrievalResult] = []
    rules: list[RetrievalResult] = []

    if STAGE_KG_CASES in stages:
        query, top_k = analyst_out.composite_query, top_k_cases
        climate_cases, rewrite = events.traced_retrieval(
            "retrieve_cases_with_fallback", {"query": query, "top_k": top_k},
            lambda: kg_index.retrieve_cases_with_fallback(query, top_k=top_k,
                                                          weak_top1_score=profile.weak_top1_score,
                                                          weak_score_spread=profile.weak_score_spread,
                                                          rewrite_model=profile.analyst_model))
        mechanism_cases = _search_cases(MECHANISM_CASE_QUERY, max(2, top_k_cases // 2))
        cases = _union_by_id(climate_cases, mechanism_cases)
        evidence.extend(_pack_case(c) for c in cases)
        diagnostics["stages"].append({
            "stage": "kg_cases",
            "n_climate": len(climate_cases),
            "n_mechanism": len(mechanism_cases),
            "n_union": len(cases),
            "fallback": rewrite.get("fallback_used", False),
            "rewritten": rewrite.get("rewritten_query"),
        })

    if STAGE_KG_RULES in stages:
        objective = analyst_out.objective_kw
        climate_rules = _search_rules(f"{objective} PCM optimal position {analyst_out.climate_query}", top_k_rules)
        mechanism_rules = _search_rules(MECHANISM_RULE_QUERY, top_k_rules)
        objective_rules = _search_rules(f"{objective} PCM placement wall layer energy reduction",
                                        max(2, top_k_rules - 1))
        rules = _union_by_id(climate_rules, mechanism_rules, objective_rules)
        evidence.extend(_pack_rule(r) for r in rules)
        diagnostics["stages"].append({
            "stage": "kg_rules",
            "n_climate": len(climate_rules),
            "n_mechanism": len(mechanism_rules),
            "n_objective": len(objective_rules),
            "n_union": len(rules),
        })

    if STAGE_GRAPH_CLIMATE in stages:
        try:
            cohort = _query_graph("cases_by_climate", climate_kw=analyst_out.climate_kw or "", limit=8)
        except Exception as exc:
            cohort = []
            diagnostics["neo4j_cohort_error"] = str(exc)
        evidence.extend(_pack_graph_row(c) for c in cohort)
        diagnostics["stages"].append({"stage": "neo4j_cases_by_climate", "n": len(cohort)})

    if STAGE_GRAPH_RULES in stages and len(rules) < profile.sufficient_rules:
        graph_rules = _query_graph("rules_by_climate_objective", climate_kw=analyst_out.climate_kw or None,
                                   objective_kw=analyst_out.objective_kw or None, limit=8)
        evidence.extend(_pack_graph_row(r) for r in graph_rules)
        diagnostics["stages"].append({"stage": "neo4j_rules_by_objective", "n": len(graph_rules)})

    if STAGE_PAPERS in stages:
        with_optimum = sum(1 for e in evidence if e.get("source") == "kg_case" and e.get("optimal_pcm_position"))
        if with_optimum < profile.sufficient_cases:
            try:
                query = _write_paper_query(analyst_out, cases, rules, profile)
                if paper_search is not None:
                    chunks = paper_search(query, top_k=profile.paper_top_k)
                else:
                    chunks = events.traced_retrieval(
                        "retrieve_paper_chunks", {"query": query, "top_k": profile.paper_top_k},
                        lambda: papers.retrieve_paper_chunks(query, top_k=profile.paper_top_k,
                                                             rerank_model=profile.paper_rerank_model))
                evidence.extend(_pack_paper(c) for c in chunks)
                diagnostics["stages"].append({"stage": "paper_chunks", "n": len(chunks), "query": query})
            except Exception as exc:
                diagnostics["paper_chunks_error"] = str(exc)
        else:
            diagnostics["stages"].append({"stage": "paper_chunks", "n": 0, "skipped": True})

    return evidence, diagnostics


def build_prompt(analyst_out: AnalystOutput, evidence: list[dict[str, Any]],
                 extra_instruction: Optional[str] = None) -> str:
    prompt = load_prompt("reasoner").format(
        pos_semantics=POSITION_SEMANTICS,
        user_task_json=json.dumps(analyst_out.to_dict(), ensure_ascii=False, indent=2),
        evidence_json=json.dumps(evidence, ensure_ascii=False, indent=2, default=str),
    )
    if extra_instruction:
        prompt += "\n\nADDITIONAL INSTRUCTION:\n" + extra_instruction
    return prompt


def reason(analyst_out: AnalystOutput, profile: Profile = PAPER_V1, top_k_cases: Optional[int] = None,
           top_k_rules: Optional[int] = None, extra_instruction: Optional[str] = None) -> ReasonerOutput:
    evidence, diagnostics = collect_evidence(
        analyst_out, profile,
        profile.first_top_k_cases if top_k_cases is None else top_k_cases,
        profile.first_top_k_rules if top_k_rules is None else top_k_rules,
    )
    chat_model = llm.create_chat_model(profile.reasoner_model, temperature=0, max_retries=2)
    reply = llm.complete(chat_model, build_prompt(analyst_out, evidence, extra_instruction), caller="reasoner")
    reply = llm.strip_code_fence(reply)
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Reasoner returned non-JSON output: {exc}\n--- raw ---\n{reply}") from exc

    position = data.get("pcm_interface_pos")
    melt_temperature = data.get("pcm_tm_c")
    return ReasonerOutput(
        position=str(data.get("position", "")),
        pcm_interface_pos=int(position) if position is not None else None,
        pcm_tm_c=float(melt_temperature) if melt_temperature is not None else None,
        tm_reasoning=[str(s) for s in (data.get("tm_reasoning") or [])],
        joint_rationale=str(data.get("joint_rationale", "")),
        reasoning_chain=[str(s) for s in (data.get("reasoning_chain") or [])],
        evidence_ids=[str(i) for i in (data.get("evidence_ids") or [])],
        confidence=float(data.get("confidence") or 0.0),
        alternatives=list(data.get("alternatives") or []),
        diagnostics=diagnostics,
    )
