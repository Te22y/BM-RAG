"""Semantic search over the knowledge-graph case and rule stores (Chroma plus JSON bundles)."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma

from bmrag import llm
from bmrag.config import CASE_COLLECTION, RULE_COLLECTION, get_paths
from bmrag.prompts import load_prompt
from bmrag.schemas import RetrievalResult

logger = logging.getLogger(__name__)
DEFAULT_REWRITE_MODEL = "gpt-5-mini"


@lru_cache(maxsize=None)
def _load_store(path: Path) -> dict[str, Any]:
    store = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(store, dict):
        raise ValueError(f"Expected a store object in {path}")
    return store


def load_case_store() -> dict[str, Any]:
    return _load_store(get_paths().case_store)


def load_rule_store() -> dict[str, Any]:
    return _load_store(get_paths().rule_store)


def _distance_to_score(distance: Optional[float]) -> Optional[float]:
    """Chroma distance (lower is better) as a bounded score (higher is better)."""
    if distance is None:
        return None
    return round(1.0 / (1.0 + max(float(distance), 0.0)), 6)


def _search_collection(query: str, top_k: int, collection: str) -> list[tuple[Any, float]]:
    store = Chroma(collection_name=collection, embedding_function=llm.create_embeddings(),
                   persist_directory=str(get_paths().kg_chroma))
    return store.similarity_search_with_score(query, k=top_k)


def search_case_bundles(query: str, top_k: int) -> list[dict[str, Any]]:
    cases = load_case_store()
    bundles = []
    for doc, distance in _search_collection(query, top_k, CASE_COLLECTION):
        case_id = str(doc.metadata.get("case_id") or "")
        bundle = cases.get(case_id)
        score = _distance_to_score(distance)
        if not bundle:
            bundles.append({"case_id": case_id, "score": score, "distance": distance,
                            "error": "case_id not found in case_store"})
            continue
        bundles.append({"case_id": bundle.get("case_id"), "score": score, "distance": distance,
                        "paper_id": bundle.get("paper_id"), "source_file": bundle.get("source_file"),
                        "retrieval_text": bundle.get("retrieval_text"), "case_profile": bundle.get("case_profile"),
                        "internal_kg": bundle.get("internal_kg"), "linked_rules": bundle.get("linked_rules", [])})
    return bundles


def search_rule_bundles(query: str, top_k: int) -> list[dict[str, Any]]:
    rules = load_rule_store()
    bundles = []
    for doc, distance in _search_collection(query, top_k, RULE_COLLECTION):
        rule_id = str(doc.metadata.get("rule_id") or doc.metadata.get("case_id") or "")
        bundle = rules.get(rule_id)
        score = _distance_to_score(distance)
        if not bundle:
            bundles.append({"rule_id": rule_id, "score": score, "distance": distance,
                            "error": "rule_id not found in rule_store"})
            continue
        bundles.append({"rule_id": bundle.get("rule_id"), "score": score, "distance": distance,
                        "paper_id": bundle.get("paper_id"), "source_file": bundle.get("source_file"),
                        "retrieval_text": bundle.get("retrieval_text"), "rule_profile": bundle.get("rule_profile"),
                        "internal_kg": bundle.get("internal_kg"),
                        "evidence_case_ids": bundle.get("evidence_case_ids", [])})
    return bundles


def retrieve_cases(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return [RetrievalResult(source="kg_case", evidence_id=bundle["case_id"], score=bundle.get("score"), payload=bundle)
            for bundle in search_case_bundles(query, top_k)]


def retrieve_rules(query: str, top_k: int = 5) -> list[RetrievalResult]:
    return [RetrievalResult(source="kg_rule", evidence_id=bundle["rule_id"], score=bundle.get("score"), payload=bundle)
            for bundle in search_rule_bundles(query, top_k)]


def _rewrite_query(query: str, model: str) -> str:
    chat_model = llm.create_chat_model(model, temperature=0, max_retries=2)
    reply = llm.complete(chat_model, load_prompt("case_query_rewrite").format(query=query), caller="case_query_rewrite")
    return reply.strip('"').strip("'")


def retrieve_cases_with_fallback(query: str, top_k: int = 5, weak_top1_score: float = 0.45,
                                 weak_score_spread: float = 0.02,
                                 rewrite_model: str = DEFAULT_REWRITE_MODEL) -> tuple[list[RetrievalResult], dict[str, Any]]:
    """retrieve_cases; when the scores look weak, rewrite the query with an LLM and search once more.

    Returns the results and a record of the decision. A failed rewrite keeps the first results.
    """
    results = retrieve_cases(query, top_k=top_k)
    scores = [r.score for r in results if r.score is not None]
    diagnostics: dict[str, Any] = {"original_query": query, "scores": scores, "fallback_used": False}
    if not scores:
        return results, diagnostics

    weak_top1 = scores[0] < weak_top1_score
    weak_spread = (max(scores) - min(scores)) < weak_score_spread
    if not (weak_top1 or weak_spread):
        return results, diagnostics

    diagnostics["fallback_reason"] = {"weak_top1": weak_top1, "weak_spread": weak_spread}
    try:
        rewritten = _rewrite_query(query, rewrite_model)
        diagnostics["rewritten_query"] = rewritten
        rewritten_results = retrieve_cases(rewritten, top_k=top_k)
    except Exception as exc:
        logger.warning("Case query rewrite failed: %s", exc)
        diagnostics["fallback_error"] = str(exc)
        return results, diagnostics
    diagnostics["fallback_used"] = True
    diagnostics["scores_after"] = [r.score for r in rewritten_results if r.score is not None]
    for result in rewritten_results:
        result.diagnostics["via_fallback"] = True
        result.diagnostics["rewritten_from"] = query
    return rewritten_results, diagnostics
