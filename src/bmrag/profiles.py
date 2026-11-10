"""Named pipeline configurations.

A profile holds every setting that can change a result. `paper_v1` reproduces the
published experiments exactly, so changes to the system must arrive as a new profile,
never as an edit to it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

from bmrag.config import load_environment

STAGE_KG_CASES = "kg_cases"
STAGE_KG_RULES = "kg_rules"
STAGE_GRAPH_CLIMATE = "neo4j_climate"
STAGE_GRAPH_RULES = "neo4j_rules"
STAGE_PAPERS = "paper_chunks"
ALL_STAGES = frozenset({STAGE_KG_CASES, STAGE_KG_RULES, STAGE_GRAPH_CLIMATE, STAGE_GRAPH_RULES, STAGE_PAPERS})

# Environment variables that override a profile's models
MODEL_OVERRIDES = {"ANALYST_MODEL": "analyst_model", "REASONER_MODEL": "reasoner_model",
                   "PAPER_RERANK_MODEL": "paper_rerank_model"}


@dataclass(frozen=True)
class Profile:
    name: str
    analyst_model: str = "gpt-5-mini"     # also rewrites case queries and writes paper queries
    reasoner_model: str = "gpt-5"
    paper_rerank_model: str = "gpt-5.2"
    retrieval_stages: frozenset[str] = ALL_STAGES
    use_analyst: bool = True              # False passes the task through without an LLM call
    use_judge: bool = True                # False accepts the first prediction unchecked
    max_retries: int = 2
    first_top_k_cases: int = 5
    first_top_k_rules: int = 3
    retry_top_k_cases: int = 6
    retry_top_k_rules: int = 4
    sufficient_cases: int = 6             # paper search runs below this many cases with a known optimum
    sufficient_rules: int = 4             # graph rule search runs below this many rules
    weak_top1_score: float = 0.45         # case query rewrite triggers
    weak_score_spread: float = 0.02
    paper_top_k: int = 3
    context_paper_query: bool = False     # paper_v1 searches papers with the Analyst's pcm_query


PAPER_V1 = Profile(name="paper_v1")
PROFILES = {PAPER_V1.name: PAPER_V1}


def get_profile(name: str = PAPER_V1.name, **overrides: Any) -> Profile:
    """A named profile with environment model overrides and explicit overrides applied."""
    load_environment()
    models = {field: os.environ[var] for var, field in MODEL_OVERRIDES.items() if os.environ.get(var)}
    return replace(PROFILES[name], **{**models, **overrides})
