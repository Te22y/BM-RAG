"""Data passed between the agents, and the fixed prediction grid."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Must match simulation/energyplus/config.yaml
POSITION_GRID = [0, 1, 2, 3, 4]
MELT_TEMPERATURE_GRID = [12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32]


@dataclass
class AnalystInput:
    """The user's design task. All fields are free text; the Analyst normalizes them."""
    climate: str                        # e.g. "Beijing, hot summer cold winter"
    building_type: str                  # e.g. "residential apartment"
    wall_construction: str              # e.g. "exterior cement render | brick 240mm | EPS 50mm | gypsum 12mm"
    pcm_parameters: str                 # e.g. "Tm=24°C, latent heat=180 kJ/kg, thickness=20 mm"
    goal: str = "minimize cooling load"
    orientation: Optional[str] = None
    season: Optional[str] = None
    # One of min_total_energy, min_cooling_peak, min_heating, thermal_comfort; drives objective_kw
    optimization_target: str = "min_total_energy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AnalystOutput:
    raw_input: dict[str, Any]
    climate_query: str
    construction_query: str
    pcm_query: str
    composite_query: str
    materials: list[str]
    objective_kw: str
    climate_kw: str
    pcm_tm_c: Optional[float]
    pcm_thickness_mm: Optional[float]
    wall_layer_count: int
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalResult:
    """Uniform result of every retriever; the Judge verifies citations through evidence_id."""
    source: str                         # kg_case | kg_rule | neo4j | paper_chunk
    evidence_id: str                    # case id, rule id, Cypher row id or paper element id
    score: Optional[float]              # similarity or rerank score; None for graph rows
    payload: dict[str, Any]
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasonerOutput:
    position: str
    pcm_interface_pos: Optional[int]
    pcm_tm_c: Optional[float]
    tm_reasoning: list[str]
    joint_rationale: str
    reasoning_chain: list[str]
    evidence_ids: list[str]
    confidence: float
    alternatives: list[dict[str, Any]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class JudgeOutput:
    verdict: str                        # pass | retry
    fatal_issues: list[str]
    soft_issues: list[str]
    evidence_audit: dict[str, Any]
    rubric_scores: dict[str, float]     # always empty since the LLM rubric was retired; kept in the record format
    overall_score: float
    retry_hint: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    user_input: dict[str, Any]
    analyst: dict[str, Any]
    reasoner_attempts: list[dict[str, Any]]
    final_prediction: dict[str, Any]
    verdict: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
