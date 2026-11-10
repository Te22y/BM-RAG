"""The PCM graph ontology: allowed node labels and relationship types, and how names are normalized.

Labels and relationship types outside these sets are rejected by the entities, standardize and
neo4j stages. Older extraction outputs used other names; the LEGACY maps translate them.
"""

from __future__ import annotations

import re
from typing import Any

PCM_NODE_LABELS = {
    "Paper",
    "Case",
    "PCMMaterial",
    "MaterialClass",
    "WallAssembly",
    "WallConfiguration",
    "WallLayer",
    "WallPosition",
    "ClimateCondition",
    "Season",
    "Orientation",
    "Property",
    "Metric",
    "PerformanceOutcome",
    "PhysicalProcess",
    "PhysicalMechanism",
    "SimulationTool",
    "KGRule",
    "PerformanceObjective",
    "EvidenceGap",
    "StudyMethod",
    "SourceStudy",
}

PCM_RELATION_TYPES = {
    "HAS_CASE",
    "HAS_RULE",
    "USES_MATERIAL",
    "HAS_PROPERTY",
    "HAS_LAYER",
    "PLACED_AT",
    "STUDIED_IN",
    "HAS_CONDITION",
    "MEASURES",
    "ACHIEVES",
    "FAVORS",
    "REDUCES",
    "INCREASES",
    "DELAYS",
    "REQUIRES",
    "SHIFTS",
    "ENABLES",
    "CAUSES",
    "LEADS_TO",
    "COMPARED_TO",
    "SUPPORTED_BY",
    "SUPPORTS_RULE",
    "DEPENDS_ON",
    "CONFLICTS_WITH",
    "HAS_OBJECTIVE",
    "HAS_EVIDENCE_GAP",
    "HAS_METHOD",
    "HAS_SOURCE_STUDY",
    "NOT_SUPPORTED_AS_UNIVERSAL",
}

LEGACY_LABEL_MAP = {
    "ClimateZone": "ClimateCondition",
    "EvaluationMetric": "Metric",
    "ExternalFactor": "ClimateCondition",
    "MaterialProperty": "Property",
    "BuildingType": "WallConfiguration",
    "WallAssembly": "WallAssembly",
    "PhysicalMechanism": "PhysicalMechanism",
    "Literal": "Property",
}

LEGACY_RELATION_MAP = {
    "HAS_PHASE_CHANGE_TEMP": "HAS_PROPERTY",
    "OPTIMAL_IN": "ACHIEVES",
    "USED_IN": "STUDIED_IN",
    "PREVENTS": "REDUCES",
    "GENERALIZES_TO": "SUPPORTED_BY",
    "COMPOSED_OF": "HAS_LAYER",
    "AMPLIFIES": "INCREASES",
    "DIMINISHES": "REDUCES",
    "CONFLICTS_WITH": "CONFLICTS_WITH",
    "DEPENDS_ON": "DEPENDS_ON",
    "FAVORS": "FAVORS",
    "SHIFTS": "SHIFTS",
    "REQUIRES": "REQUIRES",
    "NOT_SUPPORTED_AS_UNIVERSAL": "NOT_SUPPORTED_AS_UNIVERSAL",
}

CASE_LEVEL_RELATIONS = {
    "MEASURES",
    "ACHIEVES",
    "FAVORS",
    "REDUCES",
    "INCREASES",
    "DELAYS",
    "REQUIRES",
    "SHIFTS",
    "ENABLES",
    "CAUSES",
    "LEADS_TO",
    "COMPARED_TO",
    "SUPPORTS_RULE",
    "DEPENDS_ON",
    "CONFLICTS_WITH",
    "NOT_SUPPORTED_AS_UNIVERSAL",
}

PRESERVE_CASE_TERMS = {
    "pcm": "PCM",
    "hvac": "HVAC",
    "tpcm": "Tpcm",
    "rubitherm gr": "Rubitherm GR",
}


def normalize_label(label: str | None) -> str:
    label = (label or "Property").strip()
    label = LEGACY_LABEL_MAP.get(label, label)
    if label not in PCM_NODE_LABELS:
        raise ValueError(f"Illegal PCM label: {label}")
    return label


def normalize_relation(rel_type: str | None) -> str:
    rel_type = (rel_type or "SUPPORTED_BY").strip().upper()
    rel_type = LEGACY_RELATION_MAP.get(rel_type, rel_type)
    if rel_type not in PCM_RELATION_TYPES:
        raise ValueError(f"Illegal PCM relationship type: {rel_type}")
    return rel_type


def canonical_name(name: Any) -> str:
    """Collapse spaces, trim punctuation and keep known acronyms (PCM, HVAC) in their usual case."""
    text = str(name or "").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .;:,")
    if not text:
        return "Unknown"

    compact = text.lower().replace("-", " ")
    compact = re.sub(r"\s+", " ", compact).strip()
    if compact in PRESERVE_CASE_TERMS:
        return PRESERVE_CASE_TERMS[compact]
    if re.fullmatch(r"rubitherm\s*[-_ ]?\s*gr\b.*", compact, flags=re.IGNORECASE):
        suffix = re.sub(r"(?i)^rubitherm\s*[-_ ]?\s*gr\s*", "", text).strip()
        return "Rubitherm GR" if not suffix else f"Rubitherm GR {suffix}"
    if text.isupper() and len(text) <= 8:
        return text
    return " ".join(PRESERVE_CASE_TERMS.get(w.lower(), w) for w in text.split())
