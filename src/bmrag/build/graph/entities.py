"""Graph stage "entities": turn the extracted cases and rules into entity and relation records.

Reads kg_cases/ALL_papers.json and writes kg_entities/ALL_papers_structured.json: one record per
case and per rule, each with its entities, relations, evidence snippets and (for cases) the
mechanism chain. Deterministic: no LLM call.

Ids come from the extraction model. Two papers can return the same case or rule id, and the
stores and the graph key records by id alone, so the stage reports duplicates (an error with --strict).
The profile paper_v1_unique_ids renames the ids (see RecordIdSettings): it adds "case_" or
"rule_" where the id lacks it and appends a key of the paper's PDF, so "R1" becomes
"rule_R1@3f2a9c1e".
The id before renaming is kept as local_case_id / local_rule_id.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bmrag.build.graph.ontology import canonical_name, normalize_label, normalize_relation
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.profiles import RecordIdSettings
from bmrag.build.stages import BuildContext


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "|".join(str(part) for part in parts if part not in (None, "", []))
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:10]}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value or "paper").strip("_").lower()[:80] or "paper"


def first_present(data: dict, *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if value not in (None, "", []):
            return value
    return None


def as_list(value: Any) -> list:
    if value in (None, "", []):
        return []
    return value if isinstance(value, list) else [value]


def compact_value(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(compact_value(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k}={compact_value(v)}" for k, v in value.items() if v not in (None, "", []))
    return str(value)


def prompt_relation(value: str | None) -> str:
    rel = (value or "depends_on").strip().lower()
    mapping = {
        "favors": "FAVORS",
        "reduces": "REDUCES",
        "increases": "INCREASES",
        "shifts": "SHIFTS",
        "delays": "DELAYS",
        "requires": "REQUIRES",
        "conflicts_with": "CONFLICTS_WITH",
        "depends_on": "DEPENDS_ON",
        "not_supported_as_universal": "NOT_SUPPORTED_AS_UNIVERSAL",
    }
    return normalize_relation(mapping.get(rel, rel.upper()))


def entity(label: str, name: Any, evidence: list[dict] | None = None, **properties: Any) -> dict:
    label = normalize_label(label)
    normalized = canonical_name(name)
    return {
        "id": stable_id(label.lower(), normalized),
        "name": str(name) if name is not None else "Unknown",
        "type": label,
        "normalized_name": normalized,
        "properties": {k: v for k, v in properties.items() if v not in (None, "", [])},
        "evidence": evidence or [],
    }


def relation(
    source: Any,
    rel_type: str,
    target: Any,
    record_id: str,
    evidence: list[dict] | None = None,
    confidence: str = "medium",
    **properties: Any,
) -> dict:
    return {
        "source": canonical_name(source),
        "predicate": normalize_relation(rel_type),
        "target": canonical_name(target),
        "properties": {k: v for k, v in properties.items() if v not in (None, "", [])},
        "evidence": evidence or [],
        "confidence": confidence,
        "case_id": record_id,
    }


def evidence(field: str, value: Any, source: str) -> list[dict]:
    if value in (None, "", []):
        return []
    return [{"field": field, "value": value, "source": source}]


def layer_text(layers: list) -> str | None:
    if not layers:
        return None
    parts = []
    for layer in layers:
        if isinstance(layer, dict):
            material = layer.get("material") or layer.get("name") or layer.get("layer") or "unknown material"
            thickness = first_present(layer, "thickness", "thickness_mm", "thickness_value")
            if thickness not in (None, "", []):
                parts.append(f"{material}({thickness})")
            else:
                parts.append(f"{material}(thickness unknown)")
        else:
            parts.append(str(layer))
    return "|".join(parts)


def normalize_case(case: dict, paper: dict, source_path: Path, index: int) -> dict:
    metadata = paper.get("paper_metadata") or {}
    source_file = metadata.get("source_file") or case.get("source_file") or source_path.name
    paper_title = metadata.get("paper_title") or case.get("paper_title") or source_path.stem
    case_id = case.get("case_id") or f"{slug(paper_title)}_case_{index + 1}"
    layers = as_list(first_present(case, "wall_layers_exterior_to_interior", "wall_layers"))
    wall_assembly = first_present(
        case,
        "wall_layers_description_with_thickness",
        "wall_layers_description",
    ) or layer_text(layers)
    pcm_material = first_present(case, "pcm_material", "pcm_type")
    studied_position = first_present(case, "studied_pcm_position", "installation_position")
    optimal_position = first_present(case, "optimal_pcm_position", "optimal_position")
    climate = first_present(
        case, "climate_location", "formal_climate_zone", "climate_description", "climate_zone_koppen"
    )
    season = first_present(case, "season", "climate_dominant_season")
    orientation = first_present(case, "orientation", "wall_orientation")
    metrics = as_list(case.get("evaluation_metrics"))
    mechanism = first_present(case, "mechanism_explanation", "climate_position_rule")
    summary = case.get("case_summary") or (
        f"Case {case_id} studies {pcm_material or 'unknown PCM'} at "
        f"{studied_position or 'unknown position'} in {wall_assembly or 'unknown wall assembly'} "
        f"under {climate or 'unknown climate'}; main conclusion: "
        f"{first_present(case, 'main_conclusion', 'main_conclusions') or 'not reported'}"
    )
    missing = as_list(first_present(case, "missing_information", "missing_or_uncertain_fields"))
    if case.get("source_study_reference") in (None, "", []) and (
        paper.get("paper_type") == "review_paper" or case.get("evidence_type") == "cited original study"
    ):
        missing.append("source_study_reference")

    context = {
        "climate": climate,
        "season": season,
        "orientation": orientation,
        "wall_assembly": wall_assembly,
        "pcm_position": studied_position,
        "optimal_pcm_position": optimal_position,
        "performance_metrics": metrics,
    }
    snippets = []
    for field in (
        "case_id",
        "case_name",
        "evidence_type",
        "source_study_reference",
        "study_method",
        "simulation_tool_or_experiment",
        "climate_location",
        "formal_climate_zone",
        "season",
        "orientation",
        "wall_layers_exterior_to_interior",
        "pcm_material",
        "phase_change_temperature",
        "latent_heat",
        "pcm_thickness_or_quantity",
        "studied_pcm_position",
        "optimal_pcm_position",
        "evaluation_metrics",
        "quantitative_results",
        "main_conclusion",
        "mechanism_explanation",
        "missing_information",
    ):
        snippets.extend(evidence(field, case.get(field), source_file))

    entities = [
        entity(
            "Paper",
            paper_title,
            paper_id=metadata.get("paper_id"),
            source_file=source_file,
            paper_type=paper.get("paper_type"),
        ),
        entity(
            "Case",
            case_id,
            case_name=case.get("case_name"),
            evidence_type=case.get("evidence_type"),
            case_summary=summary,
        ),
    ]
    relations = [relation(paper_title, "HAS_CASE", case_id, case_id, confidence="high")]

    if case.get("study_method"):
        entities.append(entity("StudyMethod", case["study_method"]))
        relations.append(relation(case_id, "HAS_METHOD", case["study_method"], case_id, confidence="high"))
    if case.get("source_study_reference"):
        entities.append(entity("SourceStudy", case["source_study_reference"]))
        relations.append(
            relation(case_id, "HAS_SOURCE_STUDY", case["source_study_reference"], case_id, confidence="high")
        )
    if case.get("simulation_tool_or_experiment"):
        entities.append(entity("SimulationTool", case["simulation_tool_or_experiment"]))
        relations.append(relation(case_id, "SUPPORTED_BY", case["simulation_tool_or_experiment"], case_id))
    if pcm_material:
        entities.append(entity("PCMMaterial", pcm_material, material_class=case.get("pcm_material_class")))
        relations.append(relation(case_id, "USES_MATERIAL", pcm_material, case_id, confidence="high"))
    if wall_assembly:
        entities.append(
            entity("WallAssembly", wall_assembly, wall_layers=layers, insulation_layer=case.get("insulation_layer"))
        )
        relations.append(relation(case_id, "HAS_LAYER", wall_assembly, case_id, confidence="high", wall_layers=layers))
    for layer in layers:
        if isinstance(layer, dict) and layer.get("material"):
            layer_name = f"{layer.get('index_from_outside', '?')}: {layer['material']}"
            entities.append(entity("WallLayer", layer_name, **{k: v for k, v in layer.items() if k != "material"}))
            if wall_assembly:
                relations.append(relation(wall_assembly, "HAS_LAYER", layer_name, case_id, confidence="high"))
    if studied_position:
        entities.append(entity("WallPosition", studied_position))
        relations.append(relation(case_id, "PLACED_AT", studied_position, case_id, confidence="high"))
    if optimal_position:
        entities.append(entity("WallPosition", optimal_position, role="optimal_pcm_position"))
        relations.append(
            relation(
                case_id,
                "FAVORS",
                optimal_position,
                case_id,
                confidence="medium",
                condition=case.get("optimal_position_condition"),
            )
        )
    if climate:
        entities.append(entity("ClimateCondition", climate, formal_climate_zone=case.get("formal_climate_zone")))
        relations.append(relation(case_id, "STUDIED_IN", climate, case_id, confidence="high"))
    if season:
        entities.append(entity("Season", season))
        relations.append(relation(case_id, "HAS_CONDITION", season, case_id, context="season"))
    if orientation:
        entities.append(entity("Orientation", orientation))
        relations.append(relation(case_id, "HAS_CONDITION", orientation, case_id, context="orientation"))

    property_fields = {
        "phase_change_temperature": "Phase Change Temperature",
        "latent_heat": "Latent Heat",
        "pcm_thickness_or_quantity": "PCM Thickness Or Quantity",
        "pcm_density": "PCM Density",
        "pcm_thermal_conductivity": "PCM Thermal Conductivity",
        "pcm_specific_heat": "PCM Specific Heat",
    }
    for field, name in property_fields.items():
        if case.get(field) not in (None, "", []):
            entities.append(entity("Property", name, value=case.get(field)))
            relations.append(relation(case_id, "HAS_PROPERTY", name, case_id, confidence="high", value=case.get(field)))
    for metric in metrics:
        entities.append(entity("Metric", metric))
        relations.append(relation(case_id, "MEASURES", metric, case_id, confidence="high"))
    if case.get("quantitative_results"):
        outcome_name = compact_value(case["quantitative_results"])
        entities.append(entity("PerformanceOutcome", outcome_name))
        relations.append(relation(case_id, "ACHIEVES", outcome_name, case_id, confidence="medium"))
    if mechanism:
        entities.append(entity("PhysicalMechanism", mechanism))
        relations.append(relation(case_id, "ENABLES", mechanism, case_id, confidence="medium"))
    for gap in missing:
        entities.append(entity("EvidenceGap", gap))
        relations.append(relation(case_id, "HAS_EVIDENCE_GAP", gap, case_id, confidence="high"))

    return {
        "record_type": "case",
        "paper_title": paper_title,
        "paper_id": metadata.get("paper_id"),
        "paper_type": paper.get("paper_type"),
        "source_file": source_file,
        "case_id": case_id,
        "case_summary": summary,
        "evidence_snippets": snippets,
        "conditional_context": context,
        "entities": entities,
        "relations": relations,
        "mechanism_chain": normalize_mechanism_chain(case, case_id, context),
        "raw_case": case,
    }


def normalize_mechanism_chain(case: dict, case_id: str, context: dict) -> list[dict]:
    chain = []
    for index, step in enumerate(as_list(case.get("mechanism_chain_triplets")), start=1):
        if not isinstance(step, dict):
            continue
        try:
            chain.append(
                {
                    "step": int(step.get("step") or index),
                    "cause": canonical_name(step.get("cause")),
                    "cause_type": normalize_label(step.get("cause_type") or "PhysicalMechanism"),
                    "relation": normalize_relation(step.get("relation") or "LEADS_TO"),
                    "effect": canonical_name(step.get("effect")),
                    "effect_type": normalize_label(step.get("effect_type") or "PerformanceOutcome"),
                    "case_id": case_id,
                    "condition": context,
                    "confidence": step.get("confidence") or "medium",
                    "mechanism_note": step.get("mechanism_note"),
                }
            )
        except ValueError:
            continue
    return chain


def normalize_rule(rule: dict, paper: dict, case_ids: set[str], source_path: Path, index: int) -> dict:
    metadata = paper.get("paper_metadata") or {}
    paper_title = metadata.get("paper_title") or source_path.stem
    source_file = metadata.get("source_file") or source_path.name
    rule_id = rule.get("rule_id") or f"{slug(paper_title)}_rule_{index + 1}"
    evidence_case_ids = [cid for cid in as_list(rule.get("evidence_case_ids")) if cid in case_ids]
    condition_nodes = as_list(rule.get("condition_nodes"))
    outcome = rule.get("outcome_node") or "Unspecified outcome"
    objective = rule.get("performance_objective")
    missing = as_list(rule.get("missing_or_uncertain_fields"))

    entities = [
        entity(
            "Paper",
            paper_title,
            paper_id=metadata.get("paper_id"),
            source_file=source_file,
            paper_type=paper.get("paper_type"),
        ),
        entity(
            "KGRule",
            rule_id,
            relation=rule.get("relation"),
            confidence=rule.get("confidence"),
            evidence_sentence=rule.get("evidence_sentence"),
        ),
    ]
    relations = [relation(paper_title, "HAS_RULE", rule_id, rule_id, confidence="high")]
    for case_id in evidence_case_ids:
        entities.append(entity("Case", case_id))
        relations.append(
            relation(case_id, "SUPPORTS_RULE", rule_id, rule_id, confidence=rule.get("confidence") or "medium")
        )
    for node in condition_nodes:
        label = infer_condition_label(node)
        entities.append(entity(label, node))
        relations.append(relation(rule_id, "DEPENDS_ON", node, rule_id, confidence=rule.get("confidence") or "medium"))
    if objective:
        entities.append(entity("PerformanceObjective", objective))
        relations.append(relation(rule_id, "HAS_OBJECTIVE", objective, rule_id, confidence="high"))
    outcome_label = (
        "WallPosition" if "position" in str(outcome).lower() or "side" in str(outcome).lower() else "PerformanceOutcome"
    )
    entities.append(entity(outcome_label, outcome))
    relations.append(
        relation(
            rule_id,
            prompt_relation(rule.get("relation")),
            outcome,
            rule_id,
            confidence=rule.get("confidence") or "medium",
        )
    )
    for gap in missing:
        entities.append(entity("EvidenceGap", gap))
        relations.append(relation(rule_id, "HAS_EVIDENCE_GAP", gap, rule_id, confidence="high"))
    if not evidence_case_ids:
        gap = "kg_ready_rule has no valid evidence_case_ids"
        entities.append(entity("EvidenceGap", gap))
        relations.append(relation(rule_id, "HAS_EVIDENCE_GAP", gap, rule_id, confidence="high"))

    return {
        "record_type": "rule",
        "paper_title": paper_title,
        "paper_id": metadata.get("paper_id"),
        "paper_type": paper.get("paper_type"),
        "source_file": source_file,
        "rule_id": rule_id,
        "case_id": rule_id,
        "case_summary": rule.get("evidence_sentence"),
        "entities": entities,
        "relations": relations,
        "kg_rule": rule,
        "raw_case": {},
    }


def infer_condition_label(node: Any) -> str:
    text = str(node).lower()
    if "summer" in text or "winter" in text or "season" in text:
        return "Season"
    if any(word in text for word in ("south", "north", "east", "west", "orientation")):
        return "Orientation"
    if any(word in text for word in ("wall", "assembly", "layer", "insulation")):
        return "WallAssembly"
    if any(word in text for word in ("load", "comfort", "energy", "flux", "temperature", "metric", "objective")):
        return "PerformanceObjective"
    return "ClimateCondition"


def paper_objects(data: Any, source_path: Path) -> list[dict]:
    if isinstance(data, dict) and isinstance(data.get("papers"), list):
        return [p for p in data["papers"] if isinstance(p, dict)]
    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        return [data]
    if isinstance(data, dict) and isinstance(data.get("records"), list):
        return [
            {
                "paper_metadata": {"paper_title": source_path.stem, "source_file": source_path.name},
                "cases": data["records"],
                "kg_ready_rules": [],
            }
        ]
    if isinstance(data, list):
        return [
            {
                "paper_metadata": {"paper_title": source_path.stem, "source_file": source_path.name},
                "paper_type": "research_paper",
                "cases": data,
                "kg_ready_rules": [],
            }
        ]
    return []


def paper_key(paper: dict) -> str:
    """Short key of a paper: the first 8 hex digits of its PDF's SHA-256.

    The hash is the one the extract stage records in "source_pdf", not anything the model wrote,
    so the key stays the same when the PDF is renamed or extracted again, and differs between papers.
    """
    digest = (paper.get("source_pdf") or {}).get("sha256")
    if not digest:
        title = (paper.get("paper_metadata") or {}).get("paper_title") or "a paper"
        raise ValueError(
            f"{title!r} has no source_pdf hash. Run the extract stage with this profile first; "
            "it reuses the existing records and calls no API."
        )
    return str(digest)[:8]


# The prefixes bmrag.agents.judge routes cited ids by; tests check that the two agree.
CASE_ID_START = re.compile(r"^case[_A-Za-z0-9\-]")
RULE_ID_START = re.compile(r"^(?:rule[_A-Za-z0-9\-]|Rule\d)")


def renamed_id(local: str, kind: str, start: re.Pattern[str], key: str | None, typed_prefixes: bool) -> str:
    """local with "<kind>_" in front when typed_prefixes and it lacks one, and "@<key>" after when key is set."""
    new = f"{kind}_{local}" if typed_prefixes and not start.match(local) else local
    return f"{new}@{key}" if key else new


def rename_ids(paper: dict, source_path: Path, settings: RecordIdSettings) -> dict:
    """A copy of the paper with its case and rule ids renamed as settings say; the input is not changed.

    Ids the model left empty get the same fallback as normalize_case / normalize_rule first, so
    every id is renamed. Rules' evidence_case_ids use the same map, so links within the paper
    still hold; an evidence id naming no case of this paper is left as it is.
    """
    key = paper_key(paper) if settings.paper_scoped_ids else None
    metadata = paper.get("paper_metadata") or {}
    case_map: dict[str, str] = {}
    cases = []
    case_index = 0
    for case in paper.get("cases", []):
        if isinstance(case, dict):
            title = metadata.get("paper_title") or case.get("paper_title") or source_path.stem
            local = str(case.get("case_id") or f"{slug(title)}_case_{case_index + 1}")
            case_map[local] = renamed_id(local, "case", CASE_ID_START, key, settings.typed_prefixes)
            case = {**case, "case_id": case_map[local], "local_case_id": local}
            case_index += 1
        cases.append(case)
    rules = []
    rule_index = 0
    for rule in paper.get("kg_ready_rules", []):
        if isinstance(rule, dict):
            title = metadata.get("paper_title") or source_path.stem
            local = str(rule.get("rule_id") or f"{slug(title)}_rule_{rule_index + 1}")
            rule = {**rule, "rule_id": renamed_id(local, "rule", RULE_ID_START, key, settings.typed_prefixes)}
            rule["local_rule_id"] = local
            if isinstance(rule.get("evidence_case_ids"), list):
                rule["evidence_case_ids"] = [case_map.get(str(cid), cid) for cid in rule["evidence_case_ids"]]
            rule_index += 1
        rules.append(rule)
    return {**paper, "cases": cases, "kg_ready_rules": rules}


def build_records(data: Any, source_path: Path, record_ids: RecordIdSettings | None = None) -> list[dict]:
    """Case records, then rule records, for every paper in the extract stage's output."""
    record_ids = record_ids or RecordIdSettings()
    records = []
    for paper in paper_objects(data, source_path):
        if record_ids.paper_scoped_ids or record_ids.typed_prefixes:
            paper = rename_ids(paper, source_path, record_ids)
        cases = [c for c in paper.get("cases", []) if isinstance(c, dict)]
        case_ids = {c.get("case_id") for c in cases if c.get("case_id")}
        for index, case in enumerate(cases):
            records.append(normalize_case(case, paper, source_path, index))
        for index, rule in enumerate([r for r in paper.get("kg_ready_rules", []) if isinstance(r, dict)]):
            records.append(normalize_rule(rule, paper, case_ids, source_path, index))
    return records


def duplicate_ids(records: list[dict]) -> dict[str, list[str]]:
    """Case or rule ids used by more than one record, with the source files that use them."""
    counts = Counter((r.get("record_type"), r.get("case_id")) for r in records)
    duplicates: dict[str, list[str]] = {}
    for (record_type, record_id), count in counts.items():
        if count > 1:
            sources = sorted(
                {
                    str(r.get("source_file"))
                    for r in records
                    if r.get("record_type") == record_type and r.get("case_id") == record_id
                }
            )
            duplicates[f"{record_type}:{record_id}"] = sources
    return duplicates


def run(ctx: BuildContext) -> dict[str, Any]:
    source, target = ctx.layout.extracted_papers, ctx.layout.entity_records
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the extract stage first")
    try:
        records = build_records(read_json(source), source, ctx.profile.record_ids)
    except ValueError as exc:
        raise SystemExit(f"entities: {exc}") from None
    write_json(target, records)
    duplicates = duplicate_ids(records)
    if duplicates:
        ctx.fail_or_warn(
            f"entities: {len(duplicates)} id(s) used by more than one record; the stores and the "
            f"graph keep only one of each: {duplicates}"
        )
    counts = Counter(r["record_type"] for r in records)
    return {"cases": counts["case"], "rules": counts["rule"], "duplicate_ids": len(duplicates)}
