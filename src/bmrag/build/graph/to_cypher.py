"""Graph stage "cypher": write the Cypher MERGE statements for every entity record.

Reads kg_entities/ALL_papers_structured.json and writes kg_cypher/ALL_papers_structured_kg.json
({"records": [...]}), each record with one Cypher string. Nodes are merged by name, except Paper
(paper_id), Case (case_id) and KGRule (rule_id). Deterministic: no LLM call.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bmrag.build.graph.cypher import cypher_props, cypher_string
from bmrag.build.graph.ontology import PCM_NODE_LABELS, canonical_name, normalize_label, normalize_relation
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.stages import BuildContext


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", value or "paper").strip("_").lower()
    return value[:80] or "paper"


def var_name(prefix: str, counter: int) -> str:
    return f"{prefix}{counter}"


def merge_node(var: str, label: str, name: str, extra: dict[str, Any] | None = None) -> str:
    label = normalize_label(label)
    props = {"name": canonical_name(name)}
    props.update(extra or {})
    return f"MERGE ({var}:{label} {{name: {cypher_string(props.pop('name'))}}}) SET {var} += {cypher_props(props)}"


def merge_identity_node(var: str, label: str, key: str, value: str, extra: dict[str, Any] | None = None) -> str:
    label = normalize_label(label)
    props = dict(extra or {})
    return f"MERGE ({var}:{label} {{{key}: {cypher_string(value)}}}) SET {var} += {cypher_props(props)}"


def relation_stmt(left: str, rel_type: str, right: str, props: dict[str, Any] | None = None) -> str:
    rel_type = normalize_relation(rel_type)
    return (
        f"MERGE ({left})-[:{rel_type} {cypher_props(props or {})}]->({right})"
        if props
        else f"MERGE ({left})-[:{rel_type}]->({right})"
    )


def input_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return data["records"]
        if isinstance(data.get("cases"), list):
            return data["cases"]
    return []


def entity_key(entity: dict) -> str:
    return canonical_name(entity.get("normalized_name") or entity.get("name") or entity.get("id"))


def add_entity(
    statements: list[str],
    node_vars: dict[str, str],
    entity_names: set[str],
    entity: dict,
    counter: int,
) -> int:
    name = entity_key(entity)
    if name in node_vars:
        return counter
    label = normalize_label(entity.get("type"))
    if label not in PCM_NODE_LABELS:
        return counter
    var = var_name("n", counter)
    props = dict(entity.get("properties") or {})
    props["entity_id"] = entity.get("id")
    statements.append(merge_node(var, label, name, props))
    node_vars[name] = var
    entity_names.add(name)
    return counter + 1


def ensure_endpoint(
    statements: list[str],
    node_vars: dict[str, str],
    entity_names: set[str],
    name: Any,
    counter: int,
    label: str = "Property",
) -> int:
    cname = canonical_name(name)
    if cname in node_vars:
        return counter
    var = var_name("n", counter)
    statements.append(merge_node(var, label, cname))
    node_vars[cname] = var
    entity_names.add(cname)
    return counter + 1


def add_mechanism_entities(
    record: dict, statements: list[str], node_vars: dict[str, str], entity_names: set[str], counter: int
) -> int:
    for step in record.get("mechanism_chain") or []:
        try:
            cause_label = normalize_label(step.get("cause_type") or "PhysicalMechanism")
            effect_label = normalize_label(step.get("effect_type") or "PerformanceOutcome")
            counter = ensure_endpoint(statements, node_vars, entity_names, step.get("cause"), counter, cause_label)
            counter = ensure_endpoint(statements, node_vars, entity_names, step.get("effect"), counter, effect_label)
        except ValueError:
            continue
    return counter


def case_to_record(record: dict[str, Any], source_path: Path, index: int) -> dict[str, Any]:
    record_type = record.get("record_type") or "case"
    paper_title = record.get("paper_title") or source_path.stem
    source_file = record.get("source_file") or source_path.name
    record_id = record.get("case_id") or record.get("rule_id") or f"{slug(paper_title)}_{record_type}_{index + 1}"
    paper_id = record.get("paper_id") or slug(source_file)
    statements: list[str] = []
    entity_names: set[str] = {paper_title, record_id}
    node_vars: dict[str, str] = {}

    statements.append(
        merge_identity_node(
            "paper",
            "Paper",
            "paper_id",
            paper_id,
            {"title": paper_title, "source_file": source_file, "paper_type": record.get("paper_type")},
        )
    )
    node_vars[canonical_name(paper_title)] = "paper"

    if record_type == "rule":
        statements.append(
            merge_identity_node(
                "record",
                "KGRule",
                "rule_id",
                record_id,
                {"paper_title": paper_title, "evidence_sentence": record.get("case_summary")},
            )
        )
        statements.append(relation_stmt("paper", "HAS_RULE", "record"))
    else:
        statements.append(
            merge_identity_node(
                "record",
                "Case",
                "case_id",
                record_id,
                {
                    "source_file": source_file,
                    "paper_title": paper_title,
                    "case_summary": record.get("case_summary"),
                    "evidence_snippets": record.get("evidence_snippets") or [],
                    "conditional_context": record.get("conditional_context") or {},
                },
            )
        )
        statements.append(relation_stmt("paper", "HAS_CASE", "record"))
    node_vars[canonical_name(record_id)] = "record"

    counter = 0
    for ent in record.get("entities") or []:
        if isinstance(ent, dict):
            counter = add_entity(statements, node_vars, entity_names, ent, counter)
        else:
            counter = ensure_endpoint(statements, node_vars, entity_names, ent, counter)
    counter = add_mechanism_entities(record, statements, node_vars, entity_names, counter)

    for rel in record.get("relations") or []:
        if not isinstance(rel, dict):
            continue
        try:
            counter = ensure_endpoint(statements, node_vars, entity_names, rel.get("source"), counter)
            counter = ensure_endpoint(statements, node_vars, entity_names, rel.get("target"), counter)
            left = node_vars[canonical_name(rel.get("source"))]
            right = node_vars[canonical_name(rel.get("target"))]
            props = dict(rel.get("properties") or {})
            props.update(
                {
                    "confidence": rel.get("confidence"),
                    "case_id": rel.get("case_id") or record_id,
                    "evidence": rel.get("evidence") or [],
                }
            )
            statements.append(relation_stmt(left, rel.get("predicate"), right, props))
        except ValueError:
            continue

    for step in record.get("mechanism_chain") or []:
        try:
            left = node_vars[canonical_name(step.get("cause"))]
            right = node_vars[canonical_name(step.get("effect"))]
            statements.append(
                relation_stmt(
                    left,
                    step.get("relation"),
                    right,
                    {
                        "mechanism_step": step.get("step"),
                        "case_id": step.get("case_id") or record_id,
                        "confidence": step.get("confidence"),
                        "mechanism_note": step.get("mechanism_note"),
                        "condition": step.get("condition") or record.get("conditional_context") or {},
                    },
                )
            )
        except (KeyError, ValueError):
            continue

    cypher = ";\n".join(statements) + ";"
    return {
        "paper_id": paper_id,
        "case_id": record_id,
        "record_type": record_type,
        "source_file": source_file,
        "source_json": str(source_path),
        "case_summary": record.get("case_summary"),
        "conditional_context": record.get("conditional_context") or {},
        "entities": sorted(entity_names),
        "cypher": cypher,
        "raw_case": record.get("raw_case") or {},
        "kg_rule": record.get("kg_rule"),
    }


def run(ctx: BuildContext) -> dict[str, Any]:
    source, target = ctx.layout.entity_records, ctx.layout.cypher_records
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the entities stage first")
    records = [case_to_record(item, source, i) for i, item in enumerate(input_records(read_json(source)))]
    write_json(target, {"records": records})
    return {"records": len(records)}
