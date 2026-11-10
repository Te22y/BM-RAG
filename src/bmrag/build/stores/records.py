"""Turn standardized KG records into the case and rule bundles of the knowledge-graph index.

Each bundle has a retrieval_text, which is what gets embedded, the metadata stored with the
vector, and the full record, which retrieval returns. Stores are keyed by case_id / rule_id.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from bmrag.build.graph.cypher import split_cypher_statements
from bmrag.build.jsonio import read_json

NODE_RE = re.compile(
    r"\((?P<var>[A-Za-z_][A-Za-z0-9_]*):(?P<label>[A-Za-z_][A-Za-z0-9_]*)\s*\{(?P<props>.*?)\}\)",
    flags=re.DOTALL,
)
REL_RE = re.compile(
    r"\((?P<src>[A-Za-z_][A-Za-z0-9_]*)\)\s*-\s*\[:(?P<rel>[A-Za-z_][A-Za-z0-9_]*)(?:\s*\{(?P<props>.*?)\})?\]\s*->\s*\((?P<dst>[A-Za-z_][A-Za-z0-9_]*)\)",
    flags=re.DOTALL,
)
PROP_RE = re.compile(r"(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*:\s*'(?P<value>(?:\\'|[^'])*)'")


def scalar_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(scalar_text(item) for item in value if scalar_text(item))
    if isinstance(value, dict):
        return "; ".join(f"{key}: {scalar_text(val)}" for key, val in value.items() if scalar_text(val))
    return str(value).strip()


def first_nonempty(*values: Any) -> str:
    for value in values:
        text = scalar_text(value)
        if text:
            return text
    return ""


def compact(value: Any, max_len: int = 500) -> str:
    text = scalar_text(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_len:
        return text[: max_len - 3].rstrip() + "..."
    return text


def parse_cypher_props(props_text: str) -> dict[str, str]:
    props = {}
    for match in PROP_RE.finditer(props_text or ""):
        props[match.group("key")] = match.group("value").replace("\\'", "'")
    return props


def parse_internal_kg(cypher: str) -> dict[str, Any]:
    """Create a lightweight LLM-readable view while preserving original Cypher."""
    node_by_var: dict[str, dict[str, Any]] = {}
    nodes_text: list[str] = []
    relations_text: list[str] = []

    for stmt in split_cypher_statements(cypher or ""):
        for match in NODE_RE.finditer(stmt):
            var = match.group("var")
            props = parse_cypher_props(match.group("props"))
            label = match.group("label")
            name = first_nonempty(
                props.get("name"), props.get("case_id"), props.get("paper_id"), props.get("rule_id"), var
            )
            node_by_var[var] = {"label": label, "name": name, "properties": props}
            nodes_text.append(f"{var}: {label} '{name}'")

        for match in REL_RE.finditer(stmt):
            src_var = match.group("src")
            dst_var = match.group("dst")
            rel = match.group("rel")
            rel_props = parse_cypher_props(match.group("props") or "")
            src = node_by_var.get(src_var, {"label": src_var, "name": src_var})
            dst = node_by_var.get(dst_var, {"label": dst_var, "name": dst_var})
            prop_text = compact(rel_props, max_len=220)
            suffix = f" {{{prop_text}}}" if prop_text else ""
            relations_text.append(f"{src['label']} '{src['name']}' --{rel}--> {dst['label']} '{dst['name']}'{suffix}")

    return {
        "nodes_text": dedupe(nodes_text),
        "relations_text": dedupe(relations_text),
        "cypher_original": cypher or "",
    }


def dedupe(items: list[str]) -> list[str]:
    seen = set()
    unique = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def location_entities(record: dict[str, Any]) -> list[str]:
    terms = []
    for entity in record.get("entities") or []:
        text = scalar_text(entity)
        lower = text.lower()
        if any(
            token in lower
            for token in (
                "climate",
                "city",
                "region",
                "country",
                "summer",
                "winter",
                "tropical",
                "temperate",
                "cold",
                "hot",
            )
        ):
            terms.append(text)
    return terms


def build_retrieval_text(record: dict[str, Any]) -> str:
    raw_case = record.get("raw_case") or {}
    context = record.get("conditional_context") or {}
    fields = [
        ("case_id", record.get("case_id")),
        ("paper_id", record.get("paper_id")),
        ("source_file", record.get("source_file")),
        ("climate", context.get("climate")),
        ("climate_location", raw_case.get("climate_location")),
        ("formal_climate_zone", raw_case.get("formal_climate_zone")),
        ("weather_file_or_period", raw_case.get("weather_file_or_period")),
        ("season", first_nonempty(context.get("season"), raw_case.get("season"))),
        ("orientation", first_nonempty(context.get("orientation"), raw_case.get("orientation"))),
        ("wall_assembly", context.get("wall_assembly")),
        ("pcm_position", context.get("pcm_position")),
        ("optimal_pcm_position", context.get("optimal_pcm_position")),
        ("performance_metrics", context.get("performance_metrics")),
        ("case_summary", record.get("case_summary")),
        ("location_related_entities", location_entities(record)),
    ]
    lines = [f"{key}: {scalar_text(value)}" for key, value in fields if scalar_text(value)]
    return "\n".join(lines)


def build_rule_retrieval_text(record: dict[str, Any]) -> str:
    rule = record.get("kg_rule") or {}
    internal_kg = parse_internal_kg(record.get("cypher") or "")
    fields = [
        ("rule_id", record.get("case_id")),
        ("paper_id", record.get("paper_id")),
        ("source_file", record.get("source_file")),
        ("case_summary", record.get("case_summary")),
        ("rule_relation", rule.get("relation")),
        ("rule_subject", rule.get("subject")),
        ("rule_object", rule.get("object")),
        ("rule_condition", rule.get("condition")),
        ("evidence_sentence", rule.get("evidence_sentence")),
        ("evidence_case_ids", rule.get("evidence_case_ids")),
        ("rule_entities", record.get("entities")),
        ("rule_nodes", internal_kg.get("nodes_text")),
        ("rule_relations", internal_kg.get("relations_text")),
    ]
    lines = [f"{key}: {scalar_text(value)}" for key, value in fields if scalar_text(value)]
    return "\n".join(lines)


def metadata_for_record(record: dict[str, Any]) -> dict[str, str]:
    raw_case = record.get("raw_case") or {}
    context = record.get("conditional_context") or {}
    return {
        "case_id": scalar_text(record.get("case_id")),
        "paper_id": scalar_text(record.get("paper_id")),
        "source_file": scalar_text(record.get("source_file")),
        "record_type": scalar_text(record.get("record_type")),
        "climate": first_nonempty(context.get("climate"), raw_case.get("climate_location")),
        "season": first_nonempty(context.get("season"), raw_case.get("season")),
        "orientation": first_nonempty(context.get("orientation"), raw_case.get("orientation")),
        "paper_type": scalar_text(raw_case.get("paper_type")),
    }


def metadata_for_rule(record: dict[str, Any]) -> dict[str, str]:
    rule = record.get("kg_rule") or {}
    return {
        "rule_id": scalar_text(record.get("case_id")),
        "case_id": scalar_text(record.get("case_id")),
        "paper_id": scalar_text(record.get("paper_id")),
        "source_file": scalar_text(record.get("source_file")),
        "record_type": scalar_text(record.get("record_type")),
        "rule_relation": scalar_text(rule.get("relation")),
        "evidence_case_ids": scalar_text(rule.get("evidence_case_ids")),
    }


def build_rule_links(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    links: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if record.get("record_type") != "rule":
            continue
        rule = record.get("kg_rule") or {}
        for case_id in rule.get("evidence_case_ids") or []:
            links.setdefault(str(case_id), []).append(
                {
                    "rule_id": record.get("case_id"),
                    "paper_id": record.get("paper_id"),
                    "source_file": record.get("source_file"),
                    "case_summary": record.get("case_summary"),
                    "kg_rule": rule,
                    "internal_kg": parse_internal_kg(record.get("cypher") or ""),
                }
            )
    return links


def build_rule_store(records: list[dict[str, Any]]) -> dict[str, Any]:
    rule_store = {}
    for record in records:
        if record.get("record_type") != "rule":
            continue
        rule_id = scalar_text(record.get("case_id"))
        if not rule_id:
            continue
        retrieval_text = build_rule_retrieval_text(record)
        rule_store[rule_id] = {
            "rule_id": rule_id,
            "case_id": rule_id,
            "paper_id": record.get("paper_id"),
            "source_file": record.get("source_file"),
            "retrieval_text": retrieval_text,
            "metadata": metadata_for_rule(record),
            "rule_profile": {
                "case_summary": record.get("case_summary"),
                "kg_rule": record.get("kg_rule") or {},
                "entities": record.get("entities") or [],
                "conditional_context": record.get("conditional_context") or {},
            },
            "internal_kg": parse_internal_kg(record.get("cypher") or ""),
            "evidence_case_ids": (record.get("kg_rule") or {}).get("evidence_case_ids") or [],
            "record_original": record,
        }
    return rule_store


def build_case_store(records: list[dict[str, Any]]) -> dict[str, Any]:
    rule_links = build_rule_links(records)
    case_store = {}
    for record in records:
        if record.get("record_type") != "case":
            continue
        case_id = scalar_text(record.get("case_id"))
        if not case_id:
            continue
        retrieval_text = build_retrieval_text(record)
        case_store[case_id] = {
            "case_id": case_id,
            "paper_id": record.get("paper_id"),
            "source_file": record.get("source_file"),
            "retrieval_text": retrieval_text,
            "metadata": metadata_for_record(record),
            "case_profile": {
                "case_summary": record.get("case_summary"),
                "conditional_context": record.get("conditional_context") or {},
                "raw_case": record.get("raw_case") or {},
                "entities": record.get("entities") or [],
            },
            "internal_kg": parse_internal_kg(record.get("cypher") or ""),
            "linked_rules": rule_links.get(case_id, []),
            "record_original": record,
        }
    return case_store


def load_records(input_path: Path) -> list[dict[str, Any]]:
    records = read_json(input_path)
    if not isinstance(records, list):
        raise ValueError(f"Expected a list of KG records in {input_path}")
    return records


def safe_reset_dir(path: Path, workspace_root: Path) -> None:
    resolved = path.resolve()
    root = workspace_root.resolve()
    if root not in resolved.parents and resolved != root:
        raise ValueError(f"Refusing to reset path outside workspace: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
