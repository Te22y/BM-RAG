"""Graph stage "standardize": align entity names across records and enforce the PCM ontology.

Reads kg_cypher/ALL_papers_structured_kg.json and writes to kg_canonical/:
  ALL_papers_structured_standardized.json    records with canonical entity names in their Cypher
  ALL_papers_structured_standardized.cypher  the same Cypher as one file, for the neo4j stage
  canonical_map.json, canonical_alignment_report.json, step6_validation_report.json

Names are grouped by rules (known aliases, singular forms, leading numbers) and the most frequent
form wins. With standardize.use_llm an LLM then reviews the most frequent names in batches.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from bmrag.build import llm_client
from bmrag.build.graph.cypher import (
    NAME_PROP_RE,
    clean_markdown_cypher,
    extract_labels,
    extract_relations,
    split_cypher_statements,
    substitute_canonical_names,
)
from bmrag.build.graph.ontology import PCM_NODE_LABELS, PCM_RELATION_TYPES, canonical_name
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.profiles import StandardizeSettings
from bmrag.build.stages import BuildContext

log = logging.getLogger(__name__)
STAGE = "standardize"

LEADING_INDEX_RE = re.compile(r"^\s*\d+\s*[:)-]\s*")

KNOWN_CANONICAL_ALIASES = {
    "pcm": "Phase Change Material",
    "pcms": "Phase Change Material",
    "phase change material": "Phase Change Material",
    "phase change materials": "Phase Change Material",
    "phase-change material": "Phase Change Material",
    "phase-change materials": "Phase Change Material",
    "latent heat storage material": "Phase Change Material",
    "phase change temperature": "Phase Change Temperature",
    "phase transition temperature": "Phase Change Temperature",
    "melting temperature": "Phase Change Temperature",
    "melting point": "Phase Change Temperature",
    "latent heat": "Latent Heat",
    "heat of fusion": "Latent Heat",
    "pcm thickness": "PCM Thickness Or Quantity",
    "pcm quantity": "PCM Thickness Or Quantity",
    "phase change material thickness": "PCM Thickness Or Quantity",
    "thermal conductivity of pcm": "PCM Thermal Conductivity",
    "pcm thermal conductivity": "PCM Thermal Conductivity",
    "specific heat of pcm": "PCM Specific Heat",
    "pcm specific heat": "PCM Specific Heat",
    "density of pcm": "PCM Density",
    "pcm density": "PCM Density",
    "not reported": "Not Reported",
    "not specified": "Not Reported",
    "not available": "Not Reported",
    "n/a": "Not Reported",
}

SINGULAR_WORDS = {
    "materials": "material",
    "walls": "wall",
    "buildings": "building",
    "temperatures": "temperature",
    "properties": "property",
    "layers": "layer",
    "orientations": "orientation",
    "conditions": "condition",
    "outcomes": "outcome",
    "objectives": "objective",
    "methods": "method",
}


def load_records(path: Path) -> list[dict]:
    data = read_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("records"), list):
            return data["records"]
        if isinstance(data.get("cases"), list):
            return data["cases"]
    return []


def entity_name(entity) -> str:
    if isinstance(entity, dict):
        return str(entity.get("normalized_name") or entity.get("name") or entity.get("id") or "").strip()
    return str(entity).strip()


def extract_cypher_name_values(cypher: str) -> list[str]:
    values = []
    for match in NAME_PROP_RE.finditer(cypher or ""):
        values.append(match.group(2).replace("\\'", "'").strip())
    return [value for value in values if value]


def collect_entity_counts(files: list[Path]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for file in files:
        for rec in load_records(file):
            for entity in rec.get("entities", []):
                name = entity_name(entity)
                if name:
                    counts[name] += 1
            for name in extract_cypher_name_values(rec.get("cypher", "")):
                counts[name] += 1
    return counts


def collect_entities(files: list[Path]) -> set[str]:
    return set(collect_entity_counts(files))


def _simple_key(name: str) -> str:
    text = LEADING_INDEX_RE.sub("", str(name or "")).strip()
    text = re.sub(r"\((pcms?|phase change materials?)\)", "", text, flags=re.IGNORECASE)
    text = text.lower().replace("-", " ").replace("_", " ").replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    words = [SINGULAR_WORDS.get(word, word) for word in text.split()]
    return " ".join(words)


def _alias_key(name: str) -> str:
    key = _simple_key(name)
    key = re.sub(r"\bpcms?\b", "phase change material", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def _looks_like_long_description(name: str) -> bool:
    text = LEADING_INDEX_RE.sub("", name)
    return len(text) > 120 or len(re.findall(r"[A-Za-z0-9]+", text)) > 14 or "|" in text


def _canonical_rank(name: str, count: int) -> tuple[int, int, int, str]:
    stripped = LEADING_INDEX_RE.sub("", name).strip()
    known = 1 if _alias_key(stripped) in KNOWN_CANONICAL_ALIASES else 0
    expanded = 1 if "phase change" in stripped.lower() else 0
    return (known, count, expanded, -len(stripped), stripped.lower())


def _choose_group_canonical(names: list[str], counts: Counter[str]) -> str:
    candidates = [canonical_name(LEADING_INDEX_RE.sub("", name)) for name in names]
    ranked = sorted(
        (
            (candidate, sum(counts[n] for n in names if canonical_name(LEADING_INDEX_RE.sub("", n)) == candidate))
            for candidate in set(candidates)
        ),
        key=lambda item: _canonical_rank(item[0], item[1]),
        reverse=True,
    )
    return ranked[0][0] if ranked else "Unknown"


def build_frequency_alignment(
    raw_entities: set[str],
    counts: Counter[str],
) -> tuple[dict[str, str], list[dict]]:
    mapping = {name: canonical_name(LEADING_INDEX_RE.sub("", name)) for name in sorted(raw_entities)}
    decisions: list[dict] = []

    for raw in sorted(raw_entities):
        key = _alias_key(raw)
        if key in KNOWN_CANONICAL_ALIASES and not _looks_like_long_description(raw):
            canonical = KNOWN_CANONICAL_ALIASES[key]
            if mapping[raw] != canonical:
                decisions.append(
                    {
                        "method": "known_alias",
                        "raw": raw,
                        "from": mapping[raw],
                        "to": canonical,
                        "count": counts[raw],
                    }
                )
                mapping[raw] = canonical

    groups: dict[str, list[str]] = defaultdict(list)
    for raw in raw_entities:
        key = _alias_key(raw)
        if key and not _looks_like_long_description(raw):
            groups[key].append(raw)

    for key, names in sorted(groups.items()):
        canonicals = {mapping[name] for name in names}
        if len(names) < 2 or len(canonicals) < 2:
            continue
        canonical = _choose_group_canonical(names, counts)
        for name in names:
            if mapping[name] != canonical:
                decisions.append(
                    {
                        "method": "frequency_key",
                        "key": key,
                        "raw": name,
                        "from": mapping[name],
                        "to": canonical,
                        "count": counts[name],
                        "group_size": len(names),
                    }
                )
                mapping[name] = canonical

    return mapping, decisions


def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"```(?:json)?", "", text or "", flags=re.IGNORECASE).replace("```", "").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in LLM response")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("LLM response is not a JSON object")
    return parsed


def _llm_batches(items: list[dict], batch_size: int) -> list[list[dict]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def refine_canonical_map_with_llm(
    mapping: dict[str, str],
    counts: Counter[str],
    settings: StandardizeSettings,
    usage: llm_client.UsageLog | None = None,
) -> tuple[dict[str, str], list[dict]]:
    """Let an LLM merge true synonyms among the most frequent names. Returns the map and its decisions."""
    client = llm_client.openai_client()
    model, max_entities, batch_size = settings.llm_model, settings.llm_max_entities, settings.llm_batch_size
    ranked_raw = sorted(mapping, key=lambda name: (counts[name], -len(name)), reverse=True)
    selected = [
        {
            "raw": raw,
            "current_canonical": mapping[raw],
            "count": counts[raw],
        }
        for raw in ranked_raw[:max_entities]
        if not _looks_like_long_description(raw)
    ]

    system = (
        "You align entity names for a PCM building-envelope knowledge graph. "
        "Return strict JSON only. Merge only true synonyms, abbreviations, spelling/case/plural variants, "
        "and the same physical property named differently. Do not merge broader/narrower concepts, "
        "different materials, different wall layers, paper titles, case descriptions, or numeric values."
    )
    decisions: list[dict] = []
    refined = dict(mapping)

    for batch in _llm_batches(selected, batch_size):
        user = (
            "For each item, choose the best canonical entity name. Prefer concise scientific names and "
            "high-frequency forms. Keep the current canonical if uncertain.\n\n"
            "Return JSON with this schema:\n"
            '{"mapping": {"RAW_NAME": "CANONICAL_NAME"}, "notes": ["short optional notes"]}\n\n'
            f"Items:\n{json.dumps(batch, ensure_ascii=False, indent=2)}"
        )
        kwargs = llm_client.chat_kwargs(
            model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            max_tokens=settings.llm_max_tokens,
            temperature=0,
            min_gpt5_completion_tokens=settings.min_gpt5_completion_tokens,
        )
        response = llm_client.complete(client, kwargs, usage, STAGE)
        parsed = _parse_json_object(response.choices[0].message.content)
        llm_mapping = parsed.get("mapping", {})
        if not isinstance(llm_mapping, dict):
            continue
        batch_raw = {item["raw"] for item in batch}
        for raw, canonical in llm_mapping.items():
            if raw not in batch_raw:
                continue
            canonical = canonical_name(canonical)
            if not canonical or canonical == "Unknown":
                continue
            if refined[raw] != canonical:
                decisions.append(
                    {
                        "method": "llm",
                        "raw": raw,
                        "from": refined[raw],
                        "to": canonical,
                        "count": counts[raw],
                    }
                )
                refined[raw] = canonical

    return refined, decisions


def build_canonical_map(
    raw_entities: set[str],
    counts: Counter[str] | None = None,
    settings: StandardizeSettings | None = None,
    usage: llm_client.UsageLog | None = None,
) -> tuple[dict[str, str], dict]:
    """Raw name -> canonical name, and a report of the decisions."""
    settings = settings or StandardizeSettings()
    use_llm = settings.use_llm
    counts = counts or Counter(raw_entities)
    mapping, frequency_decisions = build_frequency_alignment(raw_entities, counts)
    llm_decisions: list[dict] = []
    if use_llm:
        mapping, llm_decisions = refine_canonical_map_with_llm(mapping, counts, settings, usage)
    report = {
        "raw_entity_count": len(raw_entities),
        "canonical_entity_count": len(set(mapping.values())),
        "mention_count": sum(counts.values()),
        "frequency_decision_count": len(frequency_decisions),
        "llm_enabled": use_llm,
        "llm_model": settings.llm_model if use_llm else None,
        "llm_decision_count": len(llm_decisions),
        "decisions": (frequency_decisions + llm_decisions)[:500],
        "top_entities": [{"name": name, "count": count} for name, count in counts.most_common(100)],
    }
    return mapping, report


def validate_cypher(cypher: str, *, source: str, case_id: str) -> list[dict]:
    errors = []
    if "`" in cypher or "```" in cypher:
        errors.append({"source": source, "case_id": case_id, "error": "Markdown/backtick content remains"})
    statements = split_cypher_statements(cypher)
    if not statements:
        errors.append({"source": source, "case_id": case_id, "error": "No Cypher statements"})
        return errors
    labels = extract_labels(cypher)
    rels = extract_relations(cypher)
    for label in sorted(labels - PCM_NODE_LABELS):
        errors.append({"source": source, "case_id": case_id, "error": f"Illegal label: {label}"})
    for rel in sorted(rels - PCM_RELATION_TYPES):
        errors.append({"source": source, "case_id": case_id, "error": f"Illegal relationship type: {rel}"})
    for stmt in statements:
        if stmt.count("(") != stmt.count(")") or stmt.count("{") != stmt.count("}"):
            errors.append(
                {"source": source, "case_id": case_id, "error": "Unbalanced delimiters", "sample": stmt[:240]}
            )
        if not re.search(r"\b(MERGE|MATCH|CREATE|SET|WITH|RETURN)\b", stmt, flags=re.IGNORECASE):
            errors.append(
                {"source": source, "case_id": case_id, "error": "Empty or non-Cypher statement", "sample": stmt[:240]}
            )
    return errors


def save_cypher(records: list[dict], out_path: Path) -> None:
    lines = [
        "// PCM standardized Cypher",
        f"// Total records: {len(records)}",
        "",
    ]
    for rec in records:
        lines.append(f"// source_file: {rec.get('source_file', '')} | case_id: {rec.get('case_id', '')}")
        lines.append(rec.get("cypher", "").strip())
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def standardize_records(
    records: list[dict], canonical_map: dict[str, str], source_name: str
) -> tuple[list[dict], list[dict]]:
    """Records with canonical names substituted into their entities and Cypher, and the validation errors."""
    standardized = []
    errors = []
    for rec in records:
        new_rec = dict(rec)
        new_rec["entities"] = sorted(
            {
                canonical_map.get(entity_name(e), canonical_name(entity_name(e)))
                for e in rec.get("entities", [])
                if entity_name(e)
            }
        )
        cypher = clean_markdown_cypher(rec.get("cypher", ""))
        cypher = substitute_canonical_names(cypher, canonical_map)
        new_rec["cypher"] = cypher
        errors.extend(
            validate_cypher(cypher, source=rec.get("source_file") or source_name, case_id=rec.get("case_id", ""))
        )
        standardized.append(new_rec)
    return standardized, errors


def run(ctx: BuildContext) -> dict[str, Any]:
    layout, settings = ctx.layout, ctx.profile.standardize
    source = layout.cypher_records
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the cypher stage first")
    entity_counts = collect_entity_counts([source])
    canonical_map, alignment_report = build_canonical_map(set(entity_counts), entity_counts, settings, ctx.record.usage)
    write_json(layout.canonical_map, canonical_map)
    write_json(layout.alignment_report, alignment_report)

    records, errors = standardize_records(load_records(source), canonical_map, source.name)
    save_cypher(records, layout.standardized_cypher)
    write_json(layout.standardized_records, records)

    report = {
        "files": [str(source)],
        "record_count": len(records),
        "canonical_entity_count": len(set(canonical_map.values())),
        "alignment_frequency_decisions": alignment_report["frequency_decision_count"],
        "alignment_llm_enabled": alignment_report["llm_enabled"],
        "alignment_llm_decisions": alignment_report["llm_decision_count"],
        "error_count": len(errors),
        "errors": errors[:100],
        "allowed_labels": sorted(PCM_NODE_LABELS),
        "allowed_relationships": sorted(PCM_RELATION_TYPES),
    }
    write_json(layout.standardize_report, report)
    if errors:
        ctx.fail_or_warn(f"standardize: {len(errors)} validation error(s); see {layout.standardize_report}")
    log.info("Standardized %d records into %s (%d errors)", len(records), layout.kg_canonical, len(errors))
    return {"records": len(records), "canonical_entities": report["canonical_entity_count"], "errors": len(errors)}
