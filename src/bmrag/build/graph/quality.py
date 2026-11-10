"""Graph stage "quality": check the entity records for missing fields and unknown labels or relations.

Reads kg_entities/ALL_papers_structured.json and writes kg_entities/step2_quality_report.json with
per-record issues at three levels (error, warning, info). Errors fail the stage with --strict;
otherwise they are recorded as a warning in the run record.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Any

from bmrag.build.graph.ontology import canonical_name, normalize_label, normalize_relation
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.stages import BuildContext

ISSUE_LEVELS = ("error", "warning", "info")
CRITICAL_RELATIONS = {
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
    "DEPENDS_ON",
    "CONFLICTS_WITH",
    "NOT_SUPPORTED_AS_UNIVERSAL",
}


def as_records(data: Any) -> list[dict]:
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        for key in ("records", "cases"):
            if isinstance(data.get(key), list):
                return [item for item in data[key] if isinstance(item, dict)]
    return []


def issue(level: str, code: str, message: str, **context: Any) -> dict:
    return {
        "level": level,
        "code": code,
        "message": message,
        "context": {k: v for k, v in context.items() if v not in (None, "", [])},
    }


def entity_name(entity: dict) -> str:
    return canonical_name(entity.get("normalized_name") or entity.get("name") or entity.get("id"))


def expected_identity_names(record: dict) -> set[str]:
    names = {
        canonical_name(record.get("paper_title")),
        canonical_name(record.get("case_id")),
        canonical_name(record.get("rule_id")),
    }
    return {name for name in names if name != "Unknown"}


def check_entities(record: dict) -> tuple[set[str], list[dict], Counter]:
    issues: list[dict] = []
    names: set[str] = set()
    label_counts: Counter = Counter()
    seen_entity_ids: set[str] = set()

    for index, ent in enumerate(record.get("entities") or []):
        if not isinstance(ent, dict):
            issues.append(issue("error", "entity_not_object", "Entity entry is not an object.", entity_index=index))
            continue
        try:
            label = normalize_label(ent.get("type"))
            label_counts[label] += 1
        except ValueError as exc:
            issues.append(issue("error", "illegal_entity_label", str(exc), entity_index=index, entity=ent))
            continue

        name = entity_name(ent)
        names.add(name)
        if name == "Unknown":
            issues.append(
                issue("warning", "unknown_entity_name", "Entity resolves to Unknown.", entity_index=index, entity=ent)
            )

        entity_id = ent.get("id")
        if entity_id:
            if entity_id in seen_entity_ids:
                issues.append(
                    issue("warning", "duplicate_entity_id", "Duplicate entity id within record.", entity_id=entity_id)
                )
            seen_entity_ids.add(entity_id)
        else:
            issues.append(
                issue("warning", "missing_entity_id", "Entity has no stable id.", entity_index=index, name=name)
            )

    return names, issues, label_counts


def check_relations(record: dict, entity_names: set[str]) -> tuple[list[dict], Counter]:
    issues: list[dict] = []
    relation_counts: Counter = Counter()
    known_names = set(entity_names) | expected_identity_names(record)

    for index, rel in enumerate(record.get("relations") or []):
        if not isinstance(rel, dict):
            issues.append(
                issue("error", "relation_not_object", "Relation entry is not an object.", relation_index=index)
            )
            continue

        try:
            predicate = normalize_relation(rel.get("predicate"))
            relation_counts[predicate] += 1
        except ValueError as exc:
            issues.append(issue("error", "illegal_relation_type", str(exc), relation_index=index, relation=rel))
            continue

        source = canonical_name(rel.get("source"))
        target = canonical_name(rel.get("target"))
        if source == "Unknown" or target == "Unknown":
            issues.append(
                issue(
                    "error",
                    "unknown_relation_endpoint",
                    "Relation source or target resolves to Unknown.",
                    relation_index=index,
                    predicate=predicate,
                    source=source,
                    target=target,
                )
            )
        if source not in known_names:
            issues.append(
                issue(
                    "warning",
                    "missing_source_entity",
                    "Relation source does not appear in this record's entities.",
                    relation_index=index,
                    predicate=predicate,
                    source=source,
                )
            )
        if target not in known_names:
            issues.append(
                issue(
                    "warning",
                    "missing_target_entity",
                    "Relation target does not appear in this record's entities.",
                    relation_index=index,
                    predicate=predicate,
                    target=target,
                )
            )

        if not rel.get("confidence"):
            issues.append(
                issue(
                    "warning",
                    "missing_relation_confidence",
                    "Relation has no confidence value.",
                    relation_index=index,
                    predicate=predicate,
                )
            )
        if predicate in CRITICAL_RELATIONS and not rel.get("evidence"):
            issues.append(
                issue(
                    "info",
                    "critical_relation_without_direct_evidence",
                    "Critical semantic relation has no relation-level evidence;"
                    " verify against evidence_snippets/raw_case.",
                    relation_index=index,
                    predicate=predicate,
                    source=source,
                    target=target,
                )
            )

    return issues, relation_counts


def check_mechanism_chain(record: dict, entity_names: set[str]) -> tuple[list[dict], Counter]:
    issues: list[dict] = []
    relation_counts: Counter = Counter()

    for index, step in enumerate(record.get("mechanism_chain") or []):
        if not isinstance(step, dict):
            issues.append(
                issue("error", "mechanism_step_not_object", "Mechanism chain step is not an object.", step_index=index)
            )
            continue
        for label_key, default_label in (("cause_type", "PhysicalMechanism"), ("effect_type", "PerformanceOutcome")):
            try:
                normalize_label(step.get(label_key) or default_label)
            except ValueError as exc:
                issues.append(issue("error", "illegal_mechanism_label", str(exc), step_index=index, field=label_key))
        try:
            predicate = normalize_relation(step.get("relation") or "LEADS_TO")
            relation_counts[predicate] += 1
        except ValueError as exc:
            issues.append(issue("error", "illegal_mechanism_relation", str(exc), step_index=index))
            continue

        cause = canonical_name(step.get("cause"))
        effect = canonical_name(step.get("effect"))
        if cause == "Unknown" or effect == "Unknown":
            issues.append(
                issue(
                    "error",
                    "unknown_mechanism_endpoint",
                    "Mechanism chain cause or effect resolves to Unknown.",
                    step_index=index,
                    relation=predicate,
                    cause=cause,
                    effect=effect,
                )
            )
        if cause not in entity_names:
            issues.append(
                issue(
                    "info",
                    "mechanism_cause_not_in_entities",
                    "Mechanism cause is not listed in entities;"
                    " Step5 will create it, but Step2 entities are incomplete.",
                    step_index=index,
                    cause=cause,
                )
            )
        if effect not in entity_names:
            issues.append(
                issue(
                    "info",
                    "mechanism_effect_not_in_entities",
                    "Mechanism effect is not listed in entities;"
                    " Step5 will create it, but Step2 entities are incomplete.",
                    step_index=index,
                    effect=effect,
                )
            )
        if not step.get("confidence"):
            issues.append(
                issue(
                    "warning",
                    "missing_mechanism_confidence",
                    "Mechanism step has no confidence value.",
                    step_index=index,
                )
            )

    return issues, relation_counts


def check_record(record: dict, source_path: Path, index: int) -> dict:
    record_id = record.get("case_id") or record.get("rule_id") or f"record_{index + 1}"
    issues: list[dict] = []
    entity_names, entity_issues, label_counts = check_entities(record)
    relation_issues, relation_counts = check_relations(record, entity_names)
    mechanism_issues, mechanism_relation_counts = check_mechanism_chain(record, entity_names)
    issues.extend(entity_issues)
    issues.extend(relation_issues)
    issues.extend(mechanism_issues)

    if not record.get("entities"):
        issues.append(issue("error", "no_entities", "Record has no entities."))
    if not record.get("relations"):
        issues.append(issue("warning", "no_relations", "Record has no relations."))
    if record.get("record_type") == "case" and not record.get("evidence_snippets"):
        issues.append(issue("warning", "no_evidence_snippets", "Case record has no evidence snippets."))
    if record.get("record_type") == "rule" and not record.get("kg_rule"):
        issues.append(issue("warning", "missing_kg_rule_payload", "Rule record has no kg_rule payload."))

    return {
        "source_json": str(source_path),
        "record_index": index,
        "record_type": record.get("record_type") or "case",
        "record_id": record_id,
        "paper_title": record.get("paper_title"),
        "entity_count": len(record.get("entities") or []),
        "relation_count": len(record.get("relations") or []),
        "mechanism_step_count": len(record.get("mechanism_chain") or []),
        "label_counts": dict(sorted(label_counts.items())),
        "relation_counts": dict(sorted((relation_counts + mechanism_relation_counts).items())),
        "issue_counts": count_issue_levels(issues),
        "issues": issues,
    }


def count_issue_levels(issues: list[dict]) -> dict[str, int]:
    counts = Counter(item.get("level", "info") for item in issues)
    return {level: counts.get(level, 0) for level in ISSUE_LEVELS}


def count_issue_codes(issues: list[dict]) -> dict[str, int]:
    return dict(sorted(Counter(item.get("code", "unknown_issue") for item in issues).items()))


def check_file(path: Path) -> dict:
    records = as_records(read_json(path))
    record_reports = [check_record(record, path, index) for index, record in enumerate(records)]
    all_issues = [issue_item for report in record_reports for issue_item in report["issues"]]
    label_counts = Counter()
    relation_counts = Counter()
    for report in record_reports:
        label_counts.update(report["label_counts"])
        relation_counts.update(report["relation_counts"])
    return {
        "source_json": str(path),
        "record_count": len(records),
        "issue_counts": count_issue_levels(all_issues),
        "issue_code_counts": count_issue_codes(all_issues),
        "label_counts": dict(sorted(label_counts.items())),
        "relation_counts": dict(sorted(relation_counts.items())),
        "records": record_reports,
    }


def run(ctx: BuildContext) -> dict[str, Any]:
    source = ctx.layout.entity_records
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the entities stage first")
    file_report = check_file(source)
    report = {
        "input": str(source),
        "file_count": 1,
        "record_count": file_report["record_count"],
        "issue_counts": file_report["issue_counts"],
        "issue_code_counts": file_report["issue_code_counts"],
        "files": [file_report],
    }
    write_json(ctx.layout.quality_report, report)
    counts = report["issue_counts"]
    log_line = (
        f"quality: {report['record_count']} record(s), {counts['error']} error(s), "
        f"{counts['warning']} warning(s), {counts['info']} info item(s)"
    )
    if counts["error"]:
        ctx.fail_or_warn(f"{log_line}; see {ctx.layout.quality_report}")
    else:
        logging.getLogger(__name__).info(log_line)
    return {"records": report["record_count"], **counts}
