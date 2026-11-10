"""Graph stage "neo4j": validate the standardized Cypher and optionally import it into Neo4j.

Validation always runs and writes kg_canonical/step7_dry_run_report.json: balanced delimiters, no
destructive keywords, only ontology labels and relationship types, rules backed by evidence cases.

Errors in the Cypher itself (syntax, destructive keywords, labels and relationship types outside the
ontology, constraints) always stop the stage, so such Cypher is never imported. Rule-integrity
findings (a rule without an evidence case, a FAVORS rule whose target is not a WallPosition, ...) are
about what the extraction model wrote, not about the Cypher: like the paper's step 7, the stage
reports them and imports anyway, and stops on them only with --strict. The paper's own graph has 7
such findings, so it could not be rebuilt otherwise.

The import needs an explicit target (--neo4j-database). It refuses a database that already has
nodes unless --allow-non-empty (MERGE then updates the existing graph) or --clear-database (delete
everything first) is given, so the graph behind the paper cannot be changed by accident.
The password is read from NEO4J_PASSWORD only.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from bmrag.build.graph.cypher import extract_labels, extract_relations, split_cypher_statements
from bmrag.build.graph.ontology import CASE_LEVEL_RELATIONS, PCM_NODE_LABELS, PCM_RELATION_TYPES
from bmrag.build.jsonio import read_json, write_json
from bmrag.build.stages import BuildContext, Neo4jTarget

log = logging.getLogger(__name__)


SOURCE_COMMENT_RE = re.compile(r"source_file:\s*(.*?)\s*\|\s*case_id:\s*(.*)$")


def load_companion_records(cypher_file: Path) -> dict[str, dict]:
    json_file = cypher_file.with_suffix(".json")
    if not json_file.exists():
        return {}
    try:
        records = read_json(json_file)
    except Exception:
        return {}
    return {str(rec.get("case_id", "")): rec for rec in records if rec.get("case_id")}


def contexts_from_comments(text: str) -> list[dict]:
    contexts = []
    for line in text.splitlines():
        if line.startswith("// source_file:"):
            match = SOURCE_COMMENT_RE.search(line)
            if match:
                contexts.append({"source_file": match.group(1).strip(), "case_id": match.group(2).strip()})
    return contexts


def cypher_blocks_from_text(text: str, fallback_source: str) -> list[dict]:
    blocks = []
    current_lines: list[str] = []
    current_context = {"source_file": fallback_source, "case_id": None}

    def flush_block() -> None:
        statements = split_cypher_statements("\n".join(current_lines))
        if not statements:
            return
        query_lines = [stmt.rstrip().rstrip(";").strip() for stmt in statements]
        blocks.append(
            {
                **current_context,
                "statement_count": len(statements),
                "query": "\n".join(query_lines),
            }
        )

    for line in text.splitlines():
        if line.startswith("// source_file:"):
            flush_block()
            match = SOURCE_COMMENT_RE.search(line)
            current_context = (
                {"source_file": match.group(1).strip(), "case_id": match.group(2).strip()}
                if match
                else {"source_file": fallback_source, "case_id": None}
            )
            current_lines = []
            continue
        current_lines.append(line)
    flush_block()
    return blocks


def cypher_blocks_from_file(path: Path) -> list[dict]:
    return cypher_blocks_from_text(path.read_text(encoding="utf-8"), path.name)


def strip_single_quoted_strings(stmt: str) -> str:
    return re.sub(r"'(?:\\'|[^'])*'", "''", stmt)


def validate_statement(stmt: str, context: dict, index: int) -> list[dict]:
    errors = []
    sample = stmt[:300]
    base = {"statement_index": index, "source_file": context.get("source_file"), "case_id": context.get("case_id")}
    if "`" in stmt or "```" in stmt:
        errors.append({**base, "error": "Backticks or markdown fences are not allowed", "sample": sample})
    if not stmt.strip():
        errors.append({**base, "error": "Empty statement"})
    syntax_surface = strip_single_quoted_strings(stmt)
    if (
        syntax_surface.count("(") != syntax_surface.count(")")
        or syntax_surface.count("[") != syntax_surface.count("]")
        or syntax_surface.count("{") != syntax_surface.count("}")
    ):
        errors.append({**base, "error": "Unbalanced Cypher delimiters", "sample": sample})
    if not re.search(r"\b(MERGE|MATCH|CREATE|SET|WITH|RETURN)\b", stmt, flags=re.IGNORECASE):
        errors.append({**base, "error": "Statement lacks Cypher action keyword", "sample": sample})
    # Checked without the quoted strings: paper text such as "to remove initial effects" is not Cypher. The original
    # step 7 checked the raw statement and imported anyway, so the false positive went unnoticed.
    if re.search(r"\b(DELETE|DETACH\s+DELETE|DROP\s+DATABASE|REMOVE)\b", syntax_surface, flags=re.IGNORECASE):
        errors.append({**base, "error": "Potentially destructive Cypher is not allowed", "sample": sample})
    for label in sorted(extract_labels(stmt) - PCM_NODE_LABELS):
        errors.append({**base, "error": f"Illegal label: {label}", "sample": sample})
    for rel in sorted(extract_relations(stmt) - PCM_RELATION_TYPES):
        errors.append({**base, "error": f"Illegal relationship type: {rel}", "sample": sample})
    for rel in extract_relations(stmt) & CASE_LEVEL_RELATIONS:
        if "(case)" not in stmt and ":Case" not in stmt and "case_id" not in stmt:
            errors.append(
                {**base, "error": f"Case-level relation {rel} is not attached to a Case context", "sample": sample}
            )
    return errors


def validate_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    contexts = contexts_from_comments(text)
    statements = split_cypher_statements(text)
    errors = []
    if "`" in text or "```" in text:
        errors.append(
            {"source_file": path.name, "case_id": None, "error": "File contains backticks or markdown fences"}
        )
    if not statements:
        errors.append({"source_file": path.name, "case_id": None, "error": "No executable Cypher statements found"})

    context = contexts[0] if contexts else {"source_file": path.name, "case_id": None}
    case_counter = 0
    for i, stmt in enumerate(statements):
        if re.search(r"MERGE\s*\(paper:Paper|MERGE\s*\(case:Case", stmt) and case_counter < len(contexts):
            context = contexts[case_counter]
            case_counter += 1
        errors.extend(validate_statement(stmt, context, i))
    errors.extend(validate_rule_integrity(path, text))

    return {
        "file": str(path),
        "statement_count": len(statements),
        "block_count": len(cypher_blocks_from_text(text, path.name)),
        "labels": sorted(extract_labels(text)),
        "relationships": sorted(extract_relations(text)),
        "errors": errors,
    }


RULE_INTEGRITY_ERRORS = frozenset(
    {
        "KGRule is not supported by any evidence Case",
        "FAVORS rule must connect to a WallPosition",
        "FAVORS rule must connect a PerformanceObjective",
        "Review evidence rule lacks SourceStudy or explicit missing source_study_reference gap",
    }
)


def validate_rule_integrity(cypher_file: Path, text: str) -> list[dict]:
    errors = []
    records = load_companion_records(cypher_file)
    for record_id, record in records.items():
        if record.get("record_type") != "rule":
            continue
        base = {"source_file": record.get("source_file") or cypher_file.name, "case_id": record_id}
        cypher = record.get("cypher") or ""
        rule = record.get("kg_rule") or {}
        if "SUPPORTS_RULE" not in cypher:
            errors.append({**base, "error": "KGRule is not supported by any evidence Case"})
        if (rule.get("relation") or "").lower() == "favors":
            if "FAVORS" not in cypher or ":WallPosition" not in cypher:
                errors.append({**base, "error": "FAVORS rule must connect to a WallPosition"})
            if "HAS_OBJECTIVE" not in cypher or ":PerformanceObjective" not in cypher:
                errors.append({**base, "error": "FAVORS rule must connect a PerformanceObjective"})
        if record.get("paper_type") == "review_paper":
            has_source = "HAS_SOURCE_STUDY" in text or ":SourceStudy" in cypher
            has_gap = "source_study_reference" in cypher or "not reported in the review" in cypher
            if not has_source and not has_gap:
                errors.append(
                    {
                        **base,
                        "error": "Review evidence rule lacks SourceStudy"
                        " or explicit missing source_study_reference gap",
                    }
                )
    return errors


def constraint_statements() -> list[str]:
    constraints = [
        "CREATE CONSTRAINT pcm_paper_id_unique IF NOT EXISTS FOR (n:Paper) REQUIRE n.paper_id IS UNIQUE",
        "CREATE CONSTRAINT pcm_case_id_unique IF NOT EXISTS FOR (n:Case) REQUIRE n.case_id IS UNIQUE",
    ]
    for label in sorted(PCM_NODE_LABELS - {"Paper", "Case"}):
        constraints.append(
            f"CREATE CONSTRAINT pcm_{label.lower()}_name_unique IF NOT EXISTS FOR (n:{label}) REQUIRE n.name IS UNIQUE"
        )
    return constraints


def validate_constraints(constraints: list[str]) -> list[dict]:
    errors = []
    names = [re.search(r"CREATE CONSTRAINT\s+(\w+)", c).group(1) for c in constraints]
    for name, count in Counter(names).items():
        if count > 1:
            errors.append({"error": f"Duplicate constraint name: {name}"})
    for stmt in constraints:
        if "IF NOT EXISTS" not in stmt:
            errors.append({"error": "Constraint is missing IF NOT EXISTS", "sample": stmt})
    return errors


def node_count(driver: Any, database: str) -> int:
    return driver.execute_query("MATCH (n) RETURN count(n) AS n", database_=database).records[0]["n"]


def import_to_neo4j(target: Neo4jTarget, password: str, blocks: list[dict], constraints: list[str]) -> dict[str, int]:
    """Run the constraints and every block, one transaction per block. Returns node counts before and after."""
    from neo4j import GraphDatabase

    with GraphDatabase.driver(target.uri, auth=(target.user, password)) as driver:
        before = node_count(driver, target.database)
        if before and not (target.clear or target.allow_non_empty):
            raise SystemExit(
                f"Neo4j database {target.database!r} already has {before} nodes. Import into an "
                "empty database, or pass --allow-non-empty (merge into it) or --clear-database."
            )
        with driver.session(database=target.database) as session:
            if target.clear:
                log.warning("Deleting all %d nodes in %s before the import", before, target.database)
                session.run("MATCH (n) DETACH DELETE n").consume()
            for stmt in constraints:
                session.run(stmt).consume()
            for index, block in enumerate(blocks):
                try:
                    with session.begin_transaction() as tx:
                        tx.run(block["query"]).consume()
                        tx.commit()
                except Exception as exc:
                    raise RuntimeError(
                        f"Neo4j import failed at block {index} (source_file="
                        f"{block.get('source_file')!r}, case_id={block.get('case_id')!r}): {exc}"
                    ) from exc
        return {"nodes_before": before, "nodes_after": node_count(driver, target.database)}


def run(ctx: BuildContext) -> dict[str, Any]:
    source = ctx.layout.standardized_cypher
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the standardize stage first")
    target = ctx.neo4j
    file_report = validate_file(source)
    constraints = constraint_statements()
    errors = validate_constraints(constraints) + file_report["errors"]
    text = source.read_text(encoding="utf-8")
    statements = split_cypher_statements(text)
    blocks = cypher_blocks_from_text(text, source.name)

    report_path = ctx.layout.neo4j_report
    write_json(
        report_path,
        {
            "dry_run": target is None,
            "clear": bool(target and target.clear),
            "file_count": 1,
            "statement_count": len(statements),
            "block_count": len(blocks),
            "constraint_count": len(constraints),
            "constraints": constraints,
            "error_count": len(errors),
            "errors": errors[:200],
            "files": [file_report],
        },
    )
    blocking = [e for e in errors if e.get("error") not in RULE_INTEGRITY_ERRORS]
    if blocking:
        raise SystemExit(f"Cypher validation failed: {len(blocking)} error(s); see {report_path}")
    if errors:
        ctx.fail_or_warn(
            f"neo4j: {len(errors)} rule-integrity finding(s), imported anyway as the paper's step 7 did "
            f"(--strict stops on them): {sorted(Counter(e['error'] for e in errors).items())}; see {report_path}"
        )
    log.info(
        "Validation passed: %d statements, %d blocks, %d constraints", len(statements), len(blocks), len(constraints)
    )
    stats: dict[str, Any] = {"statements": len(statements), "blocks": len(blocks), "imported_into": None}
    if target is None:
        log.info("No --neo4j-database given: validated only")
        return stats
    password = os.getenv("NEO4J_PASSWORD", "")
    if not password:
        raise SystemExit("NEO4J_PASSWORD is not set in the environment or .env")
    stats.update(import_to_neo4j(target, password, blocks, constraints), imported_into=target.database)
    log.info("Imported into %s: %d -> %d nodes", target.database, stats["nodes_before"], stats["nodes_after"])
    return stats
