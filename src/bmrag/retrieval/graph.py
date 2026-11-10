"""Parameterized Cypher queries over the Neo4j knowledge graph.

Agents never write Cypher; they call run_template with one of these names.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from neo4j import Driver, GraphDatabase

from bmrag.config import load_environment
from bmrag.schemas import RetrievalResult

_driver: Optional[Driver] = None


def _get_driver() -> Driver:
    global _driver
    if _driver is None:
        load_environment()
        _driver = GraphDatabase.driver(os.environ["NEO4J_URI"],
                                       auth=(os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"]))
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


CYPHER_TEMPLATES: dict[str, str] = {
    # A — cases under climates matching a keyword, with their favored positions
    "cases_by_climate": """
        MATCH (c:Case)-[:STUDIED_IN]->(cl:ClimateCondition)
        WHERE toLower(cl.name) CONTAINS toLower($climate_kw)
           OR toLower(coalesce(cl.formal_climate_zone,'')) CONTAINS toLower($climate_kw)
        OPTIONAL MATCH (c)-[:FAVORS]->(fp:WallPosition)
        OPTIONAL MATCH (c)-[:PLACED_AT]->(tp:WallPosition)
        OPTIONAL MATCH (c)-[:ACHIEVES]->(po:PerformanceOutcome)
        OPTIONAL MATCH (c)-[:USES_MATERIAL]->(m:PCMMaterial)
        RETURN c.case_id AS case_id, c.paper_title AS paper,
               cl.name AS climate,
               collect(DISTINCT fp.name) AS favored_positions,
               collect(DISTINCT tp.name) AS tested_positions,
               collect(DISTINCT po.name)[0..3] AS outcomes,
               collect(DISTINCT m.name) AS pcm_materials
        ORDER BY size(favored_positions) DESC
        LIMIT $limit
    """,

    # B — generic rules: under climate / objective, which WallPosition is favored.
    # Aggregate per (rule, favored_position) to avoid duplicate rows from
    # multiple DEPENDS_ON climate/objective bindings.
    "rules_by_climate_objective": """
        MATCH (r:KGRule)-[:FAVORS]->(pos:WallPosition)
        OPTIONAL MATCH (r)-[:DEPENDS_ON]->(cl:ClimateCondition)
        OPTIONAL MATCH (r)-[:HAS_OBJECTIVE]->(obj:PerformanceObjective)
        WITH r, pos,
             collect(DISTINCT cl.name)  AS climates,
             collect(DISTINCT obj.name) AS objectives
        WHERE ($climate_kw IS NULL
               OR any(c IN climates WHERE toLower(coalesce(c,'')) CONTAINS toLower($climate_kw)))
          AND ($objective_kw IS NULL
               OR any(o IN objectives WHERE toLower(coalesce(o,'')) CONTAINS toLower($objective_kw)))
        OPTIONAL MATCH (r)<-[:SUPPORTS_RULE]-(ev:Case)
        WITH r, pos, climates, objectives, count(DISTINCT ev) AS supporting_cases
        RETURN r.rule_id           AS rule_id,
               r.evidence_sentence AS evidence,
               pos.name            AS favored_position,
               climates,
               objectives,
               supporting_cases
        ORDER BY supporting_cases DESC, rule_id
        LIMIT $limit
    """,

    # C — cases whose wall layers overlap a given material list
    "cases_by_wall_materials": """
        MATCH (c:Case)-[:HAS_LAYER]->(:WallAssembly)-[:HAS_LAYER]->(l:WallLayer)
        WITH c, collect(DISTINCT toLower(l.name)) AS layer_names
        WITH c, layer_names,
             [m IN $materials
                WHERE any(ln IN layer_names WHERE ln CONTAINS toLower(m))] AS matched
        WHERE size(matched) >= $min_overlap
        OPTIONAL MATCH (c)-[:FAVORS]->(fp:WallPosition)
        OPTIONAL MATCH (c)-[:STUDIED_IN]->(cl:ClimateCondition)
        RETURN c.case_id AS case_id, c.paper_title AS paper,
               layer_names, matched,
               collect(DISTINCT fp.name) AS favored_positions,
               cl.name AS climate
        ORDER BY size(matched) DESC
        LIMIT $limit
    """,

    # D — cases by PCM + climate (string-level Tm filter; numeric filter
    # left to Reasoner because Property.value is free text)
    "cases_by_pcm_and_climate": """
        MATCH (c:Case)-[:USES_MATERIAL]->(m:PCMMaterial)
        OPTIONAL MATCH (c)-[:HAS_PROPERTY]->(p:Property)
        WHERE p.name IS NOT NULL
          AND (toLower(p.name) CONTAINS 'melting' OR toLower(p.name) CONTAINS 'tm'
               OR toLower(p.name) CONTAINS 'thickness')
        WITH c, m, collect({prop: p.name, value: p.value}) AS pcm_props
        OPTIONAL MATCH (c)-[:STUDIED_IN]->(cl:ClimateCondition)
        WHERE $climate_kw IS NULL
           OR toLower(coalesce(cl.name,'')) CONTAINS toLower($climate_kw)
        OPTIONAL MATCH (c)-[:FAVORS]->(fp:WallPosition)
        RETURN c.case_id AS case_id, m.name AS pcm, pcm_props,
               cl.name AS climate,
               collect(DISTINCT fp.name) AS favored_positions
        LIMIT $limit
    """,

    # E — Judge evidence-verification: given case_ids the Reasoner cited
    # (case_id strings as used by KG-VDB), return their ground-truth favored
    # positions + climate + supporting rules.
    "verify_case_evidence": """
        MATCH (c:Case) WHERE c.case_id IN $case_ids
        OPTIONAL MATCH (c)-[:FAVORS]->(fp:WallPosition)
        OPTIONAL MATCH (c)-[:PLACED_AT]->(tp:WallPosition)
        OPTIONAL MATCH (c)-[:STUDIED_IN]->(cl:ClimateCondition)
        OPTIONAL MATCH (c)-[:SUPPORTS_RULE]->(r:KGRule)
        RETURN c.case_id AS case_id,
               c.paper_title AS paper,
               collect(DISTINCT fp.name) AS favored_positions,
               collect(DISTINCT tp.name) AS tested_positions,
               cl.name AS climate,
               collect(DISTINCT r.rule_id) AS supporting_rules
    """,
}

# Parameter values used when the caller omits them
_TEMPLATE_DEFAULTS: dict[str, dict[str, Any]] = {
    "cases_by_climate":          {"limit": 20},
    "rules_by_climate_objective":{"climate_kw": None, "objective_kw": None, "limit": 10},
    "cases_by_wall_materials":   {"min_overlap": 2, "limit": 20},
    "cases_by_pcm_and_climate":  {"climate_kw": None, "limit": 20},
    "verify_case_evidence":      {},
}


def run_template(name: str, **params: Any) -> list[RetrievalResult]:
    """Run a named template; each row becomes one result."""
    if name not in CYPHER_TEMPLATES:
        raise KeyError(f"Unknown Cypher template '{name}'. Available: {list(CYPHER_TEMPLATES)}")
    final_params = {**_TEMPLATE_DEFAULTS.get(name, {}), **params}
    with _get_driver().session(database=os.environ.get("NEO4J_DATABASE", "neo4j")) as session:
        rows = [dict(record) for record in session.run(CYPHER_TEMPLATES[name], **final_params)]
    return [RetrievalResult(source="neo4j",
                            evidence_id=str(row.get("case_id") or row.get("rule_id") or f"{name}_row_{i}"),
                            score=None,
                            payload=row,
                            diagnostics={"template": name, "params": final_params})
            for i, row in enumerate(rows)]
