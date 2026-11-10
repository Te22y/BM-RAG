"""Rebuild the Neo4j import files from the knowledge-graph stores.

The original step 6 output is not kept. Its records survive as record_original in
case_store.json and rule_store.json, so this script writes
ALL_papers_structured_standardized.{json,cypher} again for the neo4j stage
(python -m bmrag.build neo4j) to import.

Record order: papers in first-appearance order; within a paper, cases then rules
(the original interleaving is not recoverable). Refuses to overwrite existing
output. Uses the standardize stage's own writer so the format matches the neo4j stage's parser.

Usage (from the repo root):
  python tools/rebuild_neo4j_input.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bmrag.build.graph.standardize import save_cypher  # noqa: E402
from bmrag.build.jsonio import write_json  # noqa: E402
from bmrag.build.layout import DataLayout  # noqa: E402
from bmrag.config import get_paths  # noqa: E402


def main() -> None:
    paths = get_paths()
    cases = [v["record_original"] for v in json.loads(paths.case_store.read_text(encoding="utf-8")).values()]
    rules = [v["record_original"] for v in json.loads(paths.rule_store.read_text(encoding="utf-8")).values()]

    records = []
    papers = list(dict.fromkeys(r.get("source_file") for r in cases + rules))
    for paper in papers:
        records += [r for r in cases if r.get("source_file") == paper]
        records += [r for r in rules if r.get("source_file") == paper]
    # Case and rule ids may coincide (R1..R7); they are different node labels.
    if len(records) != len({(r["record_type"], r["case_id"]) for r in records}):
        sys.exit("duplicate records in the stores")

    out = paths.kg_canonical
    target = DataLayout(paths.data_dir).standardized_records
    if target.exists():
        sys.exit(f"refusing to overwrite {target}")
    write_json(target, records)
    save_cypher(records, target.with_suffix(".cypher"))
    print(f"wrote {len(records)} records ({len(cases)} cases, {len(rules)} rules, {len(papers)} papers) to {out}")


if __name__ == "__main__":
    main()
