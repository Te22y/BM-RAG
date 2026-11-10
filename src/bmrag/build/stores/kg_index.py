"""Graph stage "kg-index": the case and rule stores and their Chroma collections.

Reads kg_canonical/ALL_papers_structured_standardized.json and writes kg_index/case_store.json,
kg_index/rule_store.json and the collections pcm_kg_cases_v1 and pcm_kg_rules_v1 in
kg_index/chroma_db. Only bundles missing from a collection are embedded; --force rebuilds the
collections from scratch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from bmrag import llm
from bmrag.build.jsonio import write_json
from bmrag.build.llm_client import UsageLog
from bmrag.build.stages import BuildContext
from bmrag.build.stores.records import build_case_store, build_rule_store, load_records, safe_reset_dir
from bmrag.config import CASE_COLLECTION, EMBEDDING_MODEL, RULE_COLLECTION

log = logging.getLogger(__name__)
STAGE = "kg-index"


def _add_to_collection(
    store: dict[str, Any], chroma_dir: Path, collection: str, embeddings: Any, usage: UsageLog | None = None
) -> int:
    """Embed the bundles not yet in the collection. Returns the collection size."""
    from langchain_chroma import Chroma
    from langchain_core.documents import Document

    vectorstore = Chroma(collection_name=collection, embedding_function=embeddings, persist_directory=str(chroma_dir))
    existing = set(vectorstore.get(include=[]).get("ids") or [])
    ids = [item_id for item_id in sorted(store) if item_id not in existing]
    if ids:
        documents = [Document(page_content=store[i]["retrieval_text"], metadata=store[i]["metadata"]) for i in ids]
        vectorstore.add_documents(documents, ids=ids)
        if usage is not None:
            usage.add_embeddings(STAGE, EMBEDDING_MODEL, [d.page_content for d in documents])
    return len(vectorstore.get(include=[]).get("ids") or [])


def run(ctx: BuildContext) -> dict[str, Any]:
    layout = ctx.layout
    source = layout.standardized_records
    if not source.exists():
        raise SystemExit(f"No {source.name} in {source.parent}; run the standardize stage first")
    if ctx.force:
        safe_reset_dir(layout.kg_chroma, layout.kg_index)
    records = load_records(source)
    case_store = build_case_store(records)
    rule_store = build_rule_store(records)
    write_json(layout.case_store, case_store)
    write_json(layout.rule_store, rule_store)

    embeddings = llm.create_embeddings()
    report = {
        "cases": len(case_store),
        "rules": len(rule_store),
        "case_vectors": _add_to_collection(case_store, layout.kg_chroma, CASE_COLLECTION, embeddings, ctx.record.usage),
        "rule_vectors": _add_to_collection(rule_store, layout.kg_chroma, RULE_COLLECTION, embeddings, ctx.record.usage),
    }
    log.info("KG index: %(cases)d cases / %(case_vectors)d vectors, %(rules)d rules / %(rule_vectors)d vectors", report)
    return report
