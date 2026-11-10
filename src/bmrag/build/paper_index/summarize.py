"""Paper index stage "summarize": one LLM summary per element, then embeddings of the summaries.

This is a multi-representation index: queries search the summary embeddings, and the docstore
returns the raw element stored under the same doc_id. Summaries are cached one file per element
(paper_index/summaries/<doc_id>.txt), so an interrupted run resumes; --force writes them again.
Only elements missing from the vector store are embedded.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

from bmrag import llm
from bmrag.build.paper_index.separate import ElementRecord, Manifest
from bmrag.build.profiles import SummarySettings
from bmrag.build.stages import BuildContext
from bmrag.config import EMBEDDING_MODEL, PAPER_COLLECTION, PAPER_ID_KEY

log = logging.getLogger(__name__)
STAGE = "summarize"


def _chat_model(settings: SummarySettings, max_retries: int | None) -> Any:
    from langchain_openai import ChatOpenAI

    options = {} if max_retries is None else {"max_retries": max_retries}
    return ChatOpenAI(model=settings.model, **options)


def _text_table_chain(settings: SummarySettings) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from bmrag.prompts import load_prompt

    prompt = ChatPromptTemplate.from_template(load_prompt(settings.text_prompt))
    return {"element": lambda x: x} | prompt | _chat_model(settings, settings.text_max_retries) | StrOutputParser()


def _image_chain(settings: SummarySettings) -> Any:
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    from bmrag.prompts import load_prompt

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {"type": "text", "text": load_prompt(settings.image_prompt)},
                    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{image}"}},
                ],
            )
        ]
    )
    return prompt | _chat_model(settings, settings.image_max_retries) | StrOutputParser()


def summary_path(root: Path, doc_id: str) -> Path:
    return root / "summaries" / f"{doc_id}.txt"


def _save_summary(root: Path, doc_id: str, summary: str) -> None:
    path = summary_path(root, doc_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(summary, encoding="utf-8")


def _load_element(root: Path, record: ElementRecord) -> Any:
    with (root / record.pickle_path).open("rb") as f:
        return pickle.load(f)


def generate_summaries(
    root: Path, settings: SummarySettings, force: bool, callbacks: list[Any]
) -> tuple[Manifest, int]:
    """Summarize every kept element without a cached summary; the manifest is saved after each batch.

    Text and tables go in batches of settings.batch_size, images one at a time.
    """
    manifest_path = root / "manifest.json"
    manifest = Manifest.load(manifest_path)
    pending: dict[str, list[ElementRecord]] = {"text": [], "table": [], "image": []}
    for record in manifest.records:
        if record.filtered:
            continue
        if not force and summary_path(root, record.doc_id).exists():
            record.has_summary = True
        elif record.element_type in pending:
            pending[record.element_type].append(record)
    manifest.save(manifest_path)
    log.info("Pending summaries: text %d, table %d, image %d", *(len(pending[k]) for k in ("text", "table", "image")))

    text_chain = _text_table_chain(settings)
    for element_type in ("text", "table"):
        records = pending[element_type]
        for start in range(0, len(records), settings.batch_size):
            batch = records[start : start + settings.batch_size]
            elements = [str(_load_element(root, record)) for record in batch]
            summaries = text_chain.batch(
                elements, config={"max_concurrency": settings.concurrency, "callbacks": callbacks}
            )
            for record, summary in zip(batch, summaries, strict=True):
                _save_summary(root, record.doc_id, summary)
                record.has_summary = True
            manifest.save(manifest_path)
            log.info("  [%s] %d/%d done", element_type, min(start + settings.batch_size, len(records)), len(records))

    if pending["image"]:
        image_chain = _image_chain(settings)
    for done, record in enumerate(pending["image"], 1):
        summary = image_chain.invoke(
            {"image": _load_element(root, record)["image_base64"]}, config={"callbacks": callbacks}
        )
        _save_summary(root, record.doc_id, summary)
        record.has_summary = True
        if done % 10 == 0 or done == len(pending["image"]):
            manifest.save(manifest_path)
            log.info("  [image] %d/%d done", done, len(pending["image"]))
    return manifest, sum(len(v) for v in pending.values())


def embed_summaries(root: Path, manifest: Manifest, settings: SummarySettings, ctx: BuildContext) -> int:
    """Add elements missing from the vector store and the docstore. Returns how many were added."""
    from langchain_chroma import Chroma
    from langchain_classic.storage import LocalFileStore, create_kv_docstore
    from langchain_core.documents import Document

    store = Chroma(
        collection_name=PAPER_COLLECTION,
        embedding_function=llm.create_embeddings(),
        persist_directory=str(ctx.layout.paper_chroma),
    )
    docstore = create_kv_docstore(LocalFileStore(str(ctx.layout.paper_docstore)))
    existing = {
        meta[PAPER_ID_KEY] for meta in store.get(include=["metadatas"])["metadatas"] if meta and PAPER_ID_KEY in meta
    }

    summaries: list[Document] = []
    raw_elements: list[tuple[str, Document]] = []
    for record in manifest.records:
        path = summary_path(root, record.doc_id)
        if record.filtered or record.doc_id in existing or not path.exists():
            continue
        metadata = {PAPER_ID_KEY: record.doc_id, "element_type": record.element_type, "source_file": record.source_file}
        summaries.append(Document(page_content=path.read_text(encoding="utf-8"), metadata=metadata))
        element = _load_element(root, record)
        content = element.get("image_base64", "") if record.element_type == "image" else str(element)
        raw_elements.append((record.doc_id, Document(page_content=content, metadata=metadata)))

    for start in range(0, len(summaries), settings.embedding_batch_size):
        batch = summaries[start : start + settings.embedding_batch_size]
        store.add_documents(batch)
        ctx.record.usage.add_embeddings(STAGE, EMBEDDING_MODEL, [d.page_content for d in batch])
        log.info("  Embedded %d/%d", min(start + settings.embedding_batch_size, len(summaries)), len(summaries))
    if raw_elements:
        docstore.mset(raw_elements)
    return len(summaries)


def run(ctx: BuildContext) -> dict[str, Any]:
    from langchain_core.callbacks import UsageMetadataCallbackHandler

    settings = ctx.profile.summaries
    root = ctx.layout.paper_index
    if not ctx.layout.paper_manifest.exists():
        raise SystemExit(f"No manifest in {root}; run the separate stage first")
    usage = UsageMetadataCallbackHandler()
    manifest, summarized = generate_summaries(root, settings, ctx.force, [usage])
    ctx.record.usage.add_langchain(STAGE, usage.usage_metadata)
    for model in usage.usage_metadata:
        ctx.record.usage.add_langchain_calls(STAGE, model, summarized)
    added = embed_summaries(root, manifest, settings, ctx)
    log.info(
        "Index ready: %d elements in the manifest, %d summarized, %d newly embedded",
        len(manifest.records),
        summarized,
        added,
    )
    return {"elements": len(manifest.records), "summarized": summarized, "embedded": added}
