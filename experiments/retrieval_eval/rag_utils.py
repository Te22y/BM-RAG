"""
rag_utils.py
============

A minimal RAG toolkit for the evaluation scripts, working on any multi-representation cache.


Provides:
  - build_retriever()          — load the summary-embedding MultiVectorRetriever
  - retrieve_stage1()          — stage 1 vector search, returns (raw_docs, doc_ids)
  - rerank_stage2()            — stage 2 LLM rerank, returns (reranked_docs, rerank_scores)
  - load_summary()             — read summaries/{doc_id}.txt
  - format_context_text()      — join raw documents into a plain-text context
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# langchain_classic is the version installed for this project
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever


# ── Constants, same as the main paper index ─────────────────────────────────
ID_KEY         = "doc_id"
EMBED_MODEL    = "text-embedding-3-small"
RETRIEVE_TOP_K = 20
RERANK_TOP_K   = 3
LLM_RERANK_MODEL = "gpt-4o"

CHROMA_DIR   = "chroma_db"
DOCSTORE_DIR = "docstore"


def _load_env() -> None:
    """Load the repository's .env so OPENAI_API_KEY is available."""
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")


# ══════════════════════════════════════════════════════════════════════════════
# Retriever loading
# ══════════════════════════════════════════════════════════════════════════════
def build_retriever(
    cache_root: Path,
    collection_name: str = "multi_modal_rag",
    top_k: int = RETRIEVE_TOP_K,
) -> MultiVectorRetriever:
    """
    Load a pre-built (already indexed) MultiVectorRetriever from disk.
    Does NOT rebuild summaries or embeddings — assumes chroma_db/ and docstore/
    already exist under cache_root.
    """
    chroma_path = cache_root / CHROMA_DIR
    docstore_path = cache_root / DOCSTORE_DIR

    if not chroma_path.exists():
        raise FileNotFoundError(
            f"chroma_db not found at {chroma_path}. "
            "Run the offline build step first: python -m bmrag.build paper-index --data <folder>"
        )

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=str(chroma_path),
    )
    docstore = create_kv_docstore(LocalFileStore(str(docstore_path)))

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key=ID_KEY,
        search_kwargs={"k": top_k},
    )

    n_docs = len(vectorstore.get(include=[])["ids"])
    print(f"[retriever] loaded '{collection_name}' from {cache_root.name} — {n_docs} docs in vectorstore")
    return retriever


# ══════════════════════════════════════════════════════════════════════════════
# Two-stage retrieval (separated so caller can capture intermediate results)
# ══════════════════════════════════════════════════════════════════════════════
def retrieve_stage1(
    query: str,
    retriever: MultiVectorRetriever,
) -> tuple[list[Document], list[str]]:
    """
    Stage 1: vector similarity search.

    Returns
    -------
    raw_docs : list[Document]
        Raw documents fetched from docstore, ordered by vector similarity score.
    doc_ids  : list[str]
        Corresponding doc_id values (metadata["doc_id"]) in the same order.
    """
    raw_docs: list[Document] = retriever.invoke(query)
    doc_ids = [doc.metadata.get(ID_KEY, "") for doc in raw_docs]
    return raw_docs, doc_ids


def load_summary(cache_root: Path, doc_id: str) -> Optional[str]:
    p = cache_root / "summaries" / f"{doc_id}.txt"
    return p.read_text(encoding="utf-8") if p.exists() else None


# LLM rerank prompt (same text as the main paper index)
_RERANK_PROMPT = """\
You are an expert relevance scoring assistant for reranking retrieved documents in a scientific literature RAG pipeline.

Your task is to evaluate how useful each document is for answering the user query.

IMPORTANT:
- Focus on the ACTUAL INFORMATION CONTENT, not superficial keyword overlap.
- Prefer documents with concrete data, methods, experimental results, or conclusions.
- Penalize vague, generic, or background-only summaries.

Scoring criteria (continuous scale 0.0–1.0):

0.0–0.2:
Completely irrelevant or unrelated to the query.

0.2–0.4:
Weak relevance. Mentions similar terms but does not provide useful information.

0.4–0.6:
Moderate relevance. Contains partial information but lacks key details.

0.6–0.8:
Strong relevance. Provides useful insights (e.g., method, mechanism, or findings).

0.8–1.0:
Highly relevant. Directly answers the query with specific, detailed, and actionable information (e.g., quantitative results, optimized parameters, experimental validation).

Evaluation dimensions (implicitly consider all):
- Topic match (is it really about the same problem?)
- Method/approach relevance
- Presence of quantitative data or experimental results
- Specificity (general review vs concrete findings)
- Novelty/usefulness for answering the query

Additional rules:
- Scores must reflect RELATIVE differences between documents.
- Avoid giving identical scores unless documents are truly similar.
- Prefer documents that could directly support answering the query in a final response.

Output format:
- Output ONLY a JSON array of objects
- No explanation, no markdown, no extra text
- Each object must contain:
    - "index": integer
    - "score": float (0.0–1.0)

The number of output items MUST equal the number of documents.

Example:
[{{"index": 0, "score": 0.91}}, {{"index": 1, "score": 0.47}}]

"""


def _get_scoring_text(
    doc: Document,
    cache_root: Path,
    scoring_mode: str,
) -> Optional[str]:
    etype = doc.metadata.get("element_type", "text")
    doc_id = doc.metadata.get(ID_KEY, "")
    summary = load_summary(cache_root, doc_id)

    if etype in {"image", "table"}:
        return summary

    if scoring_mode == "raw_text":
        return doc.page_content[:3000]

    return summary or doc.page_content[:3000]


def prepare_context_documents(
    docs: list[Document],
    cache_root: Path,
    text_use_summary: bool = False,
) -> list[Document]:
    """
    Convert retrieved/reranked docs into the content used for generation context.

    Image/table always use the cached summary.
    Text chunks use the raw chunk by default; pass text_use_summary=True to use
    the cached summary instead (Variant E behaviour).
    """
    context_docs: list[Document] = []

    for doc in docs:
        etype = doc.metadata.get("element_type", "text")
        metadata = dict(doc.metadata)

        if etype in {"image", "table"}:
            summary = load_summary(cache_root, doc.metadata.get(ID_KEY, ""))
            metadata["context_content"] = "summary" if summary else "raw_fallback"
            context_docs.append(Document(
                page_content=summary or doc.page_content,
                metadata=metadata,
            ))
            continue

        if text_use_summary:
            summary = load_summary(cache_root, doc.metadata.get(ID_KEY, ""))
            if summary:
                metadata["context_content"] = "summary"
                context_docs.append(Document(page_content=summary, metadata=metadata))
                continue
            # fall through to raw if summary not found
            metadata["context_content"] = "raw_fallback"
        else:
            metadata["context_content"] = "raw"

        context_docs.append(Document(
            page_content=doc.page_content,
            metadata=metadata,
        ))

    return context_docs


def rerank_stage2(
    query: str,
    raw_docs: list[Document],
    cache_root: Path,
    top_k: int = RERANK_TOP_K,
    model: str = LLM_RERANK_MODEL,
    batch_size: int = 10,
    scoring_mode: str = "summary",
    text_use_summary: bool = False,
) -> tuple[list[Document], list[float]]:
    """
    Stage 2: LLM rerank.

    scoring_mode="summary": score with summaries when available.
    scoring_mode="raw_text": score text chunks with raw content; image/table still use summaries.

    Returns
    -------
    reranked_docs   : top_k context docs, sorted by descending rerank score
    rerank_scores   : corresponding scores (same order as reranked_docs)
    """
    llm = ChatOpenAI(model=model, temperature=0, max_retries=2)

    scoring_texts: list[Optional[str]] = []
    for doc in raw_docs:
        scoring_texts.append(_get_scoring_text(doc, cache_root, scoring_mode))

    participate = [i for i, t in enumerate(scoring_texts) if t is not None]
    skip        = [i for i, t in enumerate(scoring_texts) if t is None]

    all_scores: dict[int, float] = {}

    for batch_start in range(0, len(participate), batch_size):
        batch_orig = participate[batch_start: batch_start + batch_size]
        doc_lines = [
            f"[{local_i}] {scoring_texts[orig_i]}"
            for local_i, orig_i in enumerate(batch_orig)
        ]
        docs_str = "\n\n".join(doc_lines)
        prompt = _RERANK_PROMPT + "\n\nQuery:\n" + query + "\n\nDocuments:\n" + docs_str

        last_err = None
        for attempt in range(3):
            try:
                resp = llm.invoke(prompt)
                raw = resp.content.strip()
                if raw.startswith("```"):
                    raw = re.sub(r"^```(?:json)?\s*", "", raw)
                    raw = re.sub(r"\s*```$", "", raw)
                items: list[dict] = json.loads(raw)
                for item in items:
                    local_i = int(item["index"])
                    score   = float(item["score"])
                    orig_i  = batch_orig[local_i]
                    all_scores[orig_i] = max(0.0, min(1.0, score))
                break
            except Exception as e:
                last_err = e
                time.sleep(2 ** attempt)
        else:
            print(f"  [warn] rerank batch failed: {last_err}")
            for orig_i in batch_orig:
                all_scores.setdefault(orig_i, 0.5)

    for orig_i in skip:
        all_scores[orig_i] = 0.0

    indexed = sorted(range(len(raw_docs)), key=lambda i: all_scores.get(i, 0.0), reverse=True)
    selected_docs = [raw_docs[i] for i in indexed[:top_k]]
    reranked_docs = prepare_context_documents(selected_docs, cache_root, text_use_summary=text_use_summary)
    rerank_scores = [all_scores.get(i, 0.0) for i in indexed[:top_k]]
    return reranked_docs, rerank_scores


# ══════════════════════════════════════════════════════════════════════════════
# Context formatting for generation & judging
# ══════════════════════════════════════════════════════════════════════════════
def format_context_text(docs: list[Document]) -> str:
    """
    Format retrieved docs into a plain-text context string for the LLM generator.
    Retrieved image/table docs should already contain summaries in page_content.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        etype  = doc.metadata.get("element_type", "text")
        source = doc.metadata.get("source_file", "unknown")
        if etype == "image" and doc.metadata.get("context_content") != "summary":
            parts.append(f"[Doc {i} | IMAGE from: {source}]\n[Image content not shown in text eval]")
        else:
            parts.append(f"[Doc {i} | {etype.upper()} from: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
# Answer generation
# ══════════════════════════════════════════════════════════════════════════════
RAG_SYSTEM_PROMPT = """\
You are an expert assistant in building environment and materials science research
(including thermal insulation, phase change materials, UHPC, and concrete technology).

Answer the question using ONLY the provided context documents. If the context does not
contain enough information to answer confidently, say so explicitly — do NOT fabricate
or infer beyond the context.

Be concise: prefer direct answers with exact values and units over long explanations.
"""


def generate_answer(
    query: str,
    context_docs: list[Document],
    model: str = "gpt-4o",
    max_retries: int = 2,
) -> str:
    """Call GPT-4o (or specified model) with the retrieved context to generate an answer."""
    context_text = format_context_text(context_docs)
    llm = ChatOpenAI(model=model, temperature=0, max_retries=max_retries)
    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user",   "content": f"Context:\n{context_text}\n\nQuestion: {query}"},
    ]
    resp = llm.invoke(messages)
    return (resp.content or "").strip()
