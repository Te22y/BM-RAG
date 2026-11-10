"""Two-stage search over paper summaries: embedding recall, then LLM rerank.

Read-only: it searches the index built by the summarize stage of bmrag.build and never
adds to it. Scoring and results use the summaries; raw chunks are not surfaced.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_core.documents import Document

from bmrag import llm
from bmrag.config import CHROMA_DIRNAME, DOCSTORE_DIRNAME, PAPER_COLLECTION, PAPER_ID_KEY, get_paths
from bmrag.prompts import load_prompt
from bmrag.schemas import RetrievalResult

logger = logging.getLogger(__name__)

CANDIDATE_COUNT = 20
RERANK_BATCH_SIZE = 10
NEUTRAL_SCORE = 0.5        # given to documents in a batch whose scoring failed
DEFAULT_RERANK_MODEL = "gpt-5.2"

_retriever: Optional[MultiVectorRetriever] = None


def _get_retriever() -> MultiVectorRetriever:
    global _retriever
    if _retriever is None:
        root = get_paths().paper_index
        store = Chroma(collection_name=PAPER_COLLECTION, embedding_function=llm.create_embeddings(),
                       persist_directory=str(root / CHROMA_DIRNAME))
        _retriever = MultiVectorRetriever(vectorstore=store,
                                          docstore=create_kv_docstore(LocalFileStore(str(root / DOCSTORE_DIRNAME))),
                                          id_key=PAPER_ID_KEY,
                                          search_kwargs={"k": CANDIDATE_COUNT})
    return _retriever


def _load_summary(root: Path, doc_id: str) -> Optional[str]:
    path = root / "summaries" / f"{doc_id}.txt"
    return path.read_text(encoding="utf-8") if path.exists() else None


@dataclass
class _Ranked:
    document: Document
    score: float
    original_rank: int


def _parse_scores(reply: str) -> list[dict[str, Any]]:
    if reply.startswith("```"):
        reply = reply.split("```")[1]
        if reply.startswith("json"):
            reply = reply[4:]
    return json.loads(reply)


def _rerank(query: str, documents: list[Document], summaries: list[Optional[str]], top_k: int,
            model: str) -> list[_Ranked]:
    """Score documents in batches. An image without a summary has nothing to score and ranks last."""
    chat_model = llm.create_chat_model(model, temperature=0, max_retries=2)
    texts: list[Optional[str]] = []
    for i, doc in enumerate(documents):
        if summaries[i]:
            texts.append(summaries[i])
        elif doc.metadata.get("element_type", "text") == "image":
            texts.append(None)
        else:
            texts.append(doc.page_content)

    scored = [i for i, text in enumerate(texts) if text is not None]
    scores: dict[int, float] = {}
    for start in range(0, len(scored), RERANK_BATCH_SIZE):
        batch = scored[start:start + RERANK_BATCH_SIZE]
        listing = "\n\n".join(f"[{local}] {texts[i]}" for local, i in enumerate(batch))
        prompt = load_prompt("paper_rerank") + "\n\nQuery:\n" + query + "\n\nDocuments:\n" + listing
        try:
            for item in _parse_scores(llm.complete(chat_model, prompt, caller="paper_rerank")):
                scores[batch[int(item["index"])]] = max(0.0, min(1.0, float(item["score"])))
        except Exception as exc:
            logger.warning("Paper rerank batch %d failed: %s", start // RERANK_BATCH_SIZE + 1, exc)
            for i in batch:
                scores.setdefault(i, NEUTRAL_SCORE)

    ranked = [_Ranked(documents[i], scores.get(i, NEUTRAL_SCORE), i) for i in scored]
    ranked += [_Ranked(documents[i], 0.0, i) for i, text in enumerate(texts) if text is None]
    ranked.sort(key=lambda r: r.score, reverse=True)
    return ranked[:top_k]


def retrieve_paper_chunks(query: str, top_k: int = 3, rerank_model: str = DEFAULT_RERANK_MODEL) -> list[RetrievalResult]:
    root = get_paths().paper_index
    candidates = _get_retriever().invoke(query)
    summaries = [_load_summary(root, doc.metadata.get(PAPER_ID_KEY, "")) for doc in candidates]

    results = []
    for ranked in _rerank(query, candidates, summaries, top_k, rerank_model):
        summary = summaries[ranked.original_rank]
        if summary is None:
            doc_id = ranked.document.metadata.get(PAPER_ID_KEY, "<unknown>")
            raise RuntimeError(f"No cached summary for doc_id={doc_id}; run the paper index build first.")
        metadata = {**ranked.document.metadata, "rerank_score": ranked.score, "original_rank": ranked.original_rank,
                    "context_content": "summary"}
        results.append(RetrievalResult(
            source="paper_chunk",
            evidence_id=str(metadata.get(PAPER_ID_KEY, "")),
            score=ranked.score,
            payload={"summary": summary, "element_type": metadata.get("element_type"),
                     "source_file": metadata.get("source_file"), "metadata": metadata},
        ))
    return results
