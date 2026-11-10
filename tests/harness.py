"""Replaces the LLM and the data stores with the synthetic scenarios from tests/golden/scenarios.py."""

from __future__ import annotations

import hashlib
import json
import types
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import scenarios as sc
from langchain_core.documents import Document

GOLDEN = Path(__file__).resolve().parent / "golden"
BASELINE = json.loads((GOLDEN / "baseline.json").read_text(encoding="utf-8"))


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Recorder:
    """Records what the code under test asks of the LLM, the graph and the indexes."""

    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.replies = list(scenario["replies"])
        self.llm_calls: list[dict[str, Any]] = []
        self.graph_calls: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    def create_chat_model(self, model: str, temperature: float = 0.0, max_retries: int = 2, **_: Any) -> Any:
        recorder = self

        class FakeChatModel:
            def invoke(self, prompt: str, *args: Any, **kwargs: Any) -> Any:
                recorder.llm_calls.append({"model": model, "temperature": temperature, "max_retries": max_retries,
                                           "prompt_sha256": sha(prompt), "prompt_chars": len(prompt)})
                return types.SimpleNamespace(content=sc.reply_to(prompt, recorder.replies))
        return FakeChatModel()

    def search_cases(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.searches.append({"index": "cases", "query": query, "top_k": top_k})
        return sc.search_cases(query, top_k, self.scenario)

    def search_rules(self, query: str, top_k: int) -> list[dict[str, Any]]:
        self.searches.append({"index": "rules", "query": query, "top_k": top_k})
        return sc.search_rules(query, top_k, self.scenario)

    def retriever(self) -> Any:
        recorder = self

        class FakeRetriever:
            def invoke(self, query: str) -> list[Document]:
                recorder.searches.append({"index": "papers", "query": query})
                return [Document(page_content=f"raw {doc_id}",
                                 metadata={"doc_id": doc_id, "element_type": "text", "source_file": "synthetic.pdf"})
                        for doc_id in sc.PAPER_DOC_IDS]
        return FakeRetriever()

    def driver(self) -> Any:
        recorder = self

        class FakeSession:
            def __enter__(self) -> "FakeSession":
                return self

            def __exit__(self, *exc: Any) -> None:
                return None

            def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
                recorder.graph_calls.append({"cypher_sha256": sha(cypher), "params": params})
                return sc.graph_rows(cypher, params)
        return types.SimpleNamespace(session=lambda database=None: FakeSession())


@contextmanager
def fake_backends(scenario: dict[str, Any]) -> Iterator[Recorder]:
    recorder = Recorder(scenario)
    rule_store = {f"rule_syn_{i}": sc.rule_bundle(i) for i in range(sc.POOL_SIZE)}
    patches = [
        mock.patch("bmrag.llm.create_chat_model", recorder.create_chat_model),
        mock.patch("bmrag.retrieval.kg_index.search_case_bundles", recorder.search_cases),
        mock.patch("bmrag.retrieval.kg_index.search_rule_bundles", recorder.search_rules),
        mock.patch("bmrag.retrieval.kg_index.load_rule_store", lambda: rule_store),
        mock.patch("bmrag.retrieval.graph._get_driver", recorder.driver),
        mock.patch("bmrag.retrieval.papers._get_retriever", recorder.retriever),
        mock.patch("bmrag.retrieval.papers._load_summary", lambda root, doc_id: f"Synthetic summary of {doc_id}."),
    ]
    for patch in patches:
        patch.start()
    try:
        yield recorder
    finally:
        for patch in patches:
            patch.stop()


def normalized(value: Any) -> Any:
    """The form a value takes in the stored JSON, so live results compare equal to it."""
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))
