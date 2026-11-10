"""Record the behaviour baseline from the code at commit 2514c33.

Works only with that commit's layout (05_Agents/, 06_Ablation/, 03_PCM_codes/).
It is kept so the baseline can be audited: check out 2514c33, restore this folder
and run it. It makes no network call and writes only tests/golden/baseline.json
and, if the local stores exist, tests/golden/baseline_local_data.json.

Usage (from the repo root):
  .venv/Scripts/python.exe -X utf8 tests/golden/record_baseline_2514c33.py
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(REPO / "05_Agents"), str(REPO / "06_Ablation"),
                str(REPO / "tools" / "diagnostics")]

import scenarios as sc  # noqa: E402

for name in ("ANALYST_MODEL", "REASONER_MODEL", "JUDGE_MODEL", "PROBLEM_SET_PATH"):
    os.environ.pop(name, None)

import llm_factory  # noqa: E402
import retrievers  # noqa: E402
import kg_case_vdb_utils  # noqa: E402
import VDB_summarization_embedding_rerank_llm_V1 as v1  # noqa: E402
import reasoner  # noqa: E402
import judge as judge_mod  # noqa: E402
import orchestrator  # noqa: E402
import analyst  # noqa: E402
import common  # noqa: E402
import runners  # noqa: E402
import eval_energyplus  # noqa: E402
from langchain_core.documents import Document  # noqa: E402


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Recorder:
    def __init__(self, scenario: dict[str, Any]) -> None:
        self.scenario = scenario
        self.replies = list(scenario["replies"])
        self.llm_calls: list[dict[str, Any]] = []
        self.graph_calls: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []

    def chat_model(self, model: str, temperature: float = 0.0, max_retries: int | None = None) -> Any:
        recorder = self

        class FakeChatModel:
            def invoke(self, prompt: str, *a: Any, **k: Any) -> Any:
                recorder.llm_calls.append({"model": model, "temperature": temperature, "max_retries": max_retries,
                                           "prompt_sha256": sha(prompt), "prompt_chars": len(prompt)})
                return types.SimpleNamespace(content=sc.reply_to(prompt, recorder.replies))
        return FakeChatModel()


class FakeSession:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def __enter__(self) -> "FakeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def run(self, cypher: str, **params: Any) -> list[dict[str, Any]]:
        self.recorder.graph_calls.append({"cypher_sha256": sha(cypher), "params": params})
        return sc.graph_rows(cypher, params)


class FakeRetriever:
    def __init__(self, recorder: Recorder) -> None:
        self.recorder = recorder

    def invoke(self, query: str) -> list[Document]:
        self.recorder.searches.append({"index": "papers", "query": query})
        return [Document(page_content=f"raw {doc_id}",
                         metadata={"doc_id": doc_id, "element_type": "text", "source_file": "synthetic.pdf"})
                for doc_id in sc.PAPER_DOC_IDS]


def install(recorder: Recorder) -> None:
    def kg_cases(query: str, top_k: int, **_: Any) -> dict[str, Any]:
        recorder.searches.append({"index": "cases", "query": query, "top_k": top_k})
        return {"results": sc.search_cases(query, top_k, recorder.scenario)}

    def kg_rules(query: str, top_k: int, **_: Any) -> dict[str, Any]:
        recorder.searches.append({"index": "rules", "query": query, "top_k": top_k})
        return {"results": sc.search_rules(query, top_k, recorder.scenario)}

    def fake_chat_openai(model: str, temperature: float = 0.0, max_retries: int | None = None) -> Any:
        return recorder.chat_model(model, temperature, max_retries)

    retrievers._kg_query_cases = kg_cases
    retrievers._kg_query_rules = kg_rules
    retrievers._get_driver = lambda: types.SimpleNamespace(session=lambda database=None: FakeSession(recorder))
    retrievers._get_pcm_vdb_retriever = lambda: FakeRetriever(recorder)
    llm_factory.get_chat_llm = fake_chat_openai
    v1.ChatOpenAI = fake_chat_openai
    v1._load_cached_summary = lambda root, doc_id: f"Synthetic summary of {doc_id}."
    runners.get_llm = lambda model, temperature=0.0: fake_chat_openai(model, temperature)
    kg_case_vdb_utils.load_rule_store = lambda *_a, **_k: {f"rule_syn_{i}": sc.rule_bundle(i) for i in range(sc.POOL_SIZE)}
    judge_mod.load_rule_store = kg_case_vdb_utils.load_rule_store
    judge_mod._RULE_STORE_CACHE = None


def clean_result(result: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(result)
    result.pop("elapsed_seconds", None)
    return result


def record_pipeline() -> dict[str, Any]:
    out = {}
    for name, scenario in sc.SCENARIOS.items():
        recorder = Recorder(scenario)
        install(recorder)
        sources = set(scenario["sources"]) if scenario["sources"] else None
        result = orchestrator.run(dict(sc.TASK), max_retries=scenario["max_retries"], retrieval_sources=sources)
        out[name] = {"llm_calls": recorder.llm_calls, "graph_calls": recorder.graph_calls,
                     "searches": recorder.searches, "result": clean_result(result.to_dict())}
    return out


def record_events() -> dict[str, Any]:
    import bmrag_trace

    def strip(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{k: v for k, v in e.items() if k not in ("t", "elapsed_sec")} for e in events]

    out = {}
    for name in ("pass_first_attempt", "soft_retry_exhausted", "rewrite_and_all_stages"):
        recorder = Recorder(sc.SCENARIOS[name])
        install(recorder)
        trace = bmrag_trace.Trace()
        bmrag_trace.run_traced(dict(sc.TASK), trace)
        out[name] = strip(trace.events)
    recorder = Recorder(sc.SCENARIOS["rewrite_and_all_stages"])
    install(recorder)
    scan = bmrag_trace.run_scan({"city": "testville", "zone": "synthetic", "analyst_input": dict(sc.TASK)})
    out["scan"] = {"events": strip(scan["events"]), "stages_attempt1": scan["stages_attempt1"],
                   "stages_retry": scan["stages_retry"], "template_check": {
                       k: v.get("n", "error") for k, v in scan["template_check"].items()}}
    return out


def record_experiments() -> dict[str, Any]:
    record = {"city": "testville", "zone": "synthetic", "analyst_input": dict(sc.TASK)}
    out = {}
    # exp6 re-imports the agent modules and changes the environment, so it runs last
    for name in ("exp1_bmrag", "exp2_only_llm_gpt5", "exp2_only_llm_llama70b", "exp3_naive_rag", "exp4_graph_rag",
                 "exp5_bmrag_no_kg", "exp7_bmrag_no_analyst", "exp8_bmrag_no_judge", "exp6_bmrag_llama"):
        recorder = Recorder(sc.SCENARIOS["rewrite_and_all_stages"])
        install(recorder)
        result = runners.EXPERIMENTS[name](dict(record))
        result.get("raw_pipeline", {}).pop("elapsed_sec", None)
        out[name] = {"llm_calls": recorder.llm_calls, "graph_calls": recorder.graph_calls,
                     "searches": recorder.searches, "result": result}
    for name in ("ANALYST_MODEL", "REASONER_MODEL", "JUDGE_MODEL"):
        os.environ.pop(name, None)
    return out


def record_constants() -> dict[str, Any]:
    texts = {
        "analyst": analyst.ANALYST_PROMPT,
        "reasoner": reasoner.REASONER_PROMPT,
        "case_query_rewrite": retrievers._REWRITE_PROMPT,
        "paper_query": reasoner._STAGE5_QUERY_PROMPT,
        "paper_rerank": v1.LLM_RERANK_PROMPT,
        "ablation_task_spec": common.TASK_OUTPUT_SPEC,
        "ablation_only_llm": runners.ONLY_LLM_PROMPT,
        "ablation_naive_rag": runners.NAIVE_RAG_PROMPT,
        "ablation_graph_rag": runners.GRAPH_RAG_PROMPT,
        "position_semantics": reasoner.PCM_POS_SEMANTICS,
        "mechanism_case_query": reasoner.MECHANISM_CASE_QUERY,
        "mechanism_rule_query": reasoner.MECHANISM_RULE_QUERY,
    }
    return {"texts": {k: sha(v) for k, v in texts.items()},
            "cypher": {k: sha(v) for k, v in retrievers.CYPHER_TEMPLATES.items()},
            "template_defaults": retrievers._TEMPLATE_DEFAULTS}


def record_scoring(tmp: Path) -> dict[str, Any]:
    csv_path = tmp / "synthetic_summary.csv"
    csv_path.write_text(sc.SYNTHETIC_LOOKUP_CSV, encoding="utf-8")
    eval_energyplus.SUMMARY_CSV = csv_path
    lookup = eval_energyplus.build_lookup()
    scored = [eval_energyplus.score_one("alpha", pos, tm, lookup) for pos, tm in sc.SCORING_CASES]
    grid = {f"{k[0]}/{k[1]}": v for k, v in lookup["alpha"]["grid"].items()}
    csv_path.unlink()
    return {"lookup": {**{k: v for k, v in lookup["alpha"].items() if k != "grid"}, "grid": grid}, "scored": scored}


KG_STAGES = {"kg_cases", "kg_rules", "neo4j_climate", "neo4j_rules"}


class HashEmbeddings:
    """Deterministic stand-in for the embedding API, so real stores can be queried offline."""

    def _vector(self, text: str) -> list[float]:
        import numpy as np
        seed = int(sha(text)[:16], 16) % (2 ** 32)
        vector = np.random.default_rng(seed).normal(size=1536)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


def record_local_data() -> dict[str, Any] | None:
    """Prompt hashes with real stores and fake embeddings. Stores only hashes.

    Paper search is left out: a random query vector has no clear neighbours among 711
    documents, so the approximate search can order ties differently between processes.
    """
    run_dir = REPO / "runs" / "20261102_005635_full"
    stores = REPO / "03_PCM_codes" / "PCM_KG_VDB"
    if not run_dir.exists() or not stores.exists():
        return None
    import query_kg_case_index
    query_kg_case_index.OpenAIEmbeddings = lambda model=None: HashEmbeddings()
    v1.OpenAIEmbeddings = lambda model=None: HashEmbeddings()

    out: dict[str, Any] = {"cities": {}}

    captured: list[dict[str, Any]] = []
    replies = [sc.reasoner_reply(4, 20.0, [])]

    def fake_chat(model: str, temperature: float = 0.0, max_retries: int | None = None) -> Any:
        class Fake:
            def invoke(self, prompt: str, *a: Any, **k: Any) -> Any:
                captured.append({"model": model, "prompt_sha256": sha(prompt), "prompt_chars": len(prompt)})
                return types.SimpleNamespace(content=sc.reply_to(prompt, replies))
        return Fake()

    llm_factory.get_chat_llm = fake_chat
    v1.ChatOpenAI = fake_chat
    retrievers._get_driver = lambda: types.SimpleNamespace(session=lambda database=None: FakeSession(Recorder(
        sc.SCENARIOS["pass_first_attempt"])))
    for city in ("beijing", "chicago", "sandiego"):
        run = json.loads((run_dir / f"{city}.json").read_text(encoding="utf-8"))
        a = analyst.AnalystOutput(**run["pipeline_result"]["analyst"])
        captured.clear()
        result = reasoner.reason(a, top_k_cases=5, top_k_rules=3, retrieval_sources=KG_STAGES)
        out["cities"][city] = {"llm_calls": list(captured), "stages": result.diagnostics["stages"]}
    return out


def main() -> None:
    import tempfile
    # Local data first: the other recordings replace retrieval functions and re-import agents
    local = record_local_data()
    if local:
        (HERE / "baseline_local_data.json").write_text(json.dumps(local, ensure_ascii=False, indent=1) + "\n",
                                                       encoding="utf-8")
        print("wrote baseline_local_data.json")
    baseline = {"source_commit": "2514c33", "constants": record_constants()}
    with tempfile.TemporaryDirectory() as tmp:
        baseline["scoring"] = record_scoring(Path(tmp))
    baseline["pipeline"] = record_pipeline()
    baseline["events"] = record_events()
    baseline["experiments"] = record_experiments()
    (HERE / "baseline.json").write_text(json.dumps(baseline, ensure_ascii=False, indent=1, default=str) + "\n",
                                        encoding="utf-8")
    print("wrote baseline.json")


if __name__ == "__main__":
    main()
