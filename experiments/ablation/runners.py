"""The ablation experiments. Each takes a city record and returns the prediction.

  exp1_bmrag             full BM-RAG
  exp2_only_llm          LLM alone, no retrieval
  exp3_naive_rag         LLM + paper index, no query rewriting
  exp4_graph_rag         LLM + case index + rule index + paper index, one combined query
  exp5_bmrag_no_kg       agent pipeline, retrieval restricted to the paper index
  exp6_bmrag_llama       full BM-RAG with Llama-3.1-8B for the agent LLM calls
  exp7_bmrag_no_analyst  raw task fields replace the Analyst
  exp8_bmrag_no_judge    Reasoner runs once, unchecked
"""

from __future__ import annotations

import traceback
from dataclasses import replace
from typing import Any, Callable

from common import TASK_OUTPUT_SPEC, build_task_text, get_llm, invoke_llm, parse_json_strict, snap_to_grid

from bmrag.pipeline import run as run_pipeline
from bmrag.profiles import STAGE_PAPERS, Profile, get_profile
from bmrag.retrieval.kg_index import retrieve_cases, retrieve_rules
from bmrag.retrieval.papers import retrieve_paper_chunks

LLAMA_8B = "meta-llama/Llama-3.1-8B-Instruct"
LLAMA_70B = "meta-llama/Llama-3.3-70B-Instruct"

ONLY_LLM_PROMPT = """\
{spec}

TASK:
{task}
"""

NAIVE_RAG_PROMPT = """\
{spec}

RETRIEVED CONTEXT (top-{k} PCM paper chunks, ranked):
{context}

TASK:
{task}
"""

GRAPH_RAG_PROMPT = """\
{spec}

RETRIEVED CONTEXT
------------------
A. KG-VDB cases (top-{kc}):
{cases}

B. KG-VDB rules (top-{kr}):
{rules}

C. PCM paper chunks (top-{kp}):
{chunks}

TASK:
{task}
"""


def _run_agents(record: dict[str, Any], profile: Profile) -> dict[str, Any]:
    result = run_pipeline(dict(record["analyst_input"]), profile)
    final = result.final_prediction
    return {
        "prediction": {"pos": final.get("pcm_interface_pos"), "tm": final.get("pcm_tm_c"),
                       "rationale": final.get("joint_rationale", "") or final.get("position", "")},
        "raw_pipeline": {"verdict": result.verdict, "attempts": len(result.reasoner_attempts),
                         "confidence": final.get("confidence"), "evidence_ids": final.get("evidence_ids", []),
                         "elapsed_sec": result.elapsed_seconds},
    }


def _ask_llm(prompt: str, model: str, **extra: Any) -> dict[str, Any]:
    raw = invoke_llm(get_llm(model), prompt)
    data = parse_json_strict(raw)
    pos, tm = snap_to_grid(data.get("pcm_interface_pos"), data.get("pcm_tm_c"))
    return {"prediction": {"pos": pos, "tm": tm, "rationale": data.get("rationale", "")},
            "raw_llm_output": raw[:800], "model": model, **extra}


def exp1_bmrag(record: dict[str, Any]) -> dict[str, Any]:
    return _run_agents(record, get_profile())


def exp2_only_llm(record: dict[str, Any], model: str = "gpt-5") -> dict[str, Any]:
    return _ask_llm(ONLY_LLM_PROMPT.format(spec=TASK_OUTPUT_SPEC, task=build_task_text(record)), model)


def exp3_naive_rag(record: dict[str, Any], model: str = "gpt-5", top_k: int = 5) -> dict[str, Any]:
    task = build_task_text(record)
    chunks = retrieve_paper_chunks(task, top_k=top_k)
    context = "\n\n".join(f"[chunk {i + 1}] {c.payload.get('summary', '')[:600]}" for i, c in enumerate(chunks))
    prompt = NAIVE_RAG_PROMPT.format(spec=TASK_OUTPUT_SPEC, k=len(chunks), context=context, task=task)
    return _ask_llm(prompt, model, retrieved_count=len(chunks))


def exp4_graph_rag(record: dict[str, Any], model: str = "gpt-5", top_k_cases: int = 5, top_k_rules: int = 3,
                   top_k_chunks: int = 3) -> dict[str, Any]:
    task = build_task_text(record)
    cases = retrieve_cases(task, top_k=top_k_cases)
    rules = retrieve_rules(task, top_k=top_k_rules)
    chunks = retrieve_paper_chunks(task, top_k=top_k_chunks)
    case_text = "\n".join(
        f"[{c.evidence_id}] {(c.payload.get('case_profile', {}) or {}).get('case_summary', '')[:400]}" for c in cases)
    rule_text = "\n".join(
        f"[{r.evidence_id}] "
        f"{((r.payload.get('rule_profile', {}) or {}).get('kg_rule', {}) or {}).get('evidence_sentence', '')[:400]}"
        for r in rules)
    chunk_text = "\n\n".join(f"[chunk {i + 1}] {c.payload.get('summary', '')[:500]}" for i, c in enumerate(chunks))
    prompt = GRAPH_RAG_PROMPT.format(spec=TASK_OUTPUT_SPEC, kc=len(cases), kr=len(rules), kp=len(chunks),
                                     cases=case_text or "(none)", rules=rule_text or "(none)",
                                     chunks=chunk_text or "(none)", task=task)
    return _ask_llm(prompt, model, retrieved={"cases": len(cases), "rules": len(rules), "chunks": len(chunks)})


def exp5_bmrag_no_kg(record: dict[str, Any]) -> dict[str, Any]:
    return _run_agents(record, replace(get_profile(), retrieval_stages=frozenset({STAGE_PAPERS})))


def exp6_bmrag_llama(record: dict[str, Any]) -> dict[str, Any]:
    """Paper rerank stays on the default model, as in the recorded runs."""
    profile = replace(get_profile(), analyst_model=LLAMA_8B, reasoner_model=LLAMA_8B)
    return {**_run_agents(record, profile), "model": LLAMA_8B}


def exp7_bmrag_no_analyst(record: dict[str, Any]) -> dict[str, Any]:
    return _run_agents(record, replace(get_profile(), use_analyst=False))


def exp8_bmrag_no_judge(record: dict[str, Any]) -> dict[str, Any]:
    return _run_agents(record, replace(get_profile(), use_judge=False, max_retries=0))


def safe_run(experiment: Callable[..., dict[str, Any]], record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """Run one city, recording a failure instead of stopping the whole experiment."""
    try:
        return {**experiment(record, **kwargs), "status": "ok"}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "trace": traceback.format_exc(limit=10),
                "prediction": {"pos": None, "tm": None, "rationale": None}}


EXPERIMENTS: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "exp1_bmrag": exp1_bmrag,
    "exp2_only_llm_gpt5": lambda record: exp2_only_llm(record, model="gpt-5"),
    "exp2_only_llm_llama70b": lambda record: exp2_only_llm(record, model=LLAMA_70B),
    "exp3_naive_rag": exp3_naive_rag,
    "exp4_graph_rag": exp4_graph_rag,
    "exp5_bmrag_no_kg": exp5_bmrag_no_kg,
    "exp6_bmrag_llama": exp6_bmrag_llama,
    "exp7_bmrag_no_analyst": exp7_bmrag_no_analyst,
    "exp8_bmrag_no_judge": exp8_bmrag_no_judge,
}
