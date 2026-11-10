"""
run_eval.py
===========

Main script of the RAG evaluation (phases 1 and 2).

Each test query runs the full pipeline, recording:
  - vector search results (stage 1 doc_ids, for Hit Rate, MRR and nDCG)
  - rerank results (stage 2 doc_ids and scores)
  - the generated answer
  - LLM-judge metrics (faithfulness, answer_correctness, context_precision, context_recall)

Five ablation variants:
  A  — summary embedding + LLM rerank (baseline); context is the raw chunk
  B  — summary embedding, no rerank (top-3 vector results); context is the raw chunk
  C  — raw chunk embedding + LLM rerank (run build_raw_chunk_index.py first)
  D  — raw chunk embedding, no rerank
  E  — summary embedding + LLM rerank; context is the text summary (the only difference from A)

Each judge field has a *_status (valid, api_failed, parse_failed, skipped) and a
*_raw (the raw reply), so an API failure cannot pass as a low score.

Usage:
  # Baseline variant A (full set)
  python eval/run_eval.py \\
      --testset eval/qa_testset_lab_be_v1.jsonl \\
      --cache_root "D:/.../multirepresentation_chunks_cache_LAB_B&E2025_chksize_6k_cleaned" \\
      --variant A \\
      --judge_model gpt-5.2 \\
      --output eval/results/variant_A.jsonl

  # Oracle upper bound (forces the gold chunk as context)
  python eval/run_eval.py --variant A --oracle ...

  # Dry run (no LLM calls)
  python eval/run_eval.py --testset ... --cache_root ... --variant A --dry_run

  # Skip the judge (evaluates retrieval only)
  python eval/run_eval.py ... --skip_judge
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_EVAL_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from rag_utils import (  # noqa: E402
    ID_KEY, RERANK_TOP_K, RETRIEVE_TOP_K,
    _load_env,
    build_retriever,
    generate_answer,
    format_context_text,
    prepare_context_documents,
    retrieve_stage1,
    rerank_stage2,
)
from judge_prompts import (  # noqa: E402
    FAITHFULNESS_PROMPT,
    ANSWER_CORRECTNESS_PROMPT,
    CONTEXT_PRECISION_PROMPT,
    CONTEXT_RECALL_PROMPT,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bmrag.config import get_paths  # noqa: E402

# ── Default paths ─────────────────────────────────────────────────────────────
DEFAULT_CACHE = get_paths().data_dir / "lab_be_index"
RAW_CHUNK_CHROMA_DIR = "chroma_db_raw"   # written by build_raw_chunk_index.py

HIT_K_VALUES = [1, 3, 5, 10, 20]

# Abstain detection
_ABSTAIN_PATTERNS = [
    r"cannot (?:find|determine|answer)",
    r"not (?:provided|mentioned|available) in the (?:context|provided|given)",
    r"no (?:information|evidence|mention) (?:about|of|regarding)",
    r"the (?:context|provided|given) (?:does not|doesn't) (?:contain|mention|include|provide)",
    r"(?:I |the model )?(?:don't|do not|cannot) have (?:enough|sufficient|any) (?:information|context)",
    r"based on the (?:provided|given) context.*?(?:cannot|can't|unable to)",
]


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval metrics (no LLM needed)
# ══════════════════════════════════════════════════════════════════════════════

def hit_at_k(gold_id: str, ranked_ids: list[str], k: int) -> bool:
    return gold_id in ranked_ids[:k]


def reciprocal_rank(gold_id: str, ranked_ids: list[str]) -> float:
    try:
        rank = ranked_ids.index(gold_id) + 1
        return 1.0 / rank
    except ValueError:
        return 0.0


def ndcg_at_k(gold_id: str, ranked_ids: list[str], k: int) -> float:
    """Binary relevance nDCG@k (1 gold doc, rest irrelevant)."""
    dcg = 0.0
    for rank, doc_id in enumerate(ranked_ids[:k], start=1):
        if doc_id == gold_id:
            dcg = 1.0 / math.log2(rank + 1)
            break
    idcg = 1.0 / math.log2(2)   # ideal: gold at position 1 → = 1.0
    return dcg / idcg


def _rank_of(gold_id: str, ranked_ids: list[str]) -> Optional[int]:
    """Return 1-based rank of gold_id in ranked_ids, or None if not found."""
    try:
        return ranked_ids.index(gold_id) + 1
    except ValueError:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM judge primitives
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(client, model: str, prompt: str, max_retries: int = 3) -> Optional[str]:
    """Call LLM and return response text, or None if all retries fail."""
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            wait = 2 ** attempt
            print(f"  [warn] LLM call failed (attempt {attempt+1}/{max_retries}): {e}; sleep {wait}s")
            time.sleep(wait)
    print(f"  [error] All {max_retries} LLM retries exhausted — returning None")
    return None   # ← None, NOT "", so callers can distinguish failure from empty response


def _parse_obj(raw: Optional[str]) -> Optional[dict]:
    """Parse JSON object from LLM response. Returns None on parse failure or None input."""
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        result = json.loads(s)
        return result if isinstance(result, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                return result if isinstance(result, dict) else None
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════════════════
# LLM judge functions — each returns a dict with status + score + raw
# ══════════════════════════════════════════════════════════════════════════════

def judge_faithfulness(
    client, model: str, question: str, context: str, answer: str
) -> dict:
    """
    Returns:
      score: float 0-1 | None
      status: "valid" | "api_failed" | "parse_failed" | "skipped"
      raw: str (original LLM response)
      claims: list
      reasoning: str
    """
    if not answer.strip():
        return {
            "score": None, "status": "skipped",
            "raw": "", "claims": [], "reasoning": "empty answer",
        }

    raw = _call_llm(client, model, FAITHFULNESS_PROMPT.format(
        question=question, context=context, answer=answer))

    if raw is None:
        return {
            "score": None, "status": "api_failed",
            "raw": "", "claims": [], "reasoning": "",
        }

    obj = _parse_obj(raw)
    if obj is None:
        return {
            "score": None, "status": "parse_failed",
            "raw": raw, "claims": [], "reasoning": "",
        }

    try:
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return {
            "score": None, "status": "parse_failed",
            "raw": raw, "claims": obj.get("claims", []), "reasoning": obj.get("reasoning", ""),
        }

    return {
        "score": score, "status": "valid",
        "raw": raw,
        "claims": obj.get("claims", []),
        "reasoning": obj.get("reasoning", ""),
    }


def judge_answer_correctness(
    client, model: str, question: str, gold: str, generated: str
) -> dict:
    """
    Returns:
      score: float (0 / 0.5 / 1) | None
      status: "valid" | "api_failed" | "parse_failed" | "skipped"
      raw: str
      reasoning: str
    """
    if not generated.strip():
        return {"score": None, "status": "skipped", "raw": "", "reasoning": "empty answer"}

    raw = _call_llm(client, model, ANSWER_CORRECTNESS_PROMPT.format(
        question=question, gold_answer=gold, generated_answer=generated))

    if raw is None:
        return {"score": None, "status": "api_failed", "raw": "", "reasoning": ""}

    obj = _parse_obj(raw)
    if obj is None:
        return {"score": None, "status": "parse_failed", "raw": raw, "reasoning": ""}

    try:
        score = float(obj.get("score", 0))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return {"score": None, "status": "parse_failed", "raw": raw, "reasoning": obj.get("reasoning", "")}

    return {
        "score": score, "status": "valid",
        "raw": raw, "reasoning": obj.get("reasoning", ""),
    }


def judge_context_precision(
    client, model: str, question: str, docs: list, k_values: list[int]
) -> dict:
    """
    Score each retrieved doc for relevance.
    Returns dict with:
      per_doc_scores: list[float | None]
      precision_at_k: {str(k): float | None}
      status: "valid" | "partial" | "api_failed"
    """
    per_doc_scores: list[Optional[float]] = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown")
        etype  = doc.metadata.get("element_type", "text")
        text   = doc.page_content[:2000]
        raw = _call_llm(client, model, CONTEXT_PRECISION_PROMPT.format(
            question=question, source=source, element_type=etype, doc_text=text))

        if raw is None:
            per_doc_scores.append(None)
            continue

        obj = _parse_obj(raw)
        if obj is None:
            per_doc_scores.append(None)
            continue

        try:
            score = float(obj.get("score", 0.5))
            per_doc_scores.append(max(0.0, min(1.0, score)))
        except (TypeError, ValueError):
            per_doc_scores.append(None)

    # Determine status
    n_valid = sum(1 for s in per_doc_scores if s is not None)
    if n_valid == len(per_doc_scores):
        status = "valid"
    elif n_valid > 0:
        status = "partial"
    else:
        status = "api_failed"

    # precision@k: use only valid scores; mark None if no valid scores at all
    precision_at_k: dict[str, Optional[float]] = {}
    for k in k_values:
        top_k = per_doc_scores[:k]
        valid_in_k = [s for s in top_k if s is not None]
        if valid_in_k:
            precision_at_k[str(k)] = round(sum(valid_in_k) / k, 4)
        else:
            precision_at_k[str(k)] = None

    return {
        "per_doc_scores": per_doc_scores,
        "precision_at_k": precision_at_k,
        "status": status,
    }


def judge_context_recall(
    client, model: str, gold_answer: str, context: str
) -> dict:
    """
    Returns:
      score: float 0-1 | None
      status: "valid" | "api_failed" | "parse_failed"
      raw: str
      missing_info: list
      reasoning: str
    """
    raw = _call_llm(client, model, CONTEXT_RECALL_PROMPT.format(
        gold_answer=gold_answer, context=context))

    if raw is None:
        return {
            "score": None, "status": "api_failed",
            "raw": "", "missing_info": [], "reasoning": "",
        }

    obj = _parse_obj(raw)
    if obj is None:
        return {
            "score": None, "status": "parse_failed",
            "raw": raw, "missing_info": [], "reasoning": "",
        }

    try:
        score = float(obj.get("score", 0.0))
        score = max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return {
            "score": None, "status": "parse_failed",
            "raw": raw, "missing_info": obj.get("missing_info", []), "reasoning": "",
        }

    return {
        "score": score, "status": "valid",
        "raw": raw,
        "missing_info": obj.get("missing_info", []),
        "reasoning": obj.get("reasoning", ""),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Per-query eval
# ══════════════════════════════════════════════════════════════════════════════

def _detect_abstain(text: str) -> bool:
    """Detect if the generated answer is an abstention/refusal."""
    if not text:
        return False
    return any(re.search(p, text, re.IGNORECASE) for p in _ABSTAIN_PATTERNS)


def eval_one(
    item: dict,
    retriever,
    cache_root: Path,
    variant: str,
    judge_client,
    judge_model: str,
    gen_model: str,
    use_rerank: bool,
    rerank_top_k: int,
    k_values: list[int],
    skip_judge: bool,
    dry_run: bool,
    oracle: bool = False,
    docstore=None,
) -> dict:
    qid          = item["qid"]
    question     = item["question"]
    gold_id      = item["gold_chunk_id"]
    gold_answer  = item["gold_answer"]

    result: dict = {
        "qid": qid,
        "question": question,
        "gold_chunk_id": gold_id,
        "gold_answer": gold_answer,
        "gold_paper": item.get("gold_paper", ""),
        "element_type": item.get("element_type", ""),
        "question_type": item.get("question_type", "unknown"),
        "category": item.get("category", "unknown"),
        "variant": variant,
        "oracle": oracle,
        # Stage 1
        "stage1_ids": [],
        "stage1_hit": {},
        "stage1_mrr": 0.0,
        "stage1_ndcg10": 0.0,
        "rank_of_gold_stage1": None,
        # Stage 2
        "stage2_ids": [],
        "stage2_scores": [],
        "stage2_hit_at_1": False,
        "stage2_hit_at_3": False,
        "stage2_hit_at_5": None,  # None when top_k < 5
        "stage2_mrr": 0.0,
        "stage2_ndcg3": 0.0,
        "rank_of_gold_stage2": None,
        "gold_found_stage": None,  # "stage2" | "stage1_only" | "not_found"
        # Generation
        "generated_answer": "",
        "context_text": "",
        "abstained": False,
        # Judge — faithfulness
        "faithfulness_score": None,
        "faithfulness_status": "skipped",
        "faithfulness_raw": "",
        "faithfulness_claims": [],
        "faithfulness_reasoning": "",
        # Judge — answer correctness
        "answer_correctness_score": None,
        "answer_correctness_status": "skipped",
        "answer_correctness_raw": "",
        "answer_correctness_reasoning": "",
        # Judge — context precision (per-doc + @k)
        "context_precision_scores": [],
        "context_precision_at_k": {},
        "context_precision_status": "skipped",
        # Judge — context recall
        "context_recall_score": None,
        "context_recall_status": "skipped",
        "context_recall_raw": "",
        # Stage-level status
        "retrieval_status": "skipped",
        "rerank_status": "skipped",
        "generation_status": "skipped",
        "judge_status": "skipped",
        # Meta
        "latency_s": 0.0,
        "error": None,
    }

    if dry_run:
        return result

    t0 = time.time()
    try:
        if oracle:
            # ── Oracle mode: force gold chunk into context ─────────────────
            result["retrieval_status"] = "skipped"
            result["rerank_status"]    = "skipped"
            if docstore is None:
                raise ValueError("Oracle mode requires docstore to be provided")
            gold_doc = docstore.mget([gold_id])[0]
            if gold_doc is None:
                raise ValueError(f"Gold chunk {gold_id} not found in docstore")
            top_docs   = [gold_doc]
            stage1_ids = [gold_id]
            stage2_ids = [gold_id]
            result["stage1_ids"]        = stage1_ids
            result["stage2_ids"]        = stage2_ids
            result["stage1_hit"]        = {str(k): True for k in k_values}
            result["stage1_mrr"]        = 1.0
            result["stage1_ndcg10"]     = 1.0
            result["rank_of_gold_stage1"] = 1
            result["stage2_hit_at_1"]   = True
            result["stage2_hit_at_3"]   = True
            result["stage2_hit_at_5"]   = True
            result["stage2_mrr"]        = 1.0
            result["stage2_ndcg3"]      = 1.0
            result["rank_of_gold_stage2"] = 1
            result["gold_found_stage"]  = "oracle"
        else:
            # ── Stage 1: vector retrieval ──────────────────────────────────
            try:
                raw_docs, stage1_ids = retrieve_stage1(question, retriever)
                if variant == "D":
                    raw_docs = [
                        doc for doc in raw_docs
                        if doc.metadata.get("element_type", "text") == "text"
                    ]
                    stage1_ids = [doc.metadata.get(ID_KEY, "") for doc in raw_docs]
                result["stage1_ids"]   = stage1_ids
                result["retrieval_status"] = "success"
            except Exception as e:
                result["retrieval_status"] = "failed"
                result["error"] = f"retrieval: {e}"
                result["latency_s"] = round(time.time() - t0, 2)
                return result

            # Retrieval metrics (pre-rerank)
            result["stage1_hit"]     = {str(k): hit_at_k(gold_id, stage1_ids, k) for k in k_values}
            result["stage1_mrr"]     = reciprocal_rank(gold_id, stage1_ids)
            result["stage1_ndcg10"]  = ndcg_at_k(gold_id, stage1_ids, 10)
            result["rank_of_gold_stage1"] = _rank_of(gold_id, stage1_ids)

            # ── Stage 2: rerank (or not) ───────────────────────────────────
            if use_rerank:
                try:
                    scoring_mode = "raw_text" if variant == "C" else "summary"
                    top_docs, rerank_scores = rerank_stage2(
                        question, raw_docs, cache_root,
                        top_k=rerank_top_k,
                        scoring_mode=scoring_mode,
                        text_use_summary=(variant == "E"),
                    )
                    stage2_ids = [doc.metadata.get(ID_KEY, "") for doc in top_docs]
                    result["stage2_ids"]    = stage2_ids
                    result["stage2_scores"] = rerank_scores
                    result["rerank_status"] = "success"
                except Exception as e:
                    result["rerank_status"] = "failed"
                    # Fall back to top-k from stage1
                    top_docs   = prepare_context_documents(raw_docs[:rerank_top_k], cache_root,
                                                           text_use_summary=(variant == "E"))
                    stage2_ids = stage1_ids[:rerank_top_k]
                    result["stage2_ids"]  = stage2_ids
                    result["error"] = f"rerank: {e}"
            else:
                # No-rerank: take first rerank_top_k from stage1 directly
                top_docs   = prepare_context_documents(raw_docs[:rerank_top_k], cache_root,
                                                       text_use_summary=(variant == "E"))
                stage2_ids = stage1_ids[:rerank_top_k]
                result["stage2_ids"]    = stage2_ids
                result["stage2_scores"] = []
                result["rerank_status"] = "skipped"

            # Stage 2 metrics
            result["stage2_hit_at_1"]  = gold_id in stage2_ids[:1]
            result["stage2_hit_at_3"]  = gold_id in stage2_ids[:3]
            result["stage2_hit_at_5"]  = (gold_id in stage2_ids[:5]) if len(stage2_ids) >= 5 else None
            result["stage2_mrr"]       = reciprocal_rank(gold_id, stage2_ids)
            result["stage2_ndcg3"]     = ndcg_at_k(gold_id, stage2_ids, 3)
            result["rank_of_gold_stage2"] = _rank_of(gold_id, stage2_ids)

            # Where was gold ultimately found?
            if gold_id in stage2_ids:
                result["gold_found_stage"] = "stage2"
            elif gold_id in stage1_ids:
                result["gold_found_stage"] = "stage1_only"
            else:
                result["gold_found_stage"] = "not_found"

        # ── Generation ────────────────────────────────────────────────────
        try:
            context_text = format_context_text(top_docs)
            result["context_text"] = context_text
            generated = generate_answer(question, top_docs, model=gen_model)
            result["generated_answer"] = generated
            result["generation_status"] = "success"
            result["abstained"] = _detect_abstain(generated)
        except Exception as e:
            result["generation_status"] = "failed"
            result["error"] = (result.get("error") or "") + f" | generation: {e}"
            result["latency_s"] = round(time.time() - t0, 2)
            return result

        # ── LLM judge ─────────────────────────────────────────────────────
        if not skip_judge and judge_client is not None:
            judge_results: list[str] = []  # collect statuses

            # Faithfulness
            faith = judge_faithfulness(
                judge_client, judge_model, question, context_text, generated)
            result["faithfulness_score"]     = faith["score"]
            result["faithfulness_status"]    = faith["status"]
            result["faithfulness_raw"]       = faith["raw"]
            result["faithfulness_claims"]    = faith["claims"]
            result["faithfulness_reasoning"] = faith["reasoning"]
            judge_results.append(faith["status"])

            # Answer correctness
            corr = judge_answer_correctness(
                judge_client, judge_model, question, gold_answer, generated)
            result["answer_correctness_score"]     = corr["score"]
            result["answer_correctness_status"]    = corr["status"]
            result["answer_correctness_raw"]       = corr["raw"]
            result["answer_correctness_reasoning"] = corr["reasoning"]
            judge_results.append(corr["status"])

            # Context precision
            prec = judge_context_precision(
                judge_client, judge_model, question, top_docs, k_values=[1, 3])
            result["context_precision_scores"] = prec["per_doc_scores"]
            result["context_precision_at_k"]   = prec["precision_at_k"]
            result["context_precision_status"] = prec["status"]
            judge_results.append(prec["status"])

            # Context recall
            recall = judge_context_recall(
                judge_client, judge_model, gold_answer, context_text)
            result["context_recall_score"]  = recall["score"]
            result["context_recall_status"] = recall["status"]
            result["context_recall_raw"]    = recall["raw"]
            judge_results.append(recall["status"])

            # Overall judge status
            if all(s in ("valid", "skipped") for s in judge_results):
                result["judge_status"] = "success"
            elif any(s == "valid" for s in judge_results):
                result["judge_status"] = "partial"
            else:
                result["judge_status"] = "failed"

    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
        print(f"  [error] {qid}: {e}")

    result["latency_s"] = round(time.time() - t0, 2)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Aggregate metrics
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(results: list[dict], k_values: list[int]) -> dict:
    n = len(results)
    if n == 0:
        return {}

    def avg(vals):
        """Mean of non-None values, or None if empty."""
        v = [x for x in vals if x is not None]
        return round(sum(v) / len(v), 4) if v else None

    def valid_scores(results, score_key, status_key):
        """Extract scores where status == 'valid'."""
        return [
            r[score_key] for r in results
            if r.get(status_key) == "valid" and r.get(score_key) is not None
        ]

    # Stage 1 metrics (all questions that didn't have retrieval errors)
    retrieval_results = [r for r in results if r.get("retrieval_status") != "failed"]

    agg: dict = {
        "n_total":   n,
        "n_errors":  sum(1 for r in results if r.get("error")),
        "n_retrieval_valid": len(retrieval_results),
        "stage1": {
            **{f"hit_rate_at_{k}": avg([
                r["stage1_hit"].get(str(k)) for r in retrieval_results
            ]) for k in k_values},
            "mrr":    avg([r["stage1_mrr"]    for r in retrieval_results]),
            "ndcg10": avg([r["stage1_ndcg10"] for r in retrieval_results]),
        },
        "stage2": {
            "hit_rate_at_1": avg([r["stage2_hit_at_1"] for r in retrieval_results]),
            "hit_rate_at_3": avg([r["stage2_hit_at_3"] for r in retrieval_results]),
            "hit_rate_at_5": avg([r["stage2_hit_at_5"] for r in retrieval_results]),
            "mrr":           avg([r["stage2_mrr"]      for r in retrieval_results]),
            "ndcg3":         avg([r["stage2_ndcg3"]    for r in retrieval_results]),
            "delta_mrr":     None,   # filled below
        },
    }

    # Stage 2 delta MRR
    s1_mrr = agg["stage1"]["mrr"] or 0.0
    s2_mrr = agg["stage2"]["mrr"] or 0.0
    agg["stage2"]["delta_mrr"] = round(s2_mrr - s1_mrr, 4)

    # Gold found stage breakdown
    gold_found_counts = {"stage2": 0, "stage1_only": 0, "not_found": 0, "oracle": 0}
    for r in results:
        stage = r.get("gold_found_stage")
        if stage in gold_found_counts:
            gold_found_counts[stage] += 1
    agg["gold_found_breakdown"] = gold_found_counts

    # Generation metrics — only count valid judge responses
    faith_scores = valid_scores(results, "faithfulness_score", "faithfulness_status")
    corr_scores  = valid_scores(results, "answer_correctness_score", "answer_correctness_status")
    recall_scores = valid_scores(results, "context_recall_score", "context_recall_status")

    # context_precision @k — only valid entries from context_precision_status
    prec_results_valid = [r for r in results if r.get("context_precision_status") in ("valid", "partial")]
    prec1_scores = [r["context_precision_at_k"].get("1") for r in prec_results_valid
                    if r["context_precision_at_k"].get("1") is not None]
    prec3_scores = [r["context_precision_at_k"].get("3") for r in prec_results_valid
                    if r["context_precision_at_k"].get("3") is not None]

    n_gen_valid = sum(1 for r in results if r.get("generation_status") == "success")

    agg["generation"] = {
        "faithfulness":           avg(faith_scores),
        "n_faithfulness_valid":   len(faith_scores),
        "answer_correctness":     avg(corr_scores),
        "n_answer_correctness_valid": len(corr_scores),
        "context_precision_at_1": avg(prec1_scores),
        "context_precision_at_3": avg(prec3_scores),
        "n_context_precision_valid": len(prec_results_valid),
        "context_recall":         avg(recall_scores),
        "n_context_recall_valid": len(recall_scores),
        "n_generation_valid":     n_gen_valid,
        "n_abstained":            sum(1 for r in results if r.get("abstained")),
    }

    agg["latency"] = {
        "mean_s": avg([r["latency_s"] for r in results]),
    }

    # Judge status summary
    agg["judge_status_counts"] = {
        s: sum(1 for r in results if r.get("judge_status") == s)
        for s in ("success", "partial", "failed", "skipped")
    }

    return agg


# ══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_summary(path: Path, variant: str, agg: dict, run_config: dict) -> None:
    summary = {
        "variant": variant,
        "aggregate": agg,
        "run_config": run_config,
    }
    path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--testset", type=Path, required=True,
                        help="Path to qa_testset_lab_be_v1.jsonl")
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE,
                        help="multirepresentation_chunks_cache_* directory (for summary index).")
    parser.add_argument("--raw_index_root", type=Path, default=None,
                        help="Legacy raw-chunk index root; current variants use the summary index.")
    parser.add_argument("--variant", choices=["A", "B", "C", "D", "E"], default="A",
                        help="A=summary+rerank, raw-chunk context (baseline); "
                             "B=summary+no-rerank, raw-chunk context; "
                             "C=summary retrieval+raw-text rerank; "
                             "D=text-only summary+no-rerank; "
                             "E=summary+rerank, summary context for text (vs A).")
    parser.add_argument("--collection_name",     type=str, default="multi_modal_rag")
    parser.add_argument("--raw_collection_name", type=str, default="multi_modal_rag_raw")
    parser.add_argument("--gen_model",   type=str, default="gpt-4o")
    parser.add_argument("--judge_model", type=str, default="gpt-5.2")
    parser.add_argument("--retrieve_top_k", type=int, default=RETRIEVE_TOP_K,
                        help=f"Number of docs for Stage 1 retrieval (default: {RETRIEVE_TOP_K})")
    parser.add_argument("--rerank_top_k",   type=int, default=RERANK_TOP_K,
                        help=f"Number of docs to keep after rerank (default: {RERANK_TOP_K})")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSONL path. Defaults to eval/results/variant_<X>.jsonl")
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip LLM calls; just verify dataset loading and retriever init.")
    parser.add_argument("--skip_judge", action="store_true",
                        help="Run retrieval + generation but skip LLM judge metrics.")
    parser.add_argument("--oracle", action="store_true",
                        help="Oracle mode: force gold chunk into context to isolate generation quality. "
                             "Skips retrieval/rerank; requires docstore access.")
    parser.add_argument("--max_questions", type=int, default=None,
                        help="Limit to first N questions (for quick pilot runs).")
    args = parser.parse_args()

    _load_env()

    # ── Determine variant config ──────────────────────────────────────────────
    use_rerank   = args.variant in ("A", "C", "D", "E") and not args.oracle
    use_raw_idx  = False

    cache_root  = args.cache_root
    index_root  = args.raw_index_root or cache_root

    # ── Build retriever ───────────────────────────────────────────────────────
    docstore = None  # needed for oracle mode

    if use_raw_idx:
        from rag_utils import Chroma, OpenAIEmbeddings, create_kv_docstore, LocalFileStore, EMBED_MODEL
        from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
        chroma_path   = index_root / RAW_CHUNK_CHROMA_DIR
        docstore_path = index_root / "docstore"
        if not chroma_path.exists():
            raise SystemExit(
                f"Raw chunk Chroma index not found at {chroma_path}. "
                "Run: python eval/build_raw_chunk_index.py first."
            )
        vectorstore = Chroma(
            collection_name=args.raw_collection_name,
            embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
            persist_directory=str(chroma_path),
        )
        docstore = create_kv_docstore(LocalFileStore(str(docstore_path)))
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            id_key=ID_KEY,
            search_kwargs={"k": args.retrieve_top_k},
        )
        n_docs = len(vectorstore.get(include=[])["ids"])
        print(f"[retriever] raw-chunk index loaded — {n_docs} docs")
    else:
        from rag_utils import create_kv_docstore, LocalFileStore
        retriever = build_retriever(cache_root, args.collection_name, top_k=args.retrieve_top_k)
        if args.variant == "D":
            retriever.search_kwargs = {
                "k": args.retrieve_top_k,
                "filter": {"element_type": "text"},
            }
        if args.oracle:
            # For oracle mode we need direct docstore access
            docstore_path = cache_root / "docstore"
            docstore = create_kv_docstore(LocalFileStore(str(docstore_path)))

    # ── Load testset ──────────────────────────────────────────────────────────
    if not args.testset.exists():
        raise SystemExit(f"Testset not found: {args.testset}")
    with open(args.testset, encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]
    print(f"[testset] loaded {len(items)} questions from {args.testset.name}")

    if args.max_questions:
        items = items[:args.max_questions]
        print(f"[testset] limited to {len(items)} questions (--max_questions)")

    # ── Output path ───────────────────────────────────────────────────────────
    output_path = args.output
    if output_path is None:
        results_dir = _EVAL_DIR / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_oracle" if args.oracle else ""
        output_path = results_dir / f"variant_{args.variant}{suffix}.jsonl"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Judge client ──────────────────────────────────────────────────────────
    if args.dry_run or args.skip_judge:
        judge_client = None
    else:
        from openai import OpenAI
        judge_client = OpenAI()

    # ── Run config snapshot ───────────────────────────────────────────────────
    run_config = {
        "variant":             args.variant,
        "oracle":              args.oracle,
        "gen_model":           args.gen_model,
        "judge_model":         args.judge_model if not (args.dry_run or args.skip_judge) else None,
        "collection_name":     args.collection_name,
        "raw_collection_name": args.raw_collection_name,
        "retrieve_top_k":      args.retrieve_top_k,
        "rerank_top_k":        args.rerank_top_k,
        "testset_path":        str(args.testset),
        "cache_root":          str(args.cache_root),
        "raw_index_root":      str(index_root),
        "n_questions_in_testset": len(items),
        "dry_run":             args.dry_run,
        "skip_judge":          args.skip_judge,
        "timestamp_utc":       datetime.now(timezone.utc).isoformat(),
    }

    # ── Run eval ──────────────────────────────────────────────────────────────
    print(f"\n[eval] Variant {args.variant} | use_rerank={use_rerank} | "
          f"raw_idx={use_raw_idx} | oracle={args.oracle} | dry_run={args.dry_run}")
    print(f"[eval] output → {output_path}\n")

    results: list[dict] = []
    for idx, item in enumerate(items, 1):
        print(f"[{idx}/{len(items)}] qid={item['qid']} | {item['question'][:80]}")
        row = eval_one(
            item=item,
            retriever=retriever,
            cache_root=cache_root,
            variant=args.variant,
            judge_client=judge_client,
            judge_model=args.judge_model,
            gen_model=args.gen_model,
            use_rerank=use_rerank,
            rerank_top_k=args.rerank_top_k,
            k_values=HIT_K_VALUES,
            skip_judge=args.skip_judge,
            dry_run=args.dry_run,
            oracle=args.oracle,
            docstore=docstore,
        )
        results.append(row)

        # incremental write
        _write_jsonl(output_path, results)

        if not args.dry_run:
            s1_hit20  = row["stage1_hit"].get("20", False) if not args.oracle else "oracle"
            s2_hit3   = row["stage2_hit_at_3"]
            corr      = row.get("answer_correctness_score")
            faith     = row.get("faithfulness_score")
            faith_st  = row.get("faithfulness_status")
            print(f"  stage1_hit@20={s1_hit20} | stage2_hit@3={s2_hit3} | "
                  f"correctness={corr} | faith={faith}({faith_st}) | latency={row['latency_s']}s")

    # ── Write summary ─────────────────────────────────────────────────────────
    agg = aggregate(results, HIT_K_VALUES)
    summary_path = output_path.with_suffix(".summary.json")
    _write_summary(summary_path, args.variant, agg, run_config)

    gen = agg.get("generation", {})
    print("\n" + "=" * 60)
    print(f"Variant {args.variant} complete — {len(results)} questions")
    print(f"  Stage1 Hit@20 : {agg.get('stage1', {}).get('hit_rate_at_20')}")
    print(f"  Stage1 MRR    : {agg.get('stage1', {}).get('mrr')}")
    print(f"  Stage2 Hit@3  : {agg.get('stage2', {}).get('hit_rate_at_3')}")
    print(f"  Stage2 ΔMRR   : {agg.get('stage2', {}).get('delta_mrr')}")
    print(f"  Faithfulness  : {gen.get('faithfulness')}  (n={gen.get('n_faithfulness_valid')})")
    print(f"  Ans Correct.  : {gen.get('answer_correctness')}  (n={gen.get('n_answer_correctness_valid')})")
    print(f"  Ctx Recall    : {gen.get('context_recall')}  (n={gen.get('n_context_recall_valid')})")
    print(f"  Abstained     : {gen.get('n_abstained')}")
    print(f"  → {output_path}")
    print(f"  → {summary_path}")


if __name__ == "__main__":
    main()
