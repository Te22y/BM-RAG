"""
report.py
=========

Summarizes the results of several variants into a Markdown comparison report.

Reads variant_A.summary.json, variant_B.summary.json and so on from results/ and
writes report_lab_be_v1.md with:
  - retrieval and generation metrics per variant, with valid sample counts
  - paired significance tests (Wilcoxon)
  - error sources (retrieval miss, rerank degradation, hallucination)
  - weak spots per question_type (needs the detailed .jsonl results)
  - failed cases (retrieval miss, hallucination)

Usage:
  # All variants (scans results/)
  python eval/report.py --results_dir eval/results --out eval/report_lab_be_v1.md

  # Selected variants only
  python eval/report.py --variants A B --results_dir eval/results --out eval/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_EVAL_DIR = Path(__file__).resolve().parent

VARIANT_LABELS = {
    "A": "A: summary + rerank (baseline)",
    "B": "B: summary + no-rerank",
    "C": "C: raw chunk + rerank",
    "D": "D: raw chunk + no-rerank",
    "E": "E: summary + rerank, summary context",
}


def load_summary(results_dir: Path, variant: str) -> Optional[dict]:
    p = results_dir / f"variant_{variant}.summary.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def load_detail(results_dir: Path, variant: str) -> list[dict]:
    p = results_dir / f"variant_{variant}.jsonl"
    if not p.exists():
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ══════════════════════════════════════════════════════════════════════════════
# Markdown table helpers
# ══════════════════════════════════════════════════════════════════════════════

def fmt(v, decimals: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def md_table(headers: list[str], rows: list[list]) -> str:
    col_widths = [max(len(str(h)), max((len(fmt(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]

    def pad(s, w):
        return str(s).ljust(w)

    header_row = "| " + " | ".join(pad(h, col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_row    = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_rows  = [
        "| " + " | ".join(pad(fmt(cell), col_widths[j]) for j, cell in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_row, sep_row] + data_rows)


# ══════════════════════════════════════════════════════════════════════════════
# Field name helpers (JSONL per-row)
# ══════════════════════════════════════════════════════════════════════════════

def _faith(r: dict) -> Optional[float]:
    """Faithfulness score (valid only)."""
    if r.get("faithfulness_status") == "valid":
        return r.get("faithfulness_score")
    return None


def _correctness(r: dict) -> Optional[float]:
    """Answer correctness score (valid only)."""
    if r.get("answer_correctness_status") == "valid":
        return r.get("answer_correctness_score")
    return None


def _recall(r: dict) -> Optional[float]:
    """Context recall score (valid only)."""
    if r.get("context_recall_status") == "valid":
        return r.get("context_recall_score")
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Section builders
# ══════════════════════════════════════════════════════════════════════════════

def section_overview(summaries: dict[str, dict]) -> str:
    lines = ["## 1. Overall Metric Comparison\n"]
    headers = [
        "Variant",
        "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR", "nDCG@10",
        "S2 Hit@1", "S2 Hit@3", "S2 nDCG@3", "ΔMRR",
        "Faithfulness (n)", "Ans.Correct (n)", "Ctx.Recall (n)",
    ]
    rows = []
    for v, data in sorted(summaries.items()):
        agg = data.get("aggregate", {})
        s1  = agg.get("stage1", {})
        s2  = agg.get("stage2", {})
        gen = agg.get("generation", {})

        def gen_cell(key, n_key):
            val = gen.get(key)
            n   = gen.get(n_key)
            if val is None:
                return "—"
            return f"{val:.4f} (n={n})"

        rows.append([
            VARIANT_LABELS.get(v, v),
            s1.get("hit_rate_at_1"),
            s1.get("hit_rate_at_3"),
            s1.get("hit_rate_at_5"),
            s1.get("hit_rate_at_10"),
            s1.get("hit_rate_at_20"),
            s1.get("mrr"),
            s1.get("ndcg10"),
            s2.get("hit_rate_at_1"),
            s2.get("hit_rate_at_3"),
            s2.get("ndcg3"),
            s2.get("delta_mrr"),
            gen_cell("faithfulness", "n_faithfulness_valid"),
            gen_cell("answer_correctness", "n_answer_correctness_valid"),
            gen_cell("context_recall", "n_context_recall_valid"),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append("> **ΔMRR** = Stage2 MRR − Stage1 MRR (positive = reranker improves ranking)")
    lines.append("> Generation metrics show mean over **valid** judge responses only (n = valid sample count).\n")
    return "\n".join(lines)


def section_ablation_delta(summaries: dict[str, dict]) -> str:
    lines = ["## 2. Ablation Impact Summary\n"]
    if "A" not in summaries:
        lines.append("_Variant A not available for comparison baseline._\n")
        return "\n".join(lines)

    base = summaries["A"]["aggregate"]
    metrics = [
        ("Stage1 Hit@20",  "stage1.hit_rate_at_20"),
        ("Stage2 Hit@3",   "stage2.hit_rate_at_3"),
        ("MRR (stage1)",   "stage1.mrr"),
        ("Δ MRR",          "stage2.delta_mrr"),
        ("Faithfulness",   "generation.faithfulness"),
        ("Ans.Correct",    "generation.answer_correctness"),
        ("Ctx.Recall",     "generation.context_recall"),
    ]

    def get_val(agg, dotpath):
        parts = dotpath.split(".")
        v = agg
        for p in parts:
            v = (v or {}).get(p)
        return v

    headers = ["Metric", "A (base)"] + [VARIANT_LABELS.get(v, v) for v in sorted(summaries) if v != "A"]
    rows = []
    for label, path in metrics:
        row = [label, get_val(base, path)]
        for v in sorted(summaries):
            if v == "A":
                continue
            val      = get_val(summaries[v]["aggregate"], path)
            base_val = get_val(base, path)
            if val is not None and base_val is not None:
                delta = val - base_val
                cell  = f"{val:.4f} ({'+' if delta >= 0 else ''}{delta:.4f})"
            else:
                cell = fmt(val)
            row.append(cell)
        rows.append(row)

    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append("> Values in parentheses show delta vs Variant A.\n")
    return "\n".join(lines)


def section_significance(details: dict[str, list[dict]]) -> str:
    """Paired Wilcoxon signed-rank test between variant A and others."""
    lines = ["## 3. Paired Significance Tests (Wilcoxon)\n"]

    try:
        from scipy.stats import wilcoxon  # type: ignore
    except ImportError:
        lines.append("_scipy not installed. Run `pip install scipy` to enable significance tests._\n")
        return "\n".join(lines)

    if "A" not in details or not details["A"]:
        lines.append("_Variant A detail data not available._\n")
        return "\n".join(lines)

    # Build qid → row index map for variant A
    a_by_qid = {r["qid"]: r for r in details["A"]}

    test_metrics = [
        ("Stage2 Hit@3",    "stage2_hit_at_3", None),
        ("Faithfulness",    "faithfulness_score", "faithfulness_status"),
        ("Ans.Correct",     "answer_correctness_score", "answer_correctness_status"),
        ("Ctx.Recall",      "context_recall_score", "context_recall_status"),
    ]

    headers = ["Comparison", "Metric", "n_pairs", "Δ mean", "p-value", "Significant?"]
    rows = []

    comparisons = [("A", v) for v in sorted(details) if v != "A"]

    for v_a, v_b in comparisons:
        b_by_qid = {r["qid"]: r for r in details.get(v_b, [])}
        shared_qids = [qid for qid in a_by_qid if qid in b_by_qid]

        for label, score_key, status_key in test_metrics:
            pairs = []
            for qid in shared_qids:
                ra = a_by_qid[qid]
                rb = b_by_qid[qid]
                # Filter by validity if status key provided
                if status_key:
                    if ra.get(status_key) != "valid" or rb.get(status_key) != "valid":
                        continue
                sa = ra.get(score_key)
                sb = rb.get(score_key)
                if sa is not None and sb is not None:
                    pairs.append((float(sa), float(sb)))

            if len(pairs) < 5:
                rows.append([
                    f"A vs {v_b}", label, len(pairs), "—", "—", "n<5 (skip)",
                ])
                continue

            scores_a = [p[0] for p in pairs]
            scores_b = [p[1] for p in pairs]
            delta_mean = round(sum(scores_b) / len(scores_b) - sum(scores_a) / len(scores_a), 4)

            # Wilcoxon requires at least one non-zero difference
            diffs = [b - a for a, b in zip(scores_a, scores_b)]
            if all(d == 0 for d in diffs):
                rows.append([
                    f"A vs {v_b}", label, len(pairs), delta_mean, "1.0000", "No (all equal)",
                ])
                continue

            try:
                _, p = wilcoxon(scores_a, scores_b, alternative="two-sided", zero_method="wilcox")
                sig = "Yes (p<0.05)" if p < 0.05 else "No"
                rows.append([
                    f"A vs {v_b}", label, len(pairs), delta_mean, round(p, 4), sig,
                ])
            except Exception as e:
                rows.append([f"A vs {v_b}", label, len(pairs), delta_mean, f"err:{e}", "—"])

    if rows:
        lines.append(md_table(headers, rows))
    else:
        lines.append("_No variant pairs available for comparison._")
    lines.append("")
    lines.append("> Wilcoxon signed-rank test (two-sided). Δ mean = B − A (positive = B better).\n")
    return "\n".join(lines)


def section_error_decomposition(details: dict[str, list[dict]]) -> str:
    """Decompose errors by failure stage for each variant."""
    lines = ["## 4. Error Decomposition by Stage\n"]

    headers = ["Variant", "n", "Retrieval Miss", "Rerank Degradation",
               "Ctx Insufficient", "Hallucination", "Judge Invalid"]
    rows = []

    for v, recs in sorted(details.items()):
        n = len(recs)
        if n == 0:
            continue

        # retrieval_miss: gold not in stage1 top-20
        retrieval_miss = sum(
            1 for r in recs
            if not r.get("stage1_hit", {}).get("20", False) and not r.get("oracle")
        )
        # rerank_degradation: gold in stage1 top-3 but NOT in stage2
        rerank_deg = sum(
            1 for r in recs
            if r.get("stage1_hit", {}).get("3", False) and not r.get("stage2_hit_at_3", True)
        )
        # context_insufficient: gold in stage2 but faithfulness < 0.5
        ctx_insuf = sum(
            1 for r in recs
            if r.get("stage2_hit_at_3") and r.get("faithfulness_status") == "valid"
            and (r.get("faithfulness_score") or 1.0) < 0.5
        )
        # hallucination: faithfulness < 0.5 (regardless of retrieval)
        halluc = sum(
            1 for r in recs
            if r.get("faithfulness_status") == "valid"
            and (r.get("faithfulness_score") or 1.0) < 0.5
        )
        # judge_invalid: any judge call failed
        judge_invalid = sum(
            1 for r in recs
            if r.get("judge_status") in ("failed", "partial")
        )

        rows.append([
            VARIANT_LABELS.get(v, v), n,
            f"{retrieval_miss} ({retrieval_miss/n:.0%})",
            f"{rerank_deg} ({rerank_deg/n:.0%})",
            f"{ctx_insuf} ({ctx_insuf/n:.0%})",
            f"{halluc} ({halluc/n:.0%})",
            f"{judge_invalid} ({judge_invalid/n:.0%})",
        ])

    lines.append(md_table(headers, rows))
    lines.append("")
    lines.append("> **Retrieval Miss**: gold chunk not in Stage1 top-20.\n"
                 "> **Rerank Degradation**: gold in Stage1 top-3 but dropped after rerank.\n"
                 "> **Ctx Insufficient**: gold in Stage2 context but faithfulness < 0.5.\n"
                 "> **Hallucination**: faithfulness < 0.5 (all cases).\n"
                 "> **Judge Invalid**: at least one judge call returned api_failed or parse_failed.\n")
    return "\n".join(lines)


def section_by_qtype(details: dict[str, list[dict]]) -> str:
    lines = ["## 5. Breakdown by question_type (Variant A)\n"]
    detail_a = details.get("A", [])
    if not detail_a:
        lines.append("_No detail data for variant A_\n")
        return "\n".join(lines)

    by_type: dict[str, list[dict]] = defaultdict(list)
    for row in detail_a:
        qt = row.get("question_type") or "unknown"
        by_type[qt].append(row)

    headers = ["question_type", "n", "Stage1 Hit@20", "Stage2 Hit@3",
               "Faithfulness", "Ans.Correct", "Abstained"]
    rows = []
    for qt, recs in sorted(by_type.items()):
        n = len(recs)

        def avg_valid(score_key, status_key, _recs=recs):
            vals = [r[score_key] for r in _recs if r.get(status_key) == "valid" and r.get(score_key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        hit20_vals = [r["stage1_hit"].get("20", False) for r in recs if isinstance(r.get("stage1_hit"), dict)]
        hit20 = round(sum(hit20_vals) / len(hit20_vals), 4) if hit20_vals else None
        hit3_vals  = [float(r["stage2_hit_at_3"]) for r in recs if r.get("stage2_hit_at_3") is not None]
        hit3 = round(sum(hit3_vals) / len(hit3_vals), 4) if hit3_vals else None

        rows.append([
            qt, n, hit20, hit3,
            avg_valid("faithfulness_score", "faithfulness_status"),
            avg_valid("answer_correctness_score", "answer_correctness_status"),
            sum(1 for r in recs if r.get("abstained")),
        ])

    lines.append(md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def section_by_category(details: dict[str, list[dict]]) -> str:
    lines = ["## 6. Breakdown by category (Variant A)\n"]
    detail_a = details.get("A", [])
    if not detail_a:
        lines.append("_No detail data for variant A_\n")
        return "\n".join(lines)

    by_cat: dict[str, list[dict]] = defaultdict(list)
    for row in detail_a:
        cat = row.get("category") or "unknown"
        by_cat[cat].append(row)

    headers = ["category", "n", "Stage2 Hit@3", "Ans.Correct", "Faithfulness"]
    rows = []
    for cat, recs in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        n = len(recs)

        def avg_valid(score_key, status_key, _recs=recs):
            vals = [r[score_key] for r in _recs if r.get(status_key) == "valid" and r.get(score_key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        hit3_vals = [float(r["stage2_hit_at_3"]) for r in recs if r.get("stage2_hit_at_3") is not None]
        hit3 = round(sum(hit3_vals) / len(hit3_vals), 4) if hit3_vals else None

        rows.append([
            cat, n, hit3,
            avg_valid("answer_correctness_score", "answer_correctness_status"),
            avg_valid("faithfulness_score", "faithfulness_status"),
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


def section_failure_cases(details: dict[str, list[dict]], n_cases: int = 5) -> str:
    lines = ["## 7. Failure Cases (Variant A)\n"]
    detail_a = details.get("A", [])
    if not detail_a:
        lines.append("_No detail data for variant A_\n")
        return "\n".join(lines)

    # Retrieval failures: gold chunk not in top-20
    retrieval_fails = [
        r for r in detail_a
        if not r.get("stage1_hit", {}).get("20", False) and not r.get("error") and not r.get("oracle")
    ]
    lines.append(f"### 7.1 Retrieval Failures (gold not in Stage1 top-20) — "
                 f"{len(retrieval_fails)} / {len(detail_a)}\n")
    for r in retrieval_fails[:n_cases]:
        lines.append(f"**{r['qid']}** | {r.get('question_type') or '?'} | {r.get('category') or '?'}")
        lines.append(f"> Q: {r['question']}")
        lines.append(f"> Gold: {r['gold_answer']}")
        lines.append(f"> Gold chunk: `{r['gold_chunk_id']}`")
        lines.append("")

    # Hallucination: faithfulness < 0.5 (valid judge only)
    halluc_fails = sorted(
        [r for r in detail_a
         if r.get("faithfulness_status") == "valid"
         and r.get("faithfulness_score") is not None
         and r["faithfulness_score"] < 0.5],
        key=lambda x: x["faithfulness_score"]
    )
    lines.append(f"### 7.2 Hallucination Cases (faithfulness < 0.5, valid judge) — "
                 f"{len(halluc_fails)} / {len(detail_a)}\n")
    for r in halluc_fails[:n_cases]:
        faith = r.get("faithfulness_score", 0)
        lines.append(f"**{r['qid']}** | faithfulness={faith:.2f} | gold_found={r.get('gold_found_stage')}")
        lines.append(f"> Q: {r['question']}")
        lines.append(f"> Generated: {r.get('generated_answer','')[:200]}")
        lines.append(f"> Gold: {r['gold_answer']}")
        lines.append(f"> Reasoning: {r.get('faithfulness_reasoning', '')[:200]}")
        lines.append("")

    # Answer correctness = 0 (valid judge only)
    wrong = [
        r for r in detail_a
        if r.get("answer_correctness_status") == "valid"
        and r.get("answer_correctness_score") == 0.0
        and not r.get("error")
    ]
    lines.append(f"### 7.3 Fully Wrong Answers (correctness=0, valid judge) — "
                 f"{len(wrong)} / {len(detail_a)}\n")
    for r in wrong[:n_cases]:
        lines.append(f"**{r['qid']}** | {r.get('question_type','?')}")
        lines.append(f"> Q: {r['question']}")
        lines.append(f"> Generated: {r.get('generated_answer','')[:200]}")
        lines.append(f"> Gold: {r['gold_answer']}")
        lines.append("")

    # Judge failures (api_failed or parse_failed)
    judge_fails = [
        r for r in detail_a
        if r.get("judge_status") in ("failed", "partial")
    ]
    lines.append(f"### 7.4 Judge API / Parse Failures — {len(judge_fails)} / {len(detail_a)}\n")
    for r in judge_fails[:n_cases]:
        lines.append(f"**{r['qid']}** | faith={r.get('faithfulness_status')} "
                     f"| corr={r.get('answer_correctness_status')} "
                     f"| recall={r.get('context_recall_status')}")
        lines.append("")

    # Abstentions
    abstained = [r for r in detail_a if r.get("abstained")]
    lines.append(f"### 7.5 Abstained Answers — {len(abstained)} / {len(detail_a)}\n")
    for r in abstained[:n_cases]:
        lines.append(f"**{r['qid']}** | gold_found={r.get('gold_found_stage')}")
        lines.append(f"> Generated: {r.get('generated_answer','')[:200]}")
        lines.append("")

    return "\n".join(lines)


def section_cost(summaries: dict[str, dict]) -> str:
    lines = ["## 8. Latency & Run Config\n"]
    headers = ["Variant", "n_total", "n_errors", "Mean Latency (s)",
               "Gen Model", "Judge Model", "Timestamp"]
    rows = []
    for v, data in sorted(summaries.items()):
        agg = data.get("aggregate", {})
        cfg = data.get("run_config", {})
        rows.append([
            VARIANT_LABELS.get(v, v),
            agg.get("n_total") or agg.get("n_questions"),
            agg.get("n_errors"),
            agg.get("latency", {}).get("mean_s"),
            cfg.get("gen_model", "—"),
            cfg.get("judge_model", "—"),
            (cfg.get("timestamp_utc") or "—")[:19],
        ])
    lines.append(md_table(headers, rows))
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def build_report(
    results_dir: Path,
    out_path: Path,
    variants: Optional[list[str]] = None,
) -> None:
    available_variants = variants or ["A", "B", "C", "D", "E"]

    summaries: dict[str, dict] = {}
    details:   dict[str, list[dict]] = {}

    for v in available_variants:
        s = load_summary(results_dir, v)
        if s:
            summaries[v] = s
            details[v]   = load_detail(results_dir, v)
            print(f"[load] variant {v}: {len(details[v])} questions")
        else:
            print(f"[skip] variant {v}: no summary found in {results_dir}")

    if not summaries:
        raise SystemExit(f"No variant summaries found in {results_dir}")

    sections = [
        "# RAG Evaluation Report — LAB_B&E2025\n",
        f"_Generated from `{results_dir}`_\n",
        section_overview(summaries),
        section_ablation_delta(summaries),
        section_significance(details),
        section_error_decomposition(details),
        section_by_qtype(details),
        section_by_category(details),
        section_failure_cases(details),
        section_cost(summaries),
    ]

    report = "\n\n".join(sections)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\n[done] report written to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results_dir", type=Path,
                        default=_EVAL_DIR / "results",
                        help="Directory containing variant_*.summary.json files.")
    parser.add_argument("--out", type=Path,
                        default=_EVAL_DIR / "report_lab_be_v1.md",
                        help="Output markdown path.")
    parser.add_argument("--variants", nargs="+", default=None,
                        choices=["A", "B", "C", "D", "E"],
                        help="Variants to include (default: all found in results_dir).")
    args = parser.parse_args()

    build_report(
        results_dir=args.results_dir,
        out_path=args.out,
        variants=args.variants,
    )


if __name__ == "__main__":
    main()
