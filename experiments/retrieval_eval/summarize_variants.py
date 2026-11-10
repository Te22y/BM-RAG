"""
Summarize A/B/C/D eval summary JSON files into one markdown table.

Usage:
  python eval/summarize_variants.py
  python eval/summarize_variants.py --results_dir eval/results --output eval/results/variants_summary.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_VARIANTS = ("A", "B", "C", "D")


def _get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for key in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _load_summary(results_dir: Path, variant: str) -> dict[str, Any]:
    path = results_dir / f"variant_{variant}.summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summary file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_markdown(results_dir: Path, variants: tuple[str, ...] = DEFAULT_VARIANTS) -> str:
    summaries = {variant: _load_summary(results_dir, variant) for variant in variants}

    metrics = [
        ("n", "aggregate.n_total"),
        ("errors", "aggregate.n_errors"),
        ("stage1_hit@20", "aggregate.stage1.hit_rate_at_20"),
        ("stage1_mrr", "aggregate.stage1.mrr"),
        ("stage2_hit@1", "aggregate.stage2.hit_rate_at_1"),
        ("stage2_hit@3", "aggregate.stage2.hit_rate_at_3"),
        ("stage2_mrr", "aggregate.stage2.mrr"),
        ("delta_mrr", "aggregate.stage2.delta_mrr"),
        ("faithfulness", "aggregate.generation.faithfulness"),
        ("answer_correctness", "aggregate.generation.answer_correctness"),
        ("ctx_precision@1", "aggregate.generation.context_precision_at_1"),
        ("ctx_precision@3", "aggregate.generation.context_precision_at_3"),
        ("ctx_recall", "aggregate.generation.context_recall"),
        ("abstained", "aggregate.generation.n_abstained"),
        ("latency_mean_s", "aggregate.latency.mean_s"),
    ]

    lines = [
        "# Variant Summary",
        "",
        "| Metric | A | B | C | D |",
        "|---|---:|---:|---:|---:|",
    ]

    for label, path in metrics:
        values = [_fmt(_get(summaries[variant], path)) for variant in variants]
        lines.append(f"| {label} | " + " | ".join(values) + " |")

    lines.extend([
        "",
        "## Variant Definitions",
        "",
        "- A: summary retrieval + summary rerank; context uses text raw, image/table summary.",
        "- B: summary retrieval without rerank; context uses text raw, image/table summary.",
        "- C: summary retrieval + raw-text rerank; image/table rerank and context use summary.",
        "- D: text-only summary retrieval + summary rerank; image/table are removed.",
    ])

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize eval variant results.")
    parser.add_argument("--results_dir", type=Path, default=Path(__file__).parent / "results")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    markdown = build_markdown(args.results_dir)
    print(markdown)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(markdown, encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
