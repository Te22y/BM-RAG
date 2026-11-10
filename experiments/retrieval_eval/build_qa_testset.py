"""
build_qa_testset.py
===================

Phase 0 of the LAB_B&E2025 RAG eval methodology:
Stratified sampling of chunks_cache, QA pairs from GPT-4o, reverse self-check with gpt-5.2, JSONL output.

Test set schema (one row per line):
  {
    "qid": "lab_be_q_0001",
    "question": "...",
    "gold_answer": "...",
    "evidence_span": "...",
    "gold_chunk_id": "<doc_id from manifest>",
    "gold_paper": "<paper_filename>",
    "gold_page": 12,
    "element_type": "text",
    "question_type": "numeric",
    "category": "thermal_insulation",
    "verified": false,
    "reverse_check": {{"asked": true, "candidate": "...", "judged_match": false}}
  }

A sidecar file `<output>.meta.json` records cache_snapshot_hash, the random seed,
model names and other frozen details, so the test set can be reproduced.

Usage:
  python eval/build_qa_testset.py \
      --chunks_cache data/lab_be_index \
      --n_samples 120 \
      --gen_model gpt-4o \
      --judge_model gpt-5.2 \
      --output eval/qa_testset_lab_be_v1.jsonl

Smoke test (no LLM calls):
  python eval/build_qa_testset.py --chunks_cache <path> --n_samples 5 --dry_run
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Force utf-8 stdout (the Windows cp949 default fails on unicode)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# Sibling modules resolve however this script is started
_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))

from chunk_loader import (  # noqa: E402
    build_paper_filename_map,
    cache_snapshot_hash,
    hydrate_chunk,
    iter_eligible_records,
)
from qa_gen_prompts import (  # noqa: E402
    ANSWER_MATCH_PROMPT,
    QA_GEN_PROMPT,
    QA_GEN_SUMMARY_PROMPT,
    REVERSE_ANSWER_PROMPT,
)


# ══════════════════════════════════════════════════════════════════════════════
# Sampling
# ══════════════════════════════════════════════════════════════════════════════

# text : table : image ≈ 6 : 3 : 1
ELEMENT_TYPE_RATIOS = {"text": 0.5, "table": 0.4, "image": 0.1}
MAX_CHUNKS_PER_PAPER = 2


def stratified_sample(
    cache_root: Path,
    n_samples: int,
    seed: int,
    include_images: bool = True,
    exclude_doc_ids: set[str] | None = None,
) -> list[dict]:
    """
    Stratified sampling:
      - per paper: at most MAX_CHUNKS_PER_PAPER chunks
      - per element type: proportions from ELEMENT_TYPE_RATIOS
    Returns manifest record dicts (sorted by doc_id for stability, then shuffled).

    exclude_doc_ids: doc_ids already used in the existing question set; skipped to avoid repeats.
    """
    rng = random.Random(seed)
    exclude = exclude_doc_ids or set()

    types = ("text", "table", "image") if include_images else ("text", "table")
    by_type: dict[str, list[dict]] = {t: [] for t in types}
    for r in iter_eligible_records(cache_root, require_summary=True, element_types=types):
        if r["doc_id"] in exclude:
            continue
        by_type[r["element_type"]].append(r)

    print(f"[sample] eligible records by type: " +
          ", ".join(f"{t}={len(by_type[t])}" for t in types))

    # Target count per type
    targets = {}
    if include_images:
        for t, ratio in ELEMENT_TYPE_RATIOS.items():
            targets[t] = int(round(n_samples * ratio))
    else:
        # No images: text:table = 6:3, i.e. 2:1
        targets["text"] = int(round(n_samples * 2 / 3))
        targets["table"] = n_samples - targets["text"]

    # Adjust so the total is exactly n_samples
    diff = n_samples - sum(targets.values())
    if diff != 0:
        targets["text"] += diff

    print(f"[sample] target per type: {targets}")

    selected: list[dict] = []
    paper_counter: Counter = Counter()

    for t in types:
        pool = by_type[t]
        rng.shuffle(pool)
        # Group by source_pkl and take one per paper in turn, so no single document dominates
        by_paper: dict[str, list[dict]] = defaultdict(list)
        for r in pool:
            by_paper[r["source_file"]].append(r)
        # round-robin
        paper_keys = list(by_paper.keys())
        rng.shuffle(paper_keys)
        round_idx = 0
        picked_this_type = 0
        while picked_this_type < targets[t] and any(by_paper[k] for k in paper_keys):
            for k in paper_keys:
                if picked_this_type >= targets[t]:
                    break
                if not by_paper[k]:
                    continue
                if paper_counter[k] >= MAX_CHUNKS_PER_PAPER:
                    by_paper[k].clear()
                    continue
                rec = by_paper[k].pop()
                selected.append(rec)
                paper_counter[k] += 1
                picked_this_type += 1
            round_idx += 1
            if round_idx > n_samples * 4:
                break  # safety

        print(f"[sample] picked {picked_this_type}/{targets[t]} for type={t}")

    rng.shuffle(selected)
    return selected


# ══════════════════════════════════════════════════════════════════════════════
# OpenAI call wrapper
# ══════════════════════════════════════════════════════════════════════════════
def _make_client():
    """Lazy import openai so dry_run can run without the package."""
    # Load the repository's .env if present, so OPENAI_API_KEY is picked up
    try:
        from dotenv import load_dotenv  # type: ignore
        env_path = _THIS.parents[2] / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    from openai import OpenAI  # type: ignore
    return OpenAI()


def call_llm_json(
    client,
    model: str,
    prompt: str,
    *,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """
    Call chat.completions and return raw text. Handles a couple of transient errors.
    JSON parsing happens at the call site.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:  # noqa: BLE001
            last_err = e
            wait = 2 ** attempt
            print(f"  [warn] LLM call failed (attempt {attempt+1}/{max_retries}): {e}; "
                  f"sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"LLM call failed after {max_retries} attempts: {last_err}")


def parse_json_array(raw: str) -> list[dict]:
    """Strip optional ```json fences and parse a top-level JSON array."""
    s = raw.strip()
    if s.startswith("```"):
        # remove fence
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    if not s:
        return []
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        # try to extract the first [...] blob
        m = re.search(r"\[.*\]", s, re.DOTALL)
        if not m:
            return []
        data = json.loads(m.group(0))
    if not isinstance(data, list):
        return []
    return data


def parse_json_object(raw: str) -> dict:
    s = raw.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return {}
        return json.loads(m.group(0))


# ══════════════════════════════════════════════════════════════════════════════
# QA generation and reverse self-check
# ══════════════════════════════════════════════════════════════════════════════

REQUIRED_FIELDS = {"question", "gold_answer", "evidence_span", "question_type", "category"}
ALLOWED_QTYPES = {"factoid", "numeric", "comparison", "mechanism", "definition"}


def truncate_chunk_text(text: str, max_chars: int = 8000) -> str:
    """Cap chunk text length to stay within model context for QA gen."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated...]"


def question_leaks_answer(question: str, gold_answer: str) -> bool:
    """
    Crude leakage check: if the answer (numeric or short phrase) appears verbatim
    inside the question, the model just copied it — drop.
    """
    q = question.lower()
    a = gold_answer.lower().strip()
    if not a:
        return True
    # ignore short stop-words
    if len(a) <= 3:
        return False
    # exact substring leak
    if a in q:
        return True
    # numeric leak (e.g., "12.3 MPa" appears in question)
    nums_a = re.findall(r"\d+\.?\d*", a)
    nums_q = set(re.findall(r"\d+\.?\d*", q))
    if nums_a and all(n in nums_q for n in nums_a):
        return True
    return False


def generate_qa_for_chunk(client, gen_model: str, chunk) -> list[dict]:
    """Call GPT-4o on one chunk; return up to 2 validated QA dicts.

    Content source selection:
      - text   → always use chunk.text (original)
      - table  → use chunk.summary if available, otherwise fall back to chunk.text
      - image  → use chunk.summary; skip if no summary (base64 is not usable as text)
    """
    summary_available = bool(chunk.summary and chunk.summary.strip())

    if chunk.element_type == "image":
        if not summary_available:
            return []
        content_text = truncate_chunk_text(chunk.summary)
        prompt_template = QA_GEN_SUMMARY_PROMPT
        content_source = "summary"
    elif chunk.element_type == "table":
        if summary_available:
            content_text = truncate_chunk_text(chunk.summary)
            prompt_template = QA_GEN_SUMMARY_PROMPT
            content_source = "summary"
        else:
            content_text = truncate_chunk_text(chunk.text or "")
            prompt_template = QA_GEN_PROMPT
            content_source = "text"
    else:
        content_text = truncate_chunk_text(chunk.text or "")
        prompt_template = QA_GEN_PROMPT
        content_source = "text"

    if len(content_text.strip()) < 200:
        return []

    prompt = prompt_template.format(
        paper_filename=chunk.paper_filename or chunk.source_pkl,
        page_number=chunk.page_number if chunk.page_number is not None else "unknown",
        element_type=chunk.element_type,
        chunk_text=content_text,
    )

    raw = call_llm_json(client, gen_model, prompt, temperature=0.0)
    items = parse_json_array(raw)

    valid: list[dict] = []
    for item in items[:2]:
        if not isinstance(item, dict):
            continue
        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            continue
        if not all(isinstance(item[k], str) and item[k].strip() for k in REQUIRED_FIELDS):
            continue
        if item["question_type"] not in ALLOWED_QTYPES:
            # tolerant: keep, but normalize
            item["question_type"] = "factoid"
        if question_leaks_answer(item["question"], item["gold_answer"]):
            continue
        item["_content_source"] = content_source   # internal marker, used when writing the row
        valid.append(item)
    return valid


def reverse_self_check(client, judge_model: str, qa: dict) -> dict:
    """
    Ask judge_model the question with NO context, then ask it (a second time)
    whether its candidate answer matches the gold_answer. Returns a dict that goes
    into qa['reverse_check']. The caller drops items where judged_match=True.
    """
    # Step 1: try to answer without context
    raw_ans = call_llm_json(
        client, judge_model,
        REVERSE_ANSWER_PROMPT.format(question=qa["question"]),
        temperature=0.0,
    )
    candidate = (raw_ans or "").strip()

    # Quick short-circuit: explicit "I don't know"
    low = candidate.lower()
    if "i don't know" in low or "i do not know" in low or low == "":
        return {"asked": True, "candidate": candidate, "judged_match": False}

    # Step 2: judge match
    raw_judge = call_llm_json(
        client, judge_model,
        ANSWER_MATCH_PROMPT.format(
            question=qa["question"],
            gold_answer=qa["gold_answer"],
            candidate_answer=candidate,
        ),
        temperature=0.0,
    )
    judged = parse_json_object(raw_judge)
    judged_match = bool(judged.get("match")) if isinstance(judged, dict) else False
    return {
        "asked": True,
        "candidate": candidate,
        "judged_match": judged_match,
        "judge_reason": (judged or {}).get("reason", "") if isinstance(judged, dict) else "",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Main flow
# ══════════════════════════════════════════════════════════════════════════════
def _load_existing_jsonl(path: Path) -> tuple[list[dict], int]:
    """
    Read an existing JSONL question set; returns (rows, max_qid_number).
    qids look like "lab_be_q_NNNN"; the largest number is parsed so numbering can continue.
    """
    if not path.exists():
        return [], 0
    rows = []
    max_n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows.append(r)
            m = re.search(r"_(\d+)$", r.get("qid", ""))
            if m:
                max_n = max(max_n, int(m.group(1)))
    return rows, max_n


def build_testset(
    cache_root: Path,
    output_path: Path,
    n_samples: int,
    gen_model: str,
    judge_model: str,
    seed: int,
    dry_run: bool,
    skip_reverse_check: bool,
    include_images: bool,
    append: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[init] cache_root      = {cache_root}")
    print(f"[init] output          = {output_path}")
    print(f"[init] n_samples       = {n_samples}")
    print(f"[init] gen_model       = {gen_model}")
    print(f"[init] judge_model     = {judge_model}")
    print(f"[init] seed            = {seed}")
    print(f"[init] dry_run         = {dry_run}")
    print(f"[init] include_images  = {include_images}")
    print(f"[init] append          = {append}")

    if not cache_root.exists():
        raise SystemExit(f"chunks_cache not found: {cache_root}")

    # 0. Append mode: load the existing set, its used chunk ids and largest qid
    if append:
        existing_rows, qid_counter = _load_existing_jsonl(output_path)
        exclude_doc_ids = {r["gold_chunk_id"] for r in existing_rows}
        print(f"[append] loaded {len(existing_rows)} existing QA rows, "
              f"max qid={qid_counter}, excluding {len(exclude_doc_ids)} chunk ids")
    else:
        existing_rows, qid_counter = [], 0
        exclude_doc_ids = set()

    # 1. Sample (excluding used chunks when appending)
    sampled_records = stratified_sample(
        cache_root, n_samples, seed, include_images=include_images,
        exclude_doc_ids=exclude_doc_ids,
    )
    print(f"[sample] total selected: {len(sampled_records)}")

    # 1b. Map source_pkl to paper_filename, for tables and images without a filename
    print("[init] building source_pkl → paper_filename map (scanning text chunks) …")
    paper_map = build_paper_filename_map(cache_root)
    print(f"[init] paper_map covers {len(paper_map)} source pkls")

    # 2. Load and, unless a dry run, call the LLM
    client = None if dry_run else _make_client()

    qa_rows: list[dict] = list(existing_rows)   # existing rows when appending, otherwise empty
    stats: Counter = Counter()

    for idx, rec in enumerate(sampled_records, 1):
        try:
            chunk = hydrate_chunk(cache_root, rec, load_text=True, load_summary_text=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [skip] chunk {rec['doc_id']}: load failed ({e})")
            stats["chunk_load_failed"] += 1
            continue

        # Backfill paper_filename: tables and images rarely have one, so borrow from a sibling text
        if not chunk.paper_filename:
            chunk.paper_filename = paper_map.get(chunk.source_pkl)

        print(f"[{idx}/{len(sampled_records)}] {chunk.element_type:5s} | "
              f"{(chunk.paper_filename or chunk.source_pkl)[:70]} | p={chunk.page_number}")

        if dry_run:
            stats["dry_run_chunks"] += 1
            continue

        try:
            qas = generate_qa_for_chunk(client, gen_model, chunk)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] QA gen failed: {e}")
            stats["gen_failed"] += 1
            continue

        if not qas:
            stats["no_qa_generated"] += 1
            continue

        stats["raw_qa_generated"] += len(qas)

        for qa in qas:
            # reverse self-check
            if skip_reverse_check:
                rev = {"asked": False, "candidate": "", "judged_match": False}
            else:
                try:
                    rev = reverse_self_check(client, judge_model, qa)
                except Exception as e:  # noqa: BLE001
                    print(f"  [warn] reverse check failed: {e}")
                    rev = {"asked": False, "candidate": "", "judged_match": False,
                           "error": str(e)}

            if rev.get("judged_match"):
                stats["dropped_reverse_match"] += 1
                continue

            qid_counter += 1
            row = {
                "qid": f"lab_be_q_{qid_counter:04d}",
                "question": qa["question"].strip(),
                "gold_answer": qa["gold_answer"].strip(),
                "evidence_span": qa["evidence_span"].strip(),
                "gold_chunk_id": chunk.doc_id,
                "gold_paper": chunk.paper_filename or chunk.source_pkl,
                "gold_page": chunk.page_number,
                "element_type": chunk.element_type,
                "content_source": qa.pop("_content_source", "text"),
                "question_type": qa["question_type"],
                "category": qa["category"],
                "verified": False,
                "reverse_check": rev,
            }
            qa_rows.append(row)
            stats["kept"] += 1

        # Write incrementally so a crash in a long run loses nothing
        _write_jsonl(output_path, qa_rows)

    # 3. Write the output and the sidecar
    _write_jsonl(output_path, qa_rows)
    _write_meta(output_path, cache_root, gen_model, judge_model, seed, n_samples, stats)
    _write_stats_md(output_path, qa_rows, stats)

    print()
    print("=" * 60)
    print("Done.")
    print(f"  kept QA              : {stats.get('kept', 0)}")
    print(f"  raw QA generated     : {stats.get('raw_qa_generated', 0)}")
    print(f"  dropped reverse match: {stats.get('dropped_reverse_match', 0)}")
    print(f"  no QA generated      : {stats.get('no_qa_generated', 0)}")
    print(f"  gen failed           : {stats.get('gen_failed', 0)}")
    print(f"  chunk load failed    : {stats.get('chunk_load_failed', 0)}")
    print(f"  output               : {output_path}")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def _write_meta(
    output_path: Path,
    cache_root: Path,
    gen_model: str,
    judge_model: str,
    seed: int,
    n_samples: int,
    stats: Counter,
) -> None:
    meta_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    print(f"[meta] computing cache snapshot hash …")
    snap = cache_snapshot_hash(cache_root)
    meta = {
        "cache_root": str(cache_root),
        "cache_snapshot_sha256": snap,
        "gen_model": gen_model,
        "judge_model": judge_model,
        "seed": seed,
        "n_samples_requested": n_samples,
        "stats": dict(stats),
        "schema_version": 1,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[meta] wrote {meta_path}")


def _write_stats_md(output_path: Path, rows: list[dict], stats: Counter) -> None:
    md_path = output_path.with_suffix(".stats.md")
    by_paper = Counter(r["gold_paper"] for r in rows)
    by_type = Counter(r["element_type"] for r in rows)
    by_source = Counter(r.get("content_source", "text") for r in rows)
    by_qtype = Counter(r["question_type"] for r in rows)
    by_cat = Counter(r["category"] for r in rows)

    lines = [
        f"# QA testset stats — {output_path.name}",
        "",
        f"- Total kept QA: **{len(rows)}**",
        "",
        "## Pipeline counters",
        "",
    ]
    for k, v in sorted(stats.items()):
        lines.append(f"- {k}: {v}")

    lines += ["", "## By element_type", ""]
    for k, v in by_type.most_common():
        lines.append(f"- {k}: {v}")

    lines += ["", "## By content_source", ""]
    for k, v in by_source.most_common():
        lines.append(f"- {k}: {v}")

    lines += ["", "## By question_type", ""]
    for k, v in by_qtype.most_common():
        lines.append(f"- {k}: {v}")

    lines += ["", "## By category", ""]
    for k, v in by_cat.most_common():
        lines.append(f"- {k}: {v}")

    lines += ["", f"## By paper (top 30 of {len(by_paper)})", ""]
    for k, v in by_paper.most_common(30):
        lines.append(f"- {v} | {k}")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[stats] wrote {md_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--chunks_cache", type=Path, required=True,
                        help="Path to multirepresentation_chunks_cache_LAB_B&E2025_chksize_6k_cleaned/")
    parser.add_argument("--output", type=Path,
                        default=Path("eval/qa_testset_lab_be_v1.jsonl"),
                        help="Output JSONL path (relative to cwd).")
    parser.add_argument("--n_samples", type=int, default=120,
                        help="Number of chunks to sample (target).")
    parser.add_argument("--gen_model", type=str, default="gpt-4o",
                        help="QA generation model (default: gpt-4o).")
    parser.add_argument("--judge_model", type=str, default="gpt-5.2",
                        help="Reverse self-check / answer-match judge model.")
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--dry_run", action="store_true",
                        help="Skip all LLM calls; just sample chunks and print a summary.")
    parser.add_argument("--skip_reverse_check", action="store_true",
                        help="Generate QA but skip the gpt-5.2 reverse self-check filter.")
    parser.add_argument("--no_images", action="store_true",
                        help="Exclude image elements from sampling (recommended for v1).")
    parser.add_argument("--append", action="store_true",
                        help="Append new QA to an existing output JSONL instead of overwriting. "
                             "Already-used chunk ids are excluded from re-sampling. "
                             "Use a different --seed to avoid duplicate sampling.")
    args = parser.parse_args()

    build_testset(
        cache_root=args.chunks_cache,
        output_path=args.output,
        n_samples=args.n_samples,
        gen_model=args.gen_model,
        judge_model=args.judge_model,
        seed=args.seed,
        dry_run=args.dry_run,
        skip_reverse_check=args.skip_reverse_check,
        include_images=not args.no_images,
        append=args.append,
    )


if __name__ == "__main__":
    main()
