"""
build_raw_chunk_index.py
========================

Preparation for ablation variants C and D.

Builds a raw-chunk Chroma index alongside the summary index:
  - embedded text = str(element), the raw chunk text
  - not the LLM summaries the main index embeds

The index goes to `chroma_db_raw/` under the same cache_root, leaving chroma_db/ alone.
The existing `docstore/` is reused; run_eval.py reads raw documents from it by doc_id.

Usage:
  python eval/build_raw_chunk_index.py \
      --cache_root "D:/.../multirepresentation_chunks_cache_LAB_B&E2025_chksize_6k_cleaned" \
      --collection_name multi_modal_rag_raw \
      --batch_size 200

Notes:
  - Run once, like the main index build.
  - Images are not embedded (their raw text is base64, which means nothing),
      unlike the summary index, where an image summary carries meaning.
  - Resumable: doc_ids already in chroma_db_raw are skipped.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

_EVAL_DIR  = Path(__file__).resolve().parent
sys.path.insert(0, str(_EVAL_DIR))

from rag_utils import (  # noqa: E402
    ID_KEY, EMBED_MODEL, _load_env,
    Chroma, OpenAIEmbeddings,
)
from langchain_core.documents import Document

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bmrag.config import get_paths  # noqa: E402


RAW_CHROMA_DIR = "chroma_db_raw"

DEFAULT_CACHE = get_paths().data_dir / "lab_be_index"


def load_manifest(cache_root: Path) -> list[dict]:
    p = cache_root / "manifest.json"
    return json.loads(p.read_text(encoding="utf-8"))["records"]


def get_existing_ids(vectorstore: Chroma) -> set[str]:
    result = vectorstore.get(include=["metadatas"])
    return {
        m[ID_KEY]
        for m in result["metadatas"]
        if m and ID_KEY in m
    }


def add_in_batches(vectorstore: Chroma, docs: list[Document], batch_size: int) -> None:
    total = len(docs)
    for i in range(0, total, batch_size):
        batch = docs[i: i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  embedded {min(i + batch_size, total)}/{total} …")


def build_raw_index(
    cache_root: Path,
    collection_name: str,
    batch_size: int,
) -> None:
    _load_env()

    chroma_path = cache_root / RAW_CHROMA_DIR
    chroma_path.mkdir(parents=True, exist_ok=True)

    print(f"[init] cache_root      = {cache_root}")
    print(f"[init] chroma_db_raw   = {chroma_path}")
    print(f"[init] collection_name = {collection_name}")
    print(f"[init] embed_model     = {EMBED_MODEL}")

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=str(chroma_path),
    )
    existing_ids = get_existing_ids(vectorstore)
    print(f"[init] existing docs in chroma_db_raw: {len(existing_ids)}")

    records = load_manifest(cache_root)
    print(f"[manifest] total records: {len(records)}")

    to_embed: list[Document] = []
    skipped_filtered = 0
    skipped_existing = 0
    skipped_image    = 0
    skipped_no_text  = 0

    for record in records:
        if record.get("filtered"):
            skipped_filtered += 1
            continue
        if record["doc_id"] in existing_ids:
            skipped_existing += 1
            continue
        if record["element_type"] == "image":
            # Raw base64 has no semantic value for embedding
            skipped_image += 1
            continue

        pkl_path = cache_root / record["pickle_path"]
        try:
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
            raw_text = str(obj)
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] failed to load {record['doc_id']}: {e}")
            skipped_no_text += 1
            continue

        if not raw_text.strip():
            skipped_no_text += 1
            continue

        # Truncate extremely long chunks to stay within embedding model limits
        # text-embedding-3-small: 8191 tokens ≈ ~32K chars for English text
        if len(raw_text) > 30_000:
            raw_text = raw_text[:30_000]

        to_embed.append(Document(
            page_content=raw_text,
            metadata={
                ID_KEY:          record["doc_id"],
                "element_type":  record["element_type"],
                "source_file":   record["source_file"],
            },
        ))

    print(f"\n[sample] skipped: filtered={skipped_filtered}, existing={skipped_existing}, "
          f"image={skipped_image}, no_text={skipped_no_text}")
    print(f"[sample] to embed: {len(to_embed)}")

    if not to_embed:
        print("[done] Nothing new to embed — raw chunk index is up to date.")
        return

    print(f"\n[embed] embedding {len(to_embed)} raw chunks in batches of {batch_size} …")
    add_in_batches(vectorstore, to_embed, batch_size)

    final_count = len(get_existing_ids(vectorstore))
    print(f"\n[done] chroma_db_raw now contains {final_count} docs.")
    print(f"[done] index path: {chroma_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache_root", type=Path, default=DEFAULT_CACHE,
                        help="multirepresentation_chunks_cache_* directory.")
    parser.add_argument("--collection_name", type=str, default="multi_modal_rag_raw",
                        help="Chroma collection name for the raw chunk index.")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Number of docs to embed per batch (default: 200).")
    args = parser.parse_args()

    build_raw_index(
        cache_root=args.cache_root,
        collection_name=args.collection_name,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
