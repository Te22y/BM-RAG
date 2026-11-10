"""
chunk_loader.py
================

Loads the manifest and pickles of a multi-representation cache into the ChunkRecord
used by the evaluation scripts.

Shared by build_qa_testset.py and run_eval.py.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional


@dataclass
class ChunkRecord:
    # ── From the manifest ───────────────────────────────────────────────────
    doc_id: str                   # canonical id, used as gold_chunk_id
    element_type: str             # text / table / image
    source_pkl: str               # source_file in the manifest (hash.pkl)
    pickle_path: str              # pickle path relative to cache_root
    has_summary: bool

    # ── From the pickle (loaded lazily) ─────────────────────────────────────
    text: Optional[str] = None              # str(unstructured Element); base64 for an image
    paper_filename: Optional[str] = None    # pickle metadata filename (the real paper name)
    paper_directory: Optional[str] = None   # pickle metadata file_directory
    page_number: Optional[int] = None
    summary: Optional[str] = None           # read from summaries/{doc_id}.txt


def load_manifest(cache_root: Path) -> list[dict]:
    """Read manifest.json and return raw record dicts (filtered records included)."""
    manifest_path = cache_root / "manifest.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return data["records"]


def iter_eligible_records(
    cache_root: Path,
    require_summary: bool = True,
    element_types: tuple[str, ...] = ("text", "table", "image"),
) -> Iterator[dict]:
    """Yield manifest records that are unfiltered, of allowed types, with summary."""
    for r in load_manifest(cache_root):
        if r.get("filtered"):
            continue
        if r["element_type"] not in element_types:
            continue
        if require_summary and not r.get("has_summary"):
            continue
        yield r


def load_summary(cache_root: Path, doc_id: str) -> Optional[str]:
    p = cache_root / "summaries" / f"{doc_id}.txt"
    if not p.exists():
        return None
    return p.read_text(encoding="utf-8")


def hydrate_chunk(
    cache_root: Path,
    record_dict: dict,
    load_text: bool = True,
    load_summary_text: bool = True,
) -> ChunkRecord:
    """
    Materialize a ChunkRecord from a manifest dict.

    For text/table elements: text = str(unstructured element).
    For image elements: text = the base64 string (NOT the binary).

    Pickle metadata (page_number, paper_filename, paper_directory) is also pulled
    from the unstructured ElementMetadata when available.
    """
    chunk = ChunkRecord(
        doc_id=record_dict["doc_id"],
        element_type=record_dict["element_type"],
        source_pkl=record_dict["source_file"],
        pickle_path=record_dict["pickle_path"],
        has_summary=bool(record_dict.get("has_summary")),
    )

    if load_text:
        pkl_path = cache_root / record_dict["pickle_path"]
        with open(pkl_path, "rb") as f:
            obj = pickle.load(f)

        if chunk.element_type == "image":
            # image pickles are dicts with image_base64
            if isinstance(obj, dict):
                chunk.text = obj.get("image_base64", "")
                # image dicts may also carry metadata
                meta = obj.get("metadata") or {}
                if isinstance(meta, dict):
                    chunk.paper_filename = meta.get("filename")
                    chunk.paper_directory = meta.get("file_directory")
                    chunk.page_number = meta.get("page_number")
            else:
                chunk.text = ""
        else:
            chunk.text = str(obj)
            md = getattr(obj, "metadata", None)
            if md is not None and hasattr(md, "to_dict"):
                d = md.to_dict()
                chunk.paper_filename = d.get("filename")
                chunk.paper_directory = d.get("file_directory")
                pn = d.get("page_number")
                if isinstance(pn, int):
                    chunk.page_number = pn

    if load_summary_text:
        chunk.summary = load_summary(cache_root, chunk.doc_id)

    return chunk


def build_paper_filename_map(cache_root: Path) -> dict[str, str]:
    """
    Many table/image pickles do NOT carry filename in their unstructured metadata.
    To still report the originating paper, we scan the *text* chunks (which DO carry
    filename) and build `source_pkl_hash → paper_filename` map.

    The hash is the basename of source_file (e.g., "abcd1234.pkl"), shared across
    all elements (text/table/image) extracted from the same source PDF.
    """
    mapping: dict[str, str] = {}
    for r in load_manifest(cache_root):
        if r.get("filtered"):
            continue
        if r["element_type"] != "text":
            continue
        src = r["source_file"]
        if src in mapping:
            continue
        try:
            pkl_path = cache_root / r["pickle_path"]
            with open(pkl_path, "rb") as f:
                obj = pickle.load(f)
        except Exception:  # noqa: BLE001
            continue
        md = getattr(obj, "metadata", None)
        if md is None or not hasattr(md, "to_dict"):
            continue
        d = md.to_dict()
        fname = d.get("filename")
        if fname:
            mapping[src] = fname
    return mapping


def cache_snapshot_hash(cache_root: Path) -> str:
    """
    Lightweight reproducibility hash: SHA256 over (relative_path, size, mtime)
    of every file under cache_root. Used to freeze the testset against a specific
    snapshot of the chunks_cache so retrieval is reproducible.
    """
    import hashlib

    h = hashlib.sha256()
    files = sorted(cache_root.rglob("*"))
    for p in files:
        if not p.is_file():
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        rel = p.relative_to(cache_root).as_posix()
        h.update(f"{rel}|{st.st_size}|{int(st.st_mtime)}\n".encode("utf-8"))
    return h.hexdigest()
