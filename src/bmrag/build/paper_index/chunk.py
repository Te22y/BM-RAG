"""Paper index stage "chunk": partition each PDF with unstructured and split it into title-based chunks.

Output: paper_chunks/<md5 of the file name>.pkl, a list of CompositeElement. The file name hash
also starts every element id of the paper index, so it is kept even though a content hash would
notice a replaced file. A cached PDF is skipped unless --force is given.

Needs unstructured[pdf] with torch, and Poppler and Tesseract on PATH for the hi_res strategy;
without them hi_res fails and the next strategy (fast, no tables or images) is used.
"""

from __future__ import annotations

import hashlib
import logging
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from bmrag.build.profiles import ChunkingSettings
from bmrag.build.stages import BuildContext

log = logging.getLogger(__name__)


def cache_name(pdf_name: str) -> str:
    return f"{hashlib.md5(pdf_name.encode('utf-8')).hexdigest()}.pkl"


def partition(pdf_path: Path, settings: ChunkingSettings) -> tuple[list[Any], str]:
    """Chunks of one PDF and the name of the strategy that produced them."""
    from unstructured.partition.pdf import partition_pdf

    for name, options in settings.strategies:
        try:
            log.info("    Trying: %s", name)
            return partition_pdf(filename=str(pdf_path), **options, **settings.partition_options()), name
        except Exception as exc:
            log.warning("    %s failed: %s: %s", name, type(exc).__name__, exc)
    raise RuntimeError("All partition strategies failed")


def _chunk_one(pdf_path: Path, out_path: Path, settings: ChunkingSettings) -> str:
    """Worker process: partition one PDF and write its cache file. Returns the strategy used."""
    chunks, strategy = partition(pdf_path, settings)
    with out_path.open("wb") as f:
        pickle.dump(chunks, f)
    return strategy


def run(ctx: BuildContext) -> dict[str, Any]:
    settings = ctx.profile.chunking
    out_dir = ctx.layout.paper_chunks
    out_dir.mkdir(parents=True, exist_ok=True)
    pdfs = ctx.selected_pdfs()
    jobs = {pdf: out_dir / cache_name(pdf.name) for pdf in pdfs}
    todo = {pdf: out for pdf, out in jobs.items() if ctx.force or not out.exists()}
    log.info(
        "Chunking %d of %d PDF(s) with %d worker(s); %d cached",
        len(todo),
        len(pdfs),
        ctx.workers,
        len(pdfs) - len(todo),
    )

    # The layout models use every core per process unless told otherwise.
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable] = str(settings.math_threads_per_worker)
    os.environ["KMP_BLOCKTIME"] = "0"

    strategies: dict[str, str] = {}
    failed: dict[str, str] = {}
    with ProcessPoolExecutor(max_workers=max(1, ctx.workers)) as pool:
        futures = {pool.submit(_chunk_one, pdf, out, settings): pdf for pdf, out in todo.items()}
        for done, future in enumerate(as_completed(futures), 1):
            pdf = futures[future]
            try:
                strategies[pdf.name] = future.result()
                log.info("[%d/%d] %s: %s", done, len(todo), strategies[pdf.name], pdf.name)
            except Exception as exc:
                failed[pdf.name] = str(exc)
                log.error("[%d/%d] FAILED %s: %s", done, len(todo), pdf.name, exc)

    fallbacks = {name: s for name, s in strategies.items() if s != settings.strategies[0][0]}
    if fallbacks:
        ctx.record.warn(f"chunk: {len(fallbacks)} PDF(s) fell back from {settings.strategies[0][0]}: {fallbacks}")
    if failed:
        ctx.fail_or_warn(f"chunk: {len(failed)} PDF(s) failed: {failed}")
    return {
        "pdfs": len(pdfs),
        "chunked": len(strategies),
        "cached": len(pdfs) - len(todo),
        "failed": len(failed),
        "strategies": strategies,
    }
