"""Paper index stage "clean": drop references and back matter from the cached chunks.

Rules, applied to the original elements inside each chunk:
- A Title naming an always-drop section (acknowledgements, appendix, ...) ends the useful text:
  as the first element of a chunk it drops the whole chunk, later in a chunk it truncates there.
- A short element (at most max_header_words words) naming such a section is treated as a header too.
- "References" counts only within the last end_threshold chunks, since the word also appears in
  normal headings earlier in a paper.
Everything after the first such point is dropped.

Output: paper_chunks_cleaned/<same name>.pkl. An existing output is kept unless --force is given.
"""

from __future__ import annotations

import copy
import logging
import pickle
import re
from typing import Any

from bmrag.build.profiles import ReferenceCleaningSettings
from bmrag.build.stages import BuildContext

log = logging.getLogger(__name__)


def matches_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    """Whether text contains a keyword as whole words: "appendix" does not match "appendixes".

    A longer heading that contains a keyword still matches ("data availability and discussion");
    for non-title elements the word limit in _is_short_header keeps body text from matching.
    """
    return any(re.search(r"\b" + re.escape(keyword) + r"\b", text) for keyword in keywords)


def _is_short_header(text: str, keywords: tuple[str, ...], settings: ReferenceCleaningSettings) -> bool:
    return len(text.split()) <= settings.max_header_words and matches_keyword(text.lower().strip(), keywords)


def filter_chunk(chunk: Any, check_references: bool, settings: ReferenceCleaningSettings) -> tuple[Any, bool]:
    """(kept chunk or None, whether to stop) for one CompositeElement."""
    from unstructured.documents.elements import CompositeElement, Title

    kept = []
    title_count = 0
    stop = False
    for element in chunk.metadata.orig_elements:
        if isinstance(element, Title):
            title_count += 1
            title = element.text.lower().strip()
            drop = matches_keyword(title, settings.always_drop_sections) or (
                check_references and matches_keyword(title, settings.end_only_sections)
            )
            if drop and title_count == 1:
                log.debug("     First element is a drop-section title %r: dropping the chunk", element.text[:50])
                return None, True
            if drop:
                log.debug("    Mid-chunk drop-section title %r: truncating", element.text[:50])
                stop = True
                break
            kept.append(element)
            continue
        if hasattr(element, "text") and (
            _is_short_header(element.text, settings.always_drop_sections, settings)
            or (check_references and _is_short_header(element.text, settings.end_only_sections, settings))
        ):
            log.debug("    Short drop-section header %r: truncating", element.text[:50])
            stop = True
            break
        kept.append(element)

    if not kept:
        return None, stop
    cleaned = CompositeElement(text="\n\n".join(e.text for e in kept if hasattr(e, "text")))
    cleaned.metadata = copy.deepcopy(chunk.metadata)  # the original chunk's metadata stays untouched
    cleaned.metadata.orig_elements = kept
    return cleaned, stop


def clean_chunks(chunks: list[Any], settings: ReferenceCleaningSettings) -> list[Any]:
    """The chunks up to the first drop point, each filtered by filter_chunk."""
    cleaned = []
    for i, chunk in enumerate(chunks):
        near_end = len(chunks) - i <= settings.end_threshold
        kept, stop = filter_chunk(chunk, check_references=near_end, settings=settings)
        if kept is not None:
            cleaned.append(kept)
        if stop:
            log.debug("  Chunk %d/%d stops the paper; %d chunks dropped", i + 1, len(chunks), len(chunks) - i - 1)
            break
    return cleaned


def run(ctx: BuildContext) -> dict[str, Any]:
    settings = ctx.profile.reference_cleaning
    source, target = ctx.layout.paper_chunks, ctx.layout.paper_chunks_cleaned
    if not source.is_dir():
        raise SystemExit(f"No chunks in {source}; run the chunk stage first")
    target.mkdir(parents=True, exist_ok=True)
    stats = {"files": 0, "cleaned": 0, "kept_existing": 0, "chunks_in": 0, "chunks_kept": 0}
    for path in sorted(source.glob("*.pkl")):
        stats["files"] += 1
        out = target / path.name
        if out.exists() and not ctx.force:
            stats["kept_existing"] += 1
            continue
        with path.open("rb") as f:
            chunks = pickle.load(f)
        cleaned = clean_chunks(chunks, settings)
        with out.open("wb") as f:
            pickle.dump(cleaned, f)
        stats["cleaned"] += 1
        stats["chunks_in"] += len(chunks)
        stats["chunks_kept"] += len(cleaned)
        log.info("%s: %d of %d chunks kept", path.name, len(cleaned), len(chunks))
    return stats
