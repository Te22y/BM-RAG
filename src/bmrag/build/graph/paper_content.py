"""What the extraction model sees of a PDF: its text, its tables as Markdown and some pages as images.

PyMuPDF reads the text and renders the pages; pdfplumber finds the tables. A page is sent as an
image when its text mentions a figure, a wall or a layer (ExtractionSettings.image_page_keywords),
because wall sections are usually only drawn, not described.
"""

from __future__ import annotations

import base64
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from bmrag.build.profiles import ExtractionSettings

log = logging.getLogger(__name__)

REFERENCES_HEADING = re.compile(r"\n\s*(?:References|Bibliography|Works Cited)\s*\n", re.IGNORECASE)


@dataclass(frozen=True)
class PaperContent:
    text: str
    tables_markdown: str
    page_images: list[tuple[int, str]]  # (1-based page number, base64 PNG)


def read_paper(pdf_path: Path, settings: ExtractionSettings) -> PaperContent:
    text, image_pages = extract_text(pdf_path, settings)
    return PaperContent(text, extract_tables_markdown(pdf_path), render_pages(pdf_path, image_pages, settings))


def extract_text(pdf_path: Path, settings: ExtractionSettings) -> tuple[str, list[int]]:
    """Full text with page markers, and the 0-based pages to send as images.

    A text over the character limit loses everything from its references heading on, or is cut
    at the limit when there is no such heading.
    """
    import pymupdf

    pages_text = []
    image_pages = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i, page in enumerate(doc):
            text = page.get_text("text")
            pages_text.append(f"\n\n---PAGE {i + 1}---\n\n{text}")
            if any(keyword in text.lower() for keyword in settings.image_page_keywords):
                image_pages.append(i)
    full_text = "".join(pages_text)

    if len(full_text) > settings.text_char_limit:
        heading = REFERENCES_HEADING.search(full_text)
        if heading:
            full_text = full_text[: heading.start()] + "\n\n[References section truncated]\n"
        else:
            full_text = full_text[: settings.text_char_limit] + "\n\n[Text truncated at character limit]\n"
    return full_text, image_pages


def extract_tables_markdown(pdf_path: Path) -> str:
    """Every table pdfplumber finds, as Markdown with a page marker; empty if extraction fails."""
    import pdfplumber

    sections = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages):
                for t_idx, table in enumerate(page.extract_tables() or []):
                    if not table or not table[0]:
                        continue
                    rows = []
                    for row_idx, row in enumerate(table):
                        cells = [str(c).strip() if c is not None else "" for c in row]
                        rows.append("| " + " | ".join(cells) + " |")
                        if row_idx == 0:
                            rows.append("| " + " | ".join(["---"] * len(cells)) + " |")
                    sections.append(f"\n\n---TABLE {t_idx + 1} ON PAGE {i + 1}---\n" + "\n".join(rows))
    except Exception as exc:
        log.warning("pdfplumber table extraction failed: %s", exc)
    return "".join(sections)


def render_pages(pdf_path: Path, page_indices: list[int], settings: ExtractionSettings) -> list[tuple[int, str]]:
    """Render the given 0-based pages as base64 PNG at settings.image_scale."""
    import pymupdf

    matrix = pymupdf.Matrix(settings.image_scale, settings.image_scale)
    results = []
    with pymupdf.open(str(pdf_path)) as doc:
        for i in page_indices:
            if i >= len(doc):
                continue
            png = doc[i].get_pixmap(matrix=matrix).tobytes("png")
            results.append((i + 1, base64.standard_b64encode(png).decode("utf-8")))
    return results
