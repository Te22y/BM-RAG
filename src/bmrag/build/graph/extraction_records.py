"""Parse the extraction model's reply and fill the provenance fields later stages rely on."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def parse_json_reply(raw: str) -> dict[str, Any]:
    """The JSON object in a reply, ignoring Markdown fences; a bare array is taken as the cases.

    Raises ValueError (json.JSONDecodeError is one) when there is no parseable JSON.
    """
    raw = re.sub(r"```(?:json)?\s*", "", raw).strip().strip("`").strip()
    object_start, object_end = raw.find("{"), raw.rfind("}")
    if object_start != -1 and object_end != -1:
        return json.loads(raw[object_start : object_end + 1])
    array_start, array_end = raw.find("["), raw.rfind("]")
    if array_start != -1 and array_end != -1:
        return {"cases": json.loads(raw[array_start : array_end + 1])}
    raise ValueError("No JSON object or array found in response")


def paper_slug(pdf_path: Path) -> str:
    """File-name-safe id of a paper: its file stem with other characters replaced, at most 60 long."""
    return re.sub(r"[^\w\-]", "_", pdf_path.stem)[:60]


def _paper_type_from_hint(hint: str | None) -> str:
    value = (hint or "").lower()
    if value in {"review", "meta_analysis"}:
        return "review_paper"
    if value == "hybrid":
        return "hybrid_paper"
    return "research_paper"


def normalize_paper_record(record: Any, pdf_path: Path, paper_type_hint: str, model: str) -> dict[str, Any]:
    """Fill paper metadata, case ids and provenance that the model left out. Mutates and returns record."""
    if not isinstance(record, dict):
        record = {"cases": []}
    cases = record.get("cases")
    if not isinstance(cases, list):
        cases = []
    metadata = record.get("paper_metadata") if isinstance(record.get("paper_metadata"), dict) else {}
    record.setdefault("paper_type", _paper_type_from_hint(paper_type_hint))
    record.setdefault("classification_reason", "paper_type inferred by Step0 prompt or filename/text heuristic")
    metadata.setdefault("paper_id", paper_slug(pdf_path))
    metadata.setdefault("paper_title", cases[0].get("paper_title") if cases else pdf_path.stem)
    metadata.setdefault("source_file", pdf_path.name)
    record["paper_metadata"] = metadata
    record.setdefault("paper_level_summary", {})
    record.setdefault("kg_ready_rules", [])
    record.setdefault("extraction_warnings", [])
    for index, case in enumerate(cases, start=1):
        case.setdefault("source_file", pdf_path.name)
        case.setdefault("paper_title", metadata.get("paper_title"))
        case.setdefault("paper_type", record.get("paper_type"))
        case.setdefault("study_type", paper_type_hint)
        if not case.get("case_id"):
            case["case_id"] = f"{paper_slug(pdf_path)}_case_{index}"
        case.setdefault("extraction_model", model)
    record["cases"] = cases
    record["extraction_model"] = model
    return record


def paper_cases(record: Any) -> list[dict[str, Any]]:
    cases = record.get("cases") if isinstance(record, dict) else []
    return cases if isinstance(cases, list) else []
