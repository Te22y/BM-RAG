"""A synthetic paper for the build pipeline tests, with the extraction reply a model could give for it.

Everything here is invented, so it can be published and run anywhere. Needs PyMuPDF to write the PDF.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

FIXED_MTIME = 1767225600  # 2026-01-01 00:00 UTC
SYNTHETIC_PDF_NAME = "Synthetic study of a PCM layer in a masonry wall.pdf"
SYNTHETIC_TITLE = "Synthetic study of a PCM layer in a masonry wall"

_PAGES = [
    [
        ("title", SYNTHETIC_TITLE),
        ("body", "A. Author, B. Author. Journal of Invented Results 1 (2026) 1-3."),
        ("heading", "Abstract"),
        (
            "body",
            "This invented study places a phase change material (PCM) layer at two positions in a "
            "masonry wall in the fictional city of Testville. The layer next to the interior plaster "
            "lowers the cooling load by 12 % in summer, while the layer behind the exterior render "
            "lowers it by 5 %. All figures in this document are synthetic test data.",
        ),
        ("heading", "1. Introduction"),
        (
            "body",
            "Latent heat storage in building walls can shift and reduce peak heat flux. The position "
            "of the PCM layer decides whether the latent heat is used during the daily cycle.",
        ),
    ],
    [
        ("heading", "2. Method"),
        (
            "body",
            "A one-dimensional finite difference model was run for 30 summer days with a hot-dry "
            "climate. The wall layers from outside are render 15 mm, brick 115 mm, insulation 40 mm, "
            "PCM board 10 mm and plaster 15 mm. The PCM melts at 26 C with a latent heat of 180 kJ/kg.",
        ),
        ("heading", "Table 1. Wall layers"),
        (
            "table",
            [
                ["Layer", "Material", "Thickness (mm)"],
                ["1", "Render", "15"],
                ["2", "Brick", "115"],
                ["3", "Insulation", "40"],
                ["4", "PCM board", "10"],
                ["5", "Plaster", "15"],
            ],
        ),
        ("figure", "Fig. 1. Wall section with the two studied PCM positions."),
    ],
    [
        ("heading", "3. Results"),
        (
            "body",
            "Case A puts the PCM board next to the interior plaster and reduces the cooling load by "
            "12 %. Case B puts it behind the exterior render and reduces the cooling load by 5 %, because "
            "the layer stays liquid during most of the day and its latent heat is not used.",
        ),
        ("heading", "4. Conclusions"),
        ("body", "In a hot-dry summer the interior position is better than the exterior position."),
        ("heading", "Acknowledgments"),
        ("body", "None. This is synthetic test data."),
        ("heading", "References"),
        ("body", "[1] A. Author, An invented reference, Journal of Invented Results 1 (2026) 4-5."),
    ],
]


def write_synthetic_pdf(path: Path) -> Path:
    """Write the synthetic paper: 3 pages with headings, a ruled table and one figure."""
    import pymupdf

    document = pymupdf.open()
    for blocks in _PAGES:
        page = document.new_page(width=595, height=842)
        y = 60.0
        for kind, content in blocks:
            if kind == "table":
                y = _draw_table(page, content, y)
            elif kind == "figure":
                y = _draw_figure(page, content, y)
            else:
                size = {"title": 16, "heading": 12, "body": 10}[kind]
                box = pymupdf.Rect(60, y, 535, y + 200)
                spare = page.insert_textbox(box, content, fontsize=size, fontname="helv")
                y += 200 - spare + 8
    # Fixed metadata keeps the file byte-identical between runs.
    document.set_metadata(
        {
            "title": SYNTHETIC_TITLE,
            "author": "bmrag tests",
            "producer": "bmrag tests",
            "creator": "bmrag tests",
            "creationDate": "D:20260101000000Z",
            "modDate": "D:20260101000000Z",
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    document.save(str(path), deflate=True, garbage=4, no_new_id=True)
    document.close()
    # unstructured stores the file's modification time in every element, so fix it too.
    os.utime(path, (FIXED_MTIME, FIXED_MTIME))
    return path


def _draw_table(page, rows: list[list[str]], y: float) -> float:
    import pymupdf

    widths = [60, 200, 120]
    height = 18
    for r, row in enumerate(rows):
        x = 60
        for c, cell in enumerate(row):
            rect = pymupdf.Rect(x, y + r * height, x + widths[c], y + (r + 1) * height)
            page.draw_rect(rect, color=(0, 0, 0), width=0.6)
            page.insert_text((x + 4, rect.y1 - 5), cell, fontsize=9, fontname="helv")
            x += widths[c]
    return y + len(rows) * height + 12


def _draw_figure(page, caption: str, y: float) -> float:
    import pymupdf

    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 240, 160), False)
    pixmap.clear_with(235)
    for x0, shade in ((20, 120), (60, 200), (120, 160), (200, 90)):
        pixmap.set_rect(pymupdf.IRect(x0, 20, x0 + 30, 140), (shade, shade, 255 - shade))
    page.insert_image(pymupdf.Rect(60, y, 300, y + 160), pixmap=pixmap)
    page.insert_textbox(pymupdf.Rect(60, y + 165, 535, y + 195), caption, fontsize=9, fontname="helv")
    return y + 205


_LAYERS = [
    {"index_from_outside": 1, "material": "Render", "thickness": "15 mm"},
    {"index_from_outside": 2, "material": "Brick", "thickness": "115 mm"},
    {"index_from_outside": 3, "material": "Insulation", "thickness": "40 mm"},
    {"index_from_outside": 4, "material": "PCM board", "thickness": "10 mm"},
    {"index_from_outside": 5, "material": "Plaster", "thickness": "15 mm"},
]


def _case(case_id: str, position: str, saving: str, mechanism: str) -> dict:
    return {
        "case_id": case_id,
        "case_name": f"PCM board {position}",
        "evidence_type": "simulation",
        "source_study_reference": None,
        "study_method": "one-dimensional finite difference model",
        "simulation_tool_or_experiment": "in-house code",
        "validation": "none (synthetic)",
        "climate_location": "Testville (fictional, hot-dry)",
        "formal_climate_zone": "BWh",
        "weather_file_or_period": "30 summer days",
        "season": "summer",
        "building_or_room": "single room",
        "operation_conditions": "cooling set point 24 C",
        "envelope_component": "external wall",
        "orientation": "south",
        "wall_layers_exterior_to_interior": _LAYERS,
        "insulation_layer": "Insulation 40 mm",
        "pcm_material": "paraffin PCM board",
        "pcm_material_class": "organic",
        "pcm_encapsulation_or_integration": "board",
        "phase_change_temperature": "26 C",
        "latent_heat": "180 kJ/kg",
        "pcm_thickness_or_quantity": "10 mm",
        "pcm_density": None,
        "pcm_thermal_conductivity": None,
        "pcm_specific_heat": None,
        "studied_pcm_position": position,
        "compared_positions": ["next to the interior plaster", "behind the exterior render"],
        "optimal_pcm_position": "next to the interior plaster",
        "optimal_position_condition": "hot-dry summer",
        "evaluation_metrics": ["cooling load"],
        "quantitative_results": f"cooling load reduced by {saving}",
        "main_conclusion": f"The PCM board {position} reduces the cooling load by {saving}.",
        "mechanism_explanation": mechanism,
        "limitations": "synthetic data",
        "missing_information": [],
        "kg_triplets": [
            {
                "subject": "paraffin PCM board",
                "subject_type": "PCMMaterial",
                "predicate": "PLACED_AT",
                "object": position,
                "object_type": "WallPosition",
            },
            {
                "subject": "paraffin PCM board",
                "subject_type": "PCMMaterial",
                "predicate": "HAS_PROPERTY",
                "object": "phase change temperature 26 C",
                "object_type": "MaterialProperty",
            },
        ],
        "mechanism_chain_triplets": [
            {
                "step": 1,
                "cause": "High daytime outdoor temperature",
                "cause_type": "ClimateCondition",
                "relation": "CAUSES",
                "effect": "Inward heat flux through the wall",
                "effect_type": "PhysicalProcess",
            },
            {
                "step": 2,
                "cause": "Inward heat flux through the wall",
                "cause_type": "PhysicalProcess",
                "relation": "ENABLES",
                "effect": f"Optimized PCM layer at {position}",
                "effect_type": "WallPosition",
            },
            {
                "step": 3,
                "cause": f"Optimized PCM layer at {position}",
                "cause_type": "WallPosition",
                "relation": "REDUCES",
                "effect": "Cooling load",
                "effect_type": "PerformanceOutcome",
            },
        ],
    }


SYNTHETIC_RECORD = {
    "paper_type": "original",
    "classification_reason": "Reports its own simulation.",
    "paper_metadata": {
        "paper_id": "synthetic_pcm_wall_2026",
        "paper_title": SYNTHETIC_TITLE,
        "source_file": SYNTHETIC_PDF_NAME,
    },
    "cases": [
        _case(
            "case_a_interior",
            "next to the interior plaster",
            "12 %",
            "The layer melts and solidifies every day, so its latent heat absorbs the peak heat flux.",
        ),
        _case(
            "case_b_exterior",
            "behind the exterior render",
            "5 %",
            "The layer stays liquid most of the day, so little latent heat is used.",
        ),
    ],
    "paper_level_summary": {
        "main_findings": "The interior position is better in a hot-dry summer.",
        "conditional_optimal_pcm_placement": ["hot-dry summer: next to the interior plaster"],
        "mechanism_explanation": "Daily melting and solidification is needed to use the latent heat.",
        "kg_triplets": [],
    },
    "kg_ready_rules": [
        {
            "rule_id": "R1",
            "condition_nodes": ["hot-dry summer"],
            "relation": "favors",
            "outcome_node": "interior side PCM position",
            "performance_objective": "cooling load",
            "evidence_case_ids": ["case_a_interior", "case_b_exterior"],
            "confidence": "high",
            "evidence_sentence": "In a hot-dry summer the interior position is better than the exterior position.",
            "missing_or_uncertain_fields": [],
        },
    ],
    "extraction_warnings": [],
}

SYNTHETIC_REPLY = json.dumps(SYNTHETIC_RECORD, ensure_ascii=False)
