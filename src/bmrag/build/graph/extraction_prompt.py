"""The messages sent to the extraction model.

The scientific criteria come from prompts/<ExtractionSettings.prompt>.txt. This module wraps them
in the JSON contract the later stages rely on, and adds a non-binding paper-type hint guessed
from the file name and abstract. The wording is part of the paper_v1 behaviour: the regression
tests compare every request byte for byte, so edit it only together with a new profile.
"""

from __future__ import annotations

import re

from bmrag.build.graph.paper_content import PaperContent
from bmrag.prompts import load_prompt

FINAL_INSTRUCTION = "\n\nNow extract the paper-level record. Return ONLY the JSON object."

# Lines of the prompt file that ask for Markdown output; the JSON contract below replaces them.
_MARKDOWN_ONLY_RULES = (
    r"do\s+not\s+output\s+json",
    r"do\s+not\s+output\s+tables",
    r"only\s+output\s+the\s+case\s+summaries",
    r"output\s+format\s*:",
    r"^#\s+case\s+summary",
    r"^#\s+review\s+evidence\s+case\s+summary",
    r"^#\s+paper-level\s+summary",
    r"^#\s+review-level\s+synthesis",
    r"^##\s+case\s+\d+",
)

_ORIGINAL_TITLE_HINTS = (
    "numerical study",
    "experimental study",
    "simulation study",
    "parametric study",
    "case study",
    "optimization",
    "optimal location",
    "energy-economic",
    "effect of",
    "energy saving potential",
)
_REVIEW_TITLE_HINTS = (
    "a review",
    "review of",
    "systematic review",
    "literature review",
    "state of the art",
    "state-of-the-art",
    "bibliometric",
    "survey",
)
_REVIEW_ABSTRACT_HINTS = (
    "this review",
    "the present review",
    "this paper reviews",
    "this article reviews",
    "we review",
    "review paper",
    "literature review",
    "systematic review",
    "state of the art",
    "state-of-the-art",
    "bibliometric",
    "survey paper",
)
_ORIGINAL_ABSTRACT_HINTS = (
    "this study investigates",
    "this paper investigates",
    "this study presents",
    "this paper presents",
    "a numerical model",
    "finite difference",
    "finite volume",
    "experimental setup",
    "simulation results",
    "validated against",
)


def sanitize_task_prompt(prompt: str) -> str:
    """Drop the prompt file's Markdown-only output rules, keep its extraction criteria."""
    kept = [
        line
        for line in prompt.splitlines()
        if not any(re.search(rule, line.strip().lower()) for rule in _MARKDOWN_ONLY_RULES)
    ]
    return "\n".join(kept).strip()


def abstract_for_type_hint(text: str) -> str:
    """The abstract, approximately: from "Abstract" to keywords or the introduction, at most 3000 characters."""
    sample = text[:20000]
    start = re.search(r"\babstract\b\s*[:.\-]?\s*", sample, flags=re.IGNORECASE)
    if not start:
        return sample[:3000]
    tail = sample[start.end() :]
    stop = re.search(
        r"\n\s*(?:keywords?|index terms|nomenclature|1\.?\s+introduction|introduction)\b", tail, flags=re.IGNORECASE
    )
    return tail[: stop.start()] if stop else tail[:3000]


def infer_paper_type(text: str, filename: str) -> str:
    """Guess "original", "review" or "meta_analysis" from the title and abstract.

    Only a hint for the model, which decides the type itself. The abstract is used rather than the
    whole text, because introductions of original papers often read like reviews.
    """
    title = filename.lower()
    abstract = abstract_for_type_hint(text).lower()
    if "meta-analysis" in f"{title}\n{abstract}" or "meta analysis" in f"{title}\n{abstract}":
        return "meta_analysis"
    if any(hint in title for hint in _ORIGINAL_TITLE_HINTS):
        return "original"
    if any(hint in title for hint in _REVIEW_TITLE_HINTS):
        return "review"
    review_score = sum(1 for hint in _REVIEW_ABSTRACT_HINTS if hint in abstract)
    original_score = sum(1 for hint in _ORIGINAL_ABSTRACT_HINTS if hint in abstract)
    return "review" if review_score >= 2 and review_score > original_score else "original"


def system_prompt(prompt_name: str, paper_type: str) -> str:
    """The extraction criteria wrapped in the paper-level JSON contract."""
    task_prompt = sanitize_task_prompt(load_prompt(prompt_name))
    return f"""You are running PCM KG Step0. The external prompt below is the
single scientific extraction standard for coverage, missing-data handling, and
KG-ready rule generation. Any Markdown/YAML natural-language output examples in
that prompt are superseded by the JSON machine contract here.

=== PROMPT-FIRST EXTRACTION CRITERIA ===
{task_prompt}

Return ONLY one valid JSON object. No markdown fences, prose, or tables.
Use null for unavailable scalar values and [] for unavailable arrays.
Strict JSON syntax requirements:
- Every multi-sentence text value MUST be encoded as a single valid JSON string.
- Escape embedded quotation marks inside strings.
- Do not use bare newlines inside string values.
- Do not use trailing commas.
- Do not include comments, Markdown, YAML, or code fences.
- If kg_triplets or mechanism_chain_triplets are uncertain, output [] instead of semi-structured text.
- If a field is unknown, use null for scalars and [] for arrays.

Detected paper type hint from filename/text: {paper_type}. The model MUST make
the final paper_type decision after reading the paper.

The top-level JSON object MUST use this shape:
{{
  "paper_type": "research_paper|review_paper|hybrid_paper",
  "classification_reason": "string",
  "paper_metadata": {{
    "paper_id": "string",
    "paper_title": "string",
    "source_file": "string"
  }},
  "cases": [
    {{
      "case_id": "string",
      "case_name": "string",
      "evidence_type": "string",
      "source_study_reference": "string or null",
      "study_method": "string or null",
      "simulation_tool_or_experiment": "string or null",
      "validation": "string or null",
      "climate_location": "string or null",
      "formal_climate_zone": "string or null",
      "weather_file_or_period": "string or null",
      "season": "string or null",
      "building_or_room": "string or null",
      "operation_conditions": "string or null",
      "envelope_component": "string or null",
      "orientation": "string or null",
      "wall_layers_exterior_to_interior": [
        {{"index_from_outside": "integer", "material": "string", "thickness": "string or number or null"}}
      ],
      "insulation_layer": "string or null",
      "pcm_material": "string or null",
      "pcm_material_class": "string or null",
      "pcm_encapsulation_or_integration": "string or null",
      "phase_change_temperature": "string, number, array, or null",
      "latent_heat": "string, number, or null",
      "pcm_thickness_or_quantity": "string, number, or null",
      "pcm_density": "string, number, or null",
      "pcm_thermal_conductivity": "string, number, or null",
      "pcm_specific_heat": "string, number, or null",
      "studied_pcm_position": "string or null",
      "compared_positions": ["array of strings"],
      "optimal_pcm_position": "string or null",
      "optimal_position_condition": "string or null",
      "evaluation_metrics": ["array of strings"],
      "quantitative_results": "string or object or null",
      "main_conclusion": "string",
      "mechanism_explanation": "string",
      "limitations": "string or null",
      "missing_information": ["array of strings"],
      "kg_triplets": ["array of optional extracted triplet objects"],
      "mechanism_chain_triplets": ["array of optional causal chain objects"]
    }}
  ],
  "paper_level_summary": {{
    "main_findings": "string",
    "conditional_optimal_pcm_placement": ["array of strings"],
    "mechanism_explanation": "string"
  }},
  "kg_ready_rules": [
    {{
      "rule_id": "string",
      "condition_nodes": ["array of condition node strings"],
      "relation": "favors|reduces|increases|shifts|delays|requires|conflicts_with|depends_on|not_supported_as_universal",
      "outcome_node": "string",
      "performance_objective": "string or null",
      "evidence_case_ids": ["array of case_id strings"],
      "confidence": "high|medium|low",
      "evidence_sentence": "string",
      "missing_or_uncertain_fields": ["array of strings"]
    }}
  ],
  "extraction_warnings": ["array of strings"]
}}

FINAL OUTPUT REQUIREMENT: return ONLY a valid JSON object. The first non-space
character must be "{{" and the last non-space character must be "}}".
"""


def _paper_text_block(content: PaperContent, filename: str) -> str:
    block = f"SOURCE FILE: {filename}\n\n=== FULL PAPER TEXT ===\n{content.text}\n\n"
    if content.tables_markdown:
        block += "=== EXTRACTED TABLES ===\n" + content.tables_markdown + "\n\n"
    if content.page_images:
        block += (
            f"=== PAGE IMAGES ===\nThe following {len(content.page_images)} page image(s) are provided. "
            "Use them to extract wall layer configs, figures, and visual table data.\n\n"
        )
    return block


def openai_user_content(content: PaperContent, filename: str) -> list[dict]:
    """The user message for OpenAI: the paper text block, each page image with a caption, the instruction."""
    parts: list[dict] = [{"type": "text", "text": _paper_text_block(content, filename)}]
    for page_num, b64 in content.page_images:
        parts.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
        parts.append({"type": "text", "text": f"[Image above: Page {page_num} of {filename}]"})
    parts.append({"type": "text", "text": FINAL_INSTRUCTION})
    return parts


def anthropic_user_content(content: PaperContent, filename: str) -> list[dict]:
    """The same message in Anthropic's image format."""
    parts: list[dict] = [{"type": "text", "text": _paper_text_block(content, filename)}]
    for page_num, b64 in content.page_images:
        parts.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}})
        parts.append({"type": "text", "text": f"[Image above: Page {page_num} of {filename}]"})
    parts.append({"type": "text", "text": FINAL_INSTRUCTION})
    return parts


JSON_REPAIR_SYSTEM = "You repair JSON syntax only. Return only valid JSON."


def json_repair_prompt(raw: str, error: Exception) -> str:
    return f"""Repair this invalid JSON object so it parses with json.loads.

Rules:
- Return ONLY the repaired JSON object.
- Do not add, remove, summarize, reinterpret, or invent scientific facts.
- Preserve all keys, values, arrays, and object structure as much as possible.
- Fix only JSON syntax: missing commas, unescaped quotes, bare newlines in strings, dangling commas, or truncated fences.
- If a value cannot be repaired safely, keep the nearest valid string value rather than inventing a new value.

Parser error:
{type(error).__name__}: {error}

Invalid JSON:
{raw}
"""
