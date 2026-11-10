"""Analyst agent: rewrites the user's task into the queries and keywords retrieval needs."""

from __future__ import annotations

import json

from bmrag import llm
from bmrag.profiles import PAPER_V1, Profile
from bmrag.prompts import load_prompt
from bmrag.schemas import AnalystInput, AnalystOutput


def analyze(task: AnalystInput, profile: Profile = PAPER_V1) -> AnalystOutput:
    task_json = json.dumps(task.to_dict(), ensure_ascii=False, indent=2)
    chat_model = llm.create_chat_model(profile.analyst_model, temperature=0, max_retries=2)
    reply = llm.complete(chat_model, load_prompt("analyst").format(user_input_json=task_json), caller="analyst")
    reply = llm.strip_code_fence(reply)
    try:
        data = json.loads(reply)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Analyst returned non-JSON output: {exc}\n--- raw ---\n{reply}") from exc

    return AnalystOutput(
        raw_input=task.to_dict(),
        climate_query=str(data.get("climate_query", "")),
        construction_query=str(data.get("construction_query", "")),
        pcm_query=str(data.get("pcm_query", "")),
        composite_query=str(data.get("composite_query", "")),
        materials=[str(m).lower() for m in (data.get("materials") or [])],
        objective_kw=str(data.get("objective_kw", "")).lower(),
        climate_kw=str(data.get("climate_kw", "")),
        pcm_tm_c=data.get("pcm_tm_c"),
        pcm_thickness_mm=data.get("pcm_thickness_mm"),
        wall_layer_count=int(data.get("wall_layer_count") or 0),
        notes=list(data.get("notes") or []),
    )


def passthrough(task: AnalystInput) -> AnalystOutput:
    """Stand-in without an LLM call: the raw task fields become the queries."""
    raw = task.to_dict()
    composite = (f"City: {raw['climate']}\n"
                 f"Building: {raw.get('building_type', '')}\n"
                 f"Wall: {raw.get('wall_construction', '')}\n"
                 f"PCM: {raw.get('pcm_parameters', '')}\n"
                 f"Goal: {raw.get('goal', '')}")
    return AnalystOutput(
        raw_input=raw,
        climate_query=raw["climate"],
        construction_query=raw.get("wall_construction", ""),
        pcm_query=raw.get("pcm_parameters", ""),
        composite_query=composite,
        materials=[],
        objective_kw="energy saving",
        climate_kw=raw["climate"],
        pcm_tm_c=None,
        pcm_thickness_mm=None,
        wall_layer_count=4,
        notes=["[no-analyst] passthrough; no LLM structuring."],
    )
