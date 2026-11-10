"""Shared helpers for the ablation experiments: problem set, scoring, LLM calls, parsing."""

from __future__ import annotations

import json
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from bmrag import llm  # noqa: E402
from bmrag.evaluation.lookup_scoring import build_lookup, score_one  # noqa: E402
from bmrag.schemas import MELT_TEMPERATURE_GRID, POSITION_GRID  # noqa: E402

PROBLEM_SET = REPO / "experiments" / "problem_sets" / "problem_set_18city.json"
ABLATION_MAX_NEW_TOKENS = 2048   # Hugging Face baselines; agents use the llm module's default

TASK_OUTPUT_SPEC = """\
You are predicting the optimal Phase-Change Material (PCM) layer placement
inside a multi-layer building exterior wall.

Wall layers are fixed (4 solid layers); 5 candidate PCM interfaces:
  pos0 = outside surface, before stucco
  pos1 = stucco / gypsum interface
  pos2 = gypsum / insulation interface
  pos3 = MIDPOINT of insulation (insulation split R/2; PCM at centre)
  pos4 = insulation / inner-gypsum interface

Pick:
  - pcm_interface_pos ∈ {0, 1, 2, 3, 4}
  - pcm_tm_c           ∈ {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

Optimization target: minimize total annual building energy (heating + cooling).

Output STRICT JSON only (no markdown, no prose), with these keys:
{
  "pcm_interface_pos": integer 0..4,
  "pcm_tm_c":          number from the allowed set,
  "rationale":         one-paragraph explanation
}
"""


def load_problem_set(path: Optional[Path] = None) -> list[dict[str, Any]]:
    """Explicit path, then PROBLEM_SET_PATH (e.g. the cairo/riyadh set), then the 18-city set."""
    env_path = os.environ.get("PROBLEM_SET_PATH")
    chosen = path or (Path(env_path) if env_path else PROBLEM_SET)
    return json.loads(Path(chosen).read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def _lookup() -> dict[str, Any]:
    return build_lookup()


def score(city: str, pred_pos: Optional[int], pred_tm: Optional[float]) -> dict[str, Any]:
    return score_one(city, pred_pos, pred_tm, _lookup())


def get_llm(model_name: str, temperature: float = 0.0) -> Any:
    return llm.create_chat_model(model_name, temperature=temperature, max_retries=2,
                                 max_new_tokens=ABLATION_MAX_NEW_TOKENS)


def invoke_llm(chat_model: Any, prompt: str) -> str:
    return llm.complete(chat_model, prompt, caller="ablation_baseline")


def parse_json_strict(raw: str) -> dict[str, Any]:
    """Parse a JSON reply, tolerating a code fence or prose around the object."""
    raw = llm.strip_code_fence(raw.strip())
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start >= 0 and end > start:
            return json.loads(raw[start:end + 1])
        raise


def snap_to_grid(pos: Any, tm: Any) -> tuple[Optional[int], Optional[float]]:
    """Nearest allowed (position, melting temperature); None for a value that cannot be read."""
    try:
        position = int(pos)
        if position not in POSITION_GRID:
            position = min(POSITION_GRID, key=lambda g: abs(g - position))
    except Exception:
        position = None
    try:
        melt = float(tm)
        if melt not in MELT_TEMPERATURE_GRID:
            melt = float(min(MELT_TEMPERATURE_GRID, key=lambda g: abs(g - melt)))
    except Exception:
        melt = None
    return position, melt


def build_task_text(record: dict[str, Any]) -> str:
    task = record["analyst_input"]
    parts = [f"City: {task['climate']}",
             f"Building: {task['building_type']}",
             f"Wall: {task['wall_construction']}",
             f"PCM parameters: {task['pcm_parameters']}",
             f"Goal: {task['goal']}"]
    if task.get("orientation"):
        parts.append(f"Orientation: {task['orientation']}")
    if task.get("season"):
        parts.append(f"Season: {task['season']}")
    return "\n".join(parts)
