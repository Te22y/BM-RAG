"""Prompt templates, stored as text files so they can be read and reviewed on their own."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """The template in <name>.txt. Line endings are normalized, so checkouts on any OS agree."""
    return (Path(__file__).parent / f"{name}.txt").read_text(encoding="utf-8")
