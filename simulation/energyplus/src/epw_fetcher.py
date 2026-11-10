"""Locate EPW files on disk by filename.

v3.1: replaces network download. EPWs must be pre-placed under cfg.epw_root.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from utils import get_logger

log = get_logger("epw_locator")


@lru_cache(maxsize=1)
def _index_epw_root(root_str: str) -> Dict[str, Path]:
    """Build a {filename: full_path} index by walking root recursively.
    Case-insensitive on filename. First match wins.
    """
    root = Path(root_str)
    if not root.exists():
        return {}
    index: Dict[str, Path] = {}
    for p in root.rglob("*.epw"):
        key = p.name.lower()
        if key not in index:
            index[key] = p
    log.info(f"EPW index built: {len(index)} files under {root}")
    return index


def locate_epw(root: Path, filename: str) -> Optional[Path]:
    idx = _index_epw_root(str(root))
    return idx.get(filename.lower())


def locate_all(root: Path, requested: List[Tuple[str, str, str]]
               ) -> Dict[Tuple[str, str], Optional[Path]]:
    """requested: list of (zone, city, epw_filename). Returns {(zone, city): Path or None}."""
    out: Dict[Tuple[str, str], Optional[Path]] = {}
    for zone, city, fname in requested:
        out[(zone, city)] = locate_epw(root, fname)
    return out
