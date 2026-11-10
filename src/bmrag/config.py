"""Locations and names shared across BM-RAG. The only place that knows the data layout.

Standard library only: the replay demo imports this package from a fresh clone.
Algorithm settings live in profiles.py, not here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_MODEL = "text-embedding-3-small"   # both vector stores were built with this model
CASE_COLLECTION = "pcm_kg_cases_v1"
RULE_COLLECTION = "pcm_kg_rules_v1"
PAPER_COLLECTION = "multi_modal_rag"
PAPER_ID_KEY = "doc_id"
CHROMA_DIRNAME = "chroma_db"
DOCSTORE_DIRNAME = "docstore"


@lru_cache(maxsize=1)
def load_environment() -> None:
    """Load the repository's .env once. Variables already set in the process win."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(REPO_ROOT / ".env")


@dataclass(frozen=True)
class Paths:
    data_dir: Path
    runs_dir: Path

    @property
    def kg_index(self) -> Path:
        return self.data_dir / "kg_index"

    @property
    def kg_chroma(self) -> Path:
        return self.kg_index / CHROMA_DIRNAME

    @property
    def case_store(self) -> Path:
        return self.kg_index / "case_store.json"

    @property
    def rule_store(self) -> Path:
        return self.kg_index / "rule_store.json"

    @property
    def kg_canonical(self) -> Path:
        return self.data_dir / "kg_canonical"

    @property
    def paper_index(self) -> Path:
        return self.data_dir / "paper_index"

    @property
    def papers(self) -> Path:
        return self.data_dir / "papers"

    @property
    def paper_chunks(self) -> Path:
        return self.data_dir / "paper_chunks"

    @property
    def paper_chunks_cleaned(self) -> Path:
        return self.data_dir / "paper_chunks_cleaned"


@lru_cache(maxsize=1)
def get_paths() -> Paths:
    """Data defaults to <repo>/data and run records to <repo>/runs; both can be redirected."""
    load_environment()
    return Paths(data_dir=Path(os.environ.get("BMRAG_DATA_DIR") or REPO_ROOT / "data"),
                 runs_dir=Path(os.environ.get("BMRAG_RUNS_DIR") or REPO_ROOT / "runs"))
