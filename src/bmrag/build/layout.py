"""Where every build stage reads and writes, relative to one data folder.

The names match the existing data/ folder, so a new build can replace an old one file by file.
Stages pass these exact paths to each other instead of globbing folders.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bmrag.config import CHROMA_DIRNAME, DOCSTORE_DIRNAME

MERGED_PAPERS = "ALL_papers"


@dataclass(frozen=True)
class DataLayout:
    root: Path

    # Inputs
    @property
    def papers(self) -> Path:
        return self.root / "papers"

    # Paper index
    @property
    def paper_chunks(self) -> Path:
        return self.root / "paper_chunks"

    @property
    def paper_chunks_cleaned(self) -> Path:
        return self.root / "paper_chunks_cleaned"

    @property
    def paper_index(self) -> Path:
        return self.root / "paper_index"

    @property
    def paper_manifest(self) -> Path:
        return self.paper_index / "manifest.json"

    @property
    def paper_chroma(self) -> Path:
        return self.paper_index / CHROMA_DIRNAME

    @property
    def paper_docstore(self) -> Path:
        return self.paper_index / DOCSTORE_DIRNAME

    # Graph step 0: one record per paper, plus the merged records the next stage reads
    @property
    def kg_cases(self) -> Path:
        return self.root / "kg_cases"

    @property
    def extracted_papers(self) -> Path:
        return self.kg_cases / f"{MERGED_PAPERS}.json"

    @property
    def extracted_cases(self) -> Path:
        return self.kg_cases / "ALL_cases.json"

    # Graph step 2
    @property
    def kg_entities(self) -> Path:
        return self.root / "kg_entities"

    @property
    def entity_records(self) -> Path:
        return self.kg_entities / f"{MERGED_PAPERS}_structured.json"

    @property
    def quality_report(self) -> Path:
        return self.kg_entities / "step2_quality_report.json"

    # Graph step 5
    @property
    def cypher_records(self) -> Path:
        return self.root / "kg_cypher" / f"{MERGED_PAPERS}_structured_kg.json"

    # Graph step 6
    @property
    def kg_canonical(self) -> Path:
        return self.root / "kg_canonical"

    @property
    def standardized_records(self) -> Path:
        return self.kg_canonical / f"{MERGED_PAPERS}_structured_standardized.json"

    @property
    def standardized_cypher(self) -> Path:
        return self.kg_canonical / f"{MERGED_PAPERS}_structured_standardized.cypher"

    @property
    def canonical_map(self) -> Path:
        return self.kg_canonical / "canonical_map.json"

    @property
    def alignment_report(self) -> Path:
        return self.kg_canonical / "canonical_alignment_report.json"

    @property
    def standardize_report(self) -> Path:
        return self.kg_canonical / "step6_validation_report.json"

    @property
    def neo4j_report(self) -> Path:
        return self.kg_canonical / "step7_dry_run_report.json"

    # Knowledge-graph index
    @property
    def kg_index(self) -> Path:
        return self.root / "kg_index"

    @property
    def kg_chroma(self) -> Path:
        return self.kg_index / CHROMA_DIRNAME

    @property
    def case_store(self) -> Path:
        return self.kg_index / "case_store.json"

    @property
    def rule_store(self) -> Path:
        return self.kg_index / "rule_store.json"

    # Run records
    @property
    def build_runs(self) -> Path:
        return self.root / "build_runs"
