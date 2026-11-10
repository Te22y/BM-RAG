"""Run the build pipeline on test PDFs and reduce everything it produces to digests.

A baseline recorded with one code version can then be compared with another: same output
files with the same bytes, same Chroma contents and the same LLM requests. Only digests
are stored, so a baseline made from a real paper contains none of its text.

Each step runs in its own process. Steps that call OpenAI run through tests/build_standin.py.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STANDIN = REPO / "tests" / "build_standin.py"
CHROMA_COLLECTIONS = {
    "paper_index/chroma_db": ["multi_modal_rag"],
    "kg_index/chroma_db": ["pcm_kg_cases_v1", "pcm_kg_rules_v1"],
}
# The paper index adds documents without ids, so Chroma assigns random UUIDs; doc_id is in the metadata.
RANDOM_ID_COLLECTIONS = {"multi_modal_rag"}
TEXT_SUFFIXES = {".json", ".jsonl", ".txt", ".cypher", ".md"}
# Windows install locations used when the tools are not on PATH
TOOL_GLOBS = [
    os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\oschwartz10612.Poppler_*\poppler-*\Library\bin"),
    r"C:\Program Files\Tesseract-OCR",
]


@dataclass(frozen=True)
class Step:
    """One command. In args, {data} becomes the data folder. stand_in routes OpenAI calls to the stand-ins.

    after, if given, is called with the data folder once the command succeeded.
    """

    name: str
    module: str
    args: tuple[str, ...] = ()
    stand_in: bool = False
    after: Callable[[Path], None] | None = None


@dataclass
class Case:
    name: str
    pdfs: dict[str, Path]  # file name in data/papers -> source path
    extraction_replies: dict[str, str] = field(default_factory=dict)


def tool_path() -> str | None:
    """PATH with Poppler and Tesseract added, or None when either is missing."""
    extra = [d for pattern in TOOL_GLOBS for d in glob.glob(pattern)]
    path = os.pathsep.join(extra + [os.environ.get("PATH", "")])
    found = all(shutil.which(tool, path=path) for tool in ("pdftoppm", "tesseract"))
    return path if found else None


def build_packages_available() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) for name in ("pymupdf", "pdfplumber", "unstructured", "chromadb"))


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_case(case: Case, steps: Sequence[Step], work: Path) -> dict:
    """Build case from scratch in work/<case.name> and return its digests."""
    root = work / case.name
    shutil.rmtree(root, ignore_errors=True)
    data = root / "data"
    (data / "papers").mkdir(parents=True)
    for name, source in case.pdfs.items():
        shutil.copy2(source, data / "papers" / name)
    replies = root / "replies.json"
    replies.write_text(json.dumps({"extraction": case.extraction_replies}), encoding="utf-8")
    requests_dir = root / "requests"
    requests_dir.mkdir()
    logs = root / "logs"
    logs.mkdir()

    env = dict(
        os.environ,
        BMRAG_DATA_DIR=str(data),
        BMRAG_RUNS_DIR=str(root / "runs"),
        PYTHONUTF8="1",
        OPENAI_API_KEY="stand-in",
        ANTHROPIC_API_KEY="",
        LANGSMITH_TRACING="false",
        LANGSMITH_TRACING_V2="false",
        LANGCHAIN_TRACING_V2="false",
        PYTHONHASHSEED="0",
    )
    env["PATH"] = tool_path() or env.get("PATH", "")
    for step in steps:
        args = [a.replace("{data}", str(data)) for a in step.args]
        command = [sys.executable, "-X", "utf8"]
        if step.stand_in:
            command += [str(STANDIN), str(requests_dir / f"{step.name}.jsonl"), str(replies), "-m", step.module]
        else:
            command += ["-m", step.module]
        result = subprocess.run(
            command + args, cwd=REPO, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        (logs / f"{step.name}.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(
                f"step {step.name} failed ({result.returncode}); see {logs / step.name}.log\n"
                f"{(result.stdout + result.stderr)[-2000:]}"
            )
        if step.after:
            step.after(data)
    return {
        "inputs": {name: sha(path.read_bytes()) for name, path in sorted(case.pdfs.items())},
        "files": file_digests(data, root),
        "chroma": chroma_digests(data),
        "requests": request_digests(requests_dir),
        "quality": quality_summary(data),
    }


def _normalize(text: str, root: Path) -> str:
    """Replace the work folder in any spelling (plain, JSON-escaped, forward slashes) with <WORK>."""
    variants = {str(root), str(root).replace("\\", "\\\\"), root.as_posix()}
    for variant in sorted(variants, key=len, reverse=True):
        text = text.replace(variant, "<WORK>")
    return text


def file_digests(data: Path, root: Path) -> dict[str, str]:
    # The folder-mode copy only fed the old quality check; its counts are compared in quality_summary.
    skipped = tuple(CHROMA_COLLECTIONS) + ("papers/", "kg_entities_folder_mode/")
    digests = {}
    for path in sorted(p for p in data.rglob("*") if p.is_file()):
        rel = path.relative_to(data).as_posix()
        if rel.startswith(skipped) or rel.endswith("step2_quality_report.json"):
            continue
        raw = path.read_bytes()
        if path.suffix in TEXT_SUFFIXES:
            raw = _normalize(raw.decode("utf-8"), root).encode("utf-8")
        elif path.suffix == ".pkl":
            raw = _normalized_pickle(raw, root)
        digests[rel] = sha(raw)
    return digests


def _normalized_pickle(raw: bytes, root: Path) -> bytes:
    """Re-pickle with the work folder and file times replaced, so the digest does not depend on the machine.

    unstructured stores each element's file_directory and last_modified (local time) in its metadata.
    """
    import pickle

    seen: set[int] = set()

    def clean(value):
        if isinstance(value, str):
            return _normalize(value, root)
        if isinstance(value, (set, frozenset)):  # set order depends on the process's hash seed
            return sorted((clean(v) for v in value), key=repr)
        if id(value) in seen:
            return value
        seen.add(id(value))
        if isinstance(value, list):
            value[:] = [clean(v) for v in value]
        elif isinstance(value, dict):
            for key in list(value):
                value[key] = "<MTIME>" if key == "last_modified" else clean(value[key])
        elif hasattr(value, "__dict__"):
            attributes = vars(value)  # written directly: some metadata fields are read-only properties
            for key in list(attributes):
                attributes[key] = "<MTIME>" if key == "last_modified" else clean(attributes[key])
        return value

    return pickle.dumps(clean(pickle.loads(raw)), protocol=4)


def chroma_digests(data: Path) -> dict[str, str]:
    import chromadb

    digests = {}
    for folder, names in CHROMA_COLLECTIONS.items():
        if not (data / folder).exists():
            continue
        client = chromadb.PersistentClient(path=str(data / folder))
        for name in names:
            got = client.get_collection(name).get(include=["documents", "metadatas", "embeddings"])
            ids = [""] * len(got["ids"]) if name in RANDOM_ID_COLLECTIONS else got["ids"]
            vectors = [[round(float(x), 6) for x in e] for e in got["embeddings"]]
            rows = sorted(
                zip(
                    ids,
                    got["documents"],
                    [json.dumps(m, sort_keys=True) for m in got["metadatas"]],
                    vectors,
                    strict=True,
                )
            )
            digests[f"{folder}/{name}"] = sha(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode())
        del client
    return digests


def request_digests(requests_dir: Path) -> dict[str, list[str]]:
    result = {}
    for path in sorted(requests_dir.glob("*.jsonl")):
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        result[path.stem] = sorted(f"{r['kind']}:{r['model']}:{r['digest']}" for r in rows)
    return result


def quality_summary(data: Path) -> dict:
    """The quality check's counts; its file list changes with how the merged file is passed."""
    reports = sorted(data.rglob("step2_quality_report.json"))
    if not reports:
        return {}
    report = json.loads(reports[0].read_text(encoding="utf-8"))
    return {key: report[key] for key in ("record_count", "issue_counts", "issue_code_counts")}


def compare(expected: dict, actual: dict, allowed_new_files: Sequence[str] = ()) -> list[str]:
    """Differences between two digest sets, as readable lines. Empty means identical."""
    problems = []
    for section in ("inputs", "chroma", "requests", "quality"):
        if expected.get(section) != actual.get(section):
            problems.append(
                f"{section} differ: expected {json.dumps(expected.get(section))[:300]} "
                f"got {json.dumps(actual.get(section))[:300]}"
            )
    exp_files, act_files = expected["files"], actual["files"]
    for name in sorted(set(exp_files) | set(act_files)):
        if name not in act_files:
            problems.append(f"missing file: {name}")
        elif name not in exp_files:
            if not any(Path(name).match(p) for p in allowed_new_files):
                problems.append(f"new file: {name}")
        elif exp_files[name] != act_files[name]:
            problems.append(f"changed file: {name}")
    return problems
