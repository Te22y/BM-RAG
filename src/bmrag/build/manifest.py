"""The run record every build writes to <data>/build_runs/<UTC time>.json.

It holds what is needed to reproduce or audit a build: code version, package and tool versions,
the full profile, the input PDFs with their hashes, each stage's time and result, token usage
with a cost estimate, and the hashes of the main output files. It never holds an API key.
"""

from __future__ import annotations

import hashlib
import logging
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from bmrag.build.jsonio import write_json
from bmrag.build.layout import DataLayout
from bmrag.build.llm_client import UsageLog
from bmrag.build.profiles import BuildProfile
from bmrag.config import REPO_ROOT

log = logging.getLogger(__name__)

PACKAGES = (
    "bmrag",
    "openai",
    "anthropic",
    "langchain-core",
    "langchain-openai",
    "langchain-chroma",
    "chromadb",
    "pymupdf",
    "pdfplumber",
    "unstructured",
    "unstructured-inference",
    "torch",
    "neo4j",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class RunRecord:
    def __init__(self, layout: DataLayout, profile: BuildProfile, *, overrides: list[str], options: dict[str, Any]):
        self.layout = layout
        self.profile = profile
        self.usage = UsageLog()
        self.started = datetime.now(UTC)
        self.data: dict[str, Any] = {
            "started_utc": self.started.isoformat(timespec="seconds"),
            "command": [Path(sys.argv[0]).name, *sys.argv[1:]],
            "data_dir": str(layout.root),
            "profile": profile.to_dict(),
            "overrides": overrides,
            "options": options,
            "environment": environment(),
            "inputs": [],
            "stages": [],
            "warnings": [],
        }

    def add_inputs(self, pdfs: list[Path]) -> None:
        known = {item["file"] for item in self.data["inputs"]}
        for pdf in pdfs:
            if pdf.name not in known:
                self.data["inputs"].append({"file": pdf.name, "bytes": pdf.stat().st_size, "sha256": sha256_file(pdf)})

    def warn(self, message: str) -> None:
        log.warning(message)
        self.data["warnings"].append(message)

    @contextmanager
    def stage(self, name: str) -> Iterator[dict[str, Any]]:
        """Time one stage. The yielded dict collects the stage's own counts for the record."""
        entry: dict[str, Any] = {"stage": name, "status": "running"}
        self.data["stages"].append(entry)
        start = time.perf_counter()
        log.info("== %s", name)
        try:
            yield entry
            entry["status"] = "ok"
        except BaseException as exc:
            entry["status"] = "failed"
            entry["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            entry["seconds"] = round(time.perf_counter() - start, 1)

    def write(self, outputs: list[Path]) -> Path:
        finished = datetime.now(UTC)
        self.data["finished_utc"] = finished.isoformat(timespec="seconds")
        self.data["status"] = "failed" if any(s["status"] != "ok" for s in self.data["stages"]) else "ok"
        self.data["usage"] = self.usage.summary()
        self.data["outputs"] = [
            {"file": p.relative_to(self.layout.root).as_posix(), "sha256": sha256_file(p)}
            for p in outputs
            if p.is_file()
        ]
        path = self.layout.build_runs / f"{self.started.strftime('%Y%m%dT%H%M%SZ')}.json"
        write_json(path, self.data)
        return path


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "git": _git_state(),
        "packages": {name: _version(name) for name in PACKAGES},
        "tools": {
            "pdftoppm": _tool_version(["pdftoppm", "-v"]),
            "tesseract": _tool_version(["tesseract", "--version"]),
        },
    }


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git_state() -> dict[str, Any]:
    def git(*args: str) -> str | None:
        try:
            result = subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    status = git("status", "--porcelain")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def _tool_version(command: list[str]) -> str | None:
    if not shutil.which(command[0]):
        return None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    lines = (result.stdout or result.stderr).strip().splitlines()
    return lines[0] if lines else None
