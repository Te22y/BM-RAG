"""BM-RAG web demo server (standard library only).

Replay mode (default) shows sample runs and needs no API key, database or third-party
package:
  python web/server.py

Live mode runs the real pipeline and spends API credit, so it must be asked for.
It needs the agent environment, the data/ folder (not public) and a running Neo4j:
  .venv/Scripts/python.exe -X utf8 web/server.py --live
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

from report import build_report, slim_event  # noqa: E402

STATIC = {"/": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/style.css": ("style.css", "text/css; charset=utf-8")}
# Recorded runs stay private; the public release has only the invented demo samples
SAMPLE_DIRS = (HERE / "samples", HERE / "demo_samples")
LIVE_RUNS = REPO / "runs" / "live"   # git-ignored; live runs cost money, so keep them
PROBLEM_SETS = (REPO / "experiments" / "problem_sets" / "problem_set_18city.json",
                REPO / "experiments" / "problem_sets" / "problem_set_olddry.json")
REQUIRED_FIELDS = ("climate", "building_type", "wall_construction", "pcm_parameters", "goal")
OPTIONAL_FIELDS = ("orientation", "season")
TARGETS = {"min_total_energy", "min_cooling_peak", "min_heating", "thermal_comfort"}
MAX_FIELD_CHARS = 4000
MAX_BODY_BYTES = 64 * 1024
POLL_SECONDS = 0.4


def load_scenarios() -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    for path in PROBLEM_SETS:
        for record in json.loads(path.read_text(encoding="utf-8")):
            if record.get("analyst_input"):
                scenarios.setdefault(record["city"], {"city": record["city"], "zone": record["zone"],
                                                      "input": record["analyst_input"]})
    return scenarios


def validate_task(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ValueError("task must be an object")
    task: dict[str, Any] = {}
    for field in REQUIRED_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field} is required")
        task[field] = value.strip()
    for field in OPTIONAL_FIELDS:
        value = raw.get(field)
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{field} must be text")
        task[field] = value.strip() if value and value.strip() else None
    if any(len(v) > MAX_FIELD_CHARS for v in task.values() if v):
        raise ValueError(f"fields are limited to {MAX_FIELD_CHARS} characters")
    target = raw.get("optimization_target", "min_total_energy")
    if target not in TARGETS:
        raise ValueError("unknown optimization_target")
    task["optimization_target"] = target
    return task


class LiveRun:
    """One pipeline run in a worker thread. Runs cost money, so only one may be active at a time."""

    def __init__(self, city: str | None, title: str, task: dict[str, Any]) -> None:
        self.id = uuid.uuid4().hex
        self.city, self.title, self.task = city, title, task
        self.trace: Any = None
        self.report: dict[str, Any] | None = None
        self.error: str | None = None
        self.done = threading.Event()
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self) -> None:
        try:
            # Imported here: replay mode must work without the agent dependencies
            from bmrag import events, pipeline
            self.trace = events.Trace()
            with events.capture(self.trace):
                result = pipeline.run(self.task).to_dict()
            meta = {"id": self.id, "title": self.title, "zone": None,
                    "recorded": time.strftime("%Y%m%d"), "source": "live"}
            self.report = build_report(meta, result, self.trace.events,
                                       events.summarize_llm_usage(self.trace.events), self.city)
            LIVE_RUNS.mkdir(parents=True, exist_ok=True)
            (LIVE_RUNS / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.city or 'custom'}.json").write_text(
                json.dumps(self.report, ensure_ascii=False, indent=1), encoding="utf-8")
        except ImportError as exc:
            self.error = (f"Agent dependencies are missing ({exc.name}). "
                          "Install requirements-agents.lock.txt and start with the project .venv.")
        except Exception as exc:  # shown to the local user running the demo
            self.error = f"{type(exc).__name__}: {str(exc)[:500]}"
        finally:
            self.done.set()


class State:
    live_enabled = False
    scenarios: dict[str, dict[str, Any]] = {}
    runs: dict[str, LiveRun] = {}
    active: LiveRun | None = None
    lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    server_version = "BMRAGDemo"

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(data, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8", status)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json({"error": message}, status)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path in STATIC:
            name, content_type = STATIC[path]
            self._send((HERE / "static" / name).read_bytes(), content_type)
        elif path == "/api/config":
            self._json({"live_enabled": State.live_enabled, "running": bool(State.active and not State.active.done.is_set())})
        elif path == "/api/samples":
            self._json(self._sample_index())
        elif path.startswith("/api/samples/"):
            self._sample(path.rsplit("/", 1)[1])
        elif path == "/api/scenarios":
            self._json(list(State.scenarios.values()))
        elif path.startswith("/api/run/") and path.endswith("/events"):
            self._stream(path.split("/")[3])
        else:
            self._error(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._error(HTTPStatus.NOT_FOUND, "not found")
        elif not State.live_enabled:
            self._error(HTTPStatus.FORBIDDEN, "Live mode is off. Start the server with --live.")
        else:
            self._start_run()

    @staticmethod
    def _sample_files() -> dict[str, Path]:
        return {path.stem: path for folder in SAMPLE_DIRS for path in sorted(folder.glob("*.json"))}

    @staticmethod
    def _sample_index() -> list[dict[str, Any]]:
        rows = []
        for path in Handler._sample_files().values():
            report = json.loads(path.read_text(encoding="utf-8"))
            rows.append({**report["meta"], "verdict": report["verdict"], "attempts": len(report["attempts"]),
                         "score": report["score"]})
        return rows

    def _sample(self, sample_id: str) -> None:
        known = self._sample_files()
        if sample_id not in known:
            self._error(HTTPStatus.NOT_FOUND, "unknown sample")
        else:
            self._send(known[sample_id].read_bytes(), "application/json; charset=utf-8")

    def _start_run(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY_BYTES:
            self._error(HTTPStatus.BAD_REQUEST, "invalid request size")
            return
        try:
            body = json.loads(self.rfile.read(length))
            task = validate_task(body.get("task"))
        except (ValueError, AttributeError) as exc:
            self._error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        # Score only an unedited scenario: the optimum belongs to that exact task.
        scenario = State.scenarios.get(body.get("city"))
        unedited = scenario is not None and all(
            (scenario["input"].get(k) or None) == task.get(k) for k in task)
        with State.lock:
            if State.active and not State.active.done.is_set():
                self._error(HTTPStatus.CONFLICT, "A run is already in progress.")
                return
            run = LiveRun(scenario["city"] if unedited else None,
                          scenario["city"].title() if unedited else "Custom task", task)
            State.active = State.runs[run.id] = run
        self._json({"id": run.id, "scored": unedited})

    def _stream(self, run_id: str) -> None:
        run = State.runs.get(run_id)
        if run is None:
            self._error(HTTPStatus.NOT_FOUND, "unknown run")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        def emit(kind: str, data: Any) -> None:
            self.wfile.write(f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n".encode("utf-8"))
            self.wfile.flush()

        sent = 0
        try:
            while True:
                finished = run.done.is_set()   # read first so no event is missed at the end
                events = run.trace.events if run.trace else []
                for event in events[sent:]:
                    emit("trace", slim_event(event))
                sent = len(events)
                if finished:
                    if run.error:
                        emit("failure", {"message": run.error})
                    else:
                        emit("report", run.report)
                    return
                time.sleep(POLL_SECONDS)
        except (BrokenPipeError, ConnectionError):
            pass   # page closed; the run itself continues


def main() -> None:
    ap = argparse.ArgumentParser(description="BM-RAG web demo")
    ap.add_argument("--host", default="127.0.0.1", help="bind address; keep local in live mode")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--live", action="store_true", help="enable real pipeline runs (spends API credit)")
    args = ap.parse_args()

    State.live_enabled = args.live
    State.scenarios = load_scenarios()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    mode = "live + replay" if args.live else "replay only"
    print(f"BM-RAG demo ({mode}): http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
