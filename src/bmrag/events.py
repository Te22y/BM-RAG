"""Run tracing: agent timeline, retrieval calls and LLM calls with token cost.

Agents report through emit(); nothing is recorded unless a Trace is active. The
active Trace lives in a context variable, so concurrent runs do not mix.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.tracers.context import register_configure_hook

# USD per 1M tokens: (input, cached input, output). Assumed list prices, not reconciled with invoices.
PRICES = {
    "gpt-5-mini": (0.25, 0.025, 2.00),
    "gpt-5.2": (1.75, 0.175, 14.00),
    "gpt-5": (1.25, 0.125, 10.00),
}
USAGE_FIELDS = ("input_tokens", "cached_tokens", "output_tokens", "reasoning_tokens")


def _cost(model: Optional[str], usage: dict[str, Any]) -> Optional[float]:
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if (model or "").startswith(k)), None)
    if key is None or not usage:
        return None
    price_input, price_cached, price_output = PRICES[key]
    cached = (usage.get("input_token_details") or {}).get("cache_read", 0) or 0
    fresh = (usage.get("input_tokens") or 0) - cached
    return (fresh * price_input + cached * price_cached + (usage.get("output_tokens") or 0) * price_output) / 1e6


class Trace:
    """Ordered event list of one pipeline run."""

    def __init__(self) -> None:
        self.started = time.time()
        self.events: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {"agent": None, "attempt": 0}

    def add(self, kind: str, **fields: Any) -> None:
        self.events.append({"i": len(self.events) + 1, "t": round(time.time() - self.started, 3),
                            "kind": kind, **self.context, **fields})


class _LLMRecorder(BaseCallbackHandler):
    """Records every chat-model call made while a Trace is active."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.pending: dict[Any, dict[str, Any]] = {}

    def on_chat_model_start(self, serialized: Any, messages: Any, *, run_id: Any, **kwargs: Any) -> None:
        params = kwargs.get("invocation_params") or {}
        chars = sum(len(m.content) for batch in messages for m in batch if isinstance(getattr(m, "content", None), str))
        self.pending[run_id] = {"start": time.time(), "prompt_chars": chars,
                                "caller": (kwargs.get("metadata") or {}).get("bmrag_caller", "unknown"),
                                "model_requested": params.get("model") or params.get("model_name")}

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs: Any) -> None:
        call = self.pending.pop(run_id, {"start": time.time()})
        message = getattr(response.generations[0][0], "message", None)
        usage = dict(getattr(message, "usage_metadata", None) or {})
        model = (getattr(message, "response_metadata", None) or {}).get("model_name") or call.get("model_requested")
        self.trace.add("llm", caller=call.get("caller"), model=model,
                       elapsed_sec=round(time.time() - call["start"], 3),
                       prompt_chars=call.get("prompt_chars"),
                       input_tokens=usage.get("input_tokens"),
                       cached_tokens=(usage.get("input_token_details") or {}).get("cache_read"),
                       output_tokens=usage.get("output_tokens"),
                       reasoning_tokens=(usage.get("output_token_details") or {}).get("reasoning"),
                       cost_usd=_cost(model, usage))

    def on_llm_error(self, error: BaseException, *, run_id: Any, **kwargs: Any) -> None:
        call = self.pending.pop(run_id, {"start": time.time()})
        self.trace.add("llm_error", caller=call.get("caller"), model=call.get("model_requested"),
                       elapsed_sec=round(time.time() - call["start"], 3), error=repr(error)[:500])


_active_trace: ContextVar[Optional[Trace]] = ContextVar("bmrag_trace", default=None)
_llm_recorder: ContextVar[Optional[_LLMRecorder]] = ContextVar("bmrag_llm_recorder", default=None)
register_configure_hook(_llm_recorder, inheritable=True)


@contextmanager
def capture(trace: Trace) -> Iterator[Trace]:
    """Record everything the pipeline does inside this block into `trace`."""
    trace_token = _active_trace.set(trace)
    recorder_token = _llm_recorder.set(_LLMRecorder(trace))
    try:
        yield trace
    finally:
        _llm_recorder.reset(recorder_token)
        _active_trace.reset(trace_token)


def emit(kind: str, **fields: Any) -> None:
    trace = _active_trace.get()
    if trace is not None:
        trace.add(kind, **fields)


@dataclass
class AgentSpan:
    summary: dict[str, Any] = field(default_factory=dict)   # filled by the caller, stored on agent_end


@contextmanager
def agent_span(agent: str) -> Iterator[AgentSpan]:
    """Mark one agent's work. Each Reasoner span starts a new attempt."""
    trace = _active_trace.get()
    span = AgentSpan()
    if trace is None:
        yield span
        return
    if agent == "reasoner":
        trace.context["attempt"] += 1
    trace.context["agent"] = agent
    started = time.time()
    trace.add("agent_start")
    try:
        yield span
    finally:
        trace.add("agent_end", elapsed_sec=round(time.time() - started, 3), **span.summary)
        trace.context["agent"] = None


def _result_summary(items: list[Any]) -> dict[str, Any]:
    return {"n": len(items),
            "ids": [getattr(x, "evidence_id", None) for x in items][:12],
            "scores": [round(x.score, 4) if isinstance(getattr(x, "score", None), float) else None
                       for x in items][:12]}


def traced_retrieval(label: str, details: dict[str, Any], call: Callable[[], Any]) -> Any:
    """Run a retrieval and record it. A (results, diagnostics) return carries rewrite details."""
    if _active_trace.get() is None:
        return call()
    started = time.time()
    try:
        result = call()
    except Exception as exc:
        emit("retrieval_error", fn=label, **details, elapsed_sec=round(time.time() - started, 3), error=repr(exc)[:500])
        raise
    items, diagnostics = result if isinstance(result, tuple) else (result, None)
    rewrite = {}
    if diagnostics:
        rewrite = {"fallback_used": diagnostics.get("fallback_used"),
                   "fallback_reason": diagnostics.get("fallback_reason"),
                   "rewritten_query": diagnostics.get("rewritten_query"),
                   "scores_before": [round(s, 4) for s in diagnostics.get("scores", [])]}
    emit("retrieval", fn=label, **details, elapsed_sec=round(time.time() - started, 3),
         **_result_summary(items), **rewrite)
    return result


def summarize_llm_usage(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Calls, tokens, time and cost per caller and per model."""
    by_caller: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["kind"] != "llm":
            continue
        for key, bucket in ((event["caller"], by_caller), (event["model"], by_model)):
            totals = bucket.setdefault(key or "unknown", {"calls": 0, **dict.fromkeys(USAGE_FIELDS, 0),
                                                          "elapsed_sec": 0.0, "cost_usd": 0.0})
            totals["calls"] += 1
            for name in USAGE_FIELDS:
                totals[name] += event.get(name) or 0
            totals["elapsed_sec"] = round(totals["elapsed_sec"] + (event.get("elapsed_sec") or 0), 3)
            totals["cost_usd"] = round(totals["cost_usd"] + (event.get("cost_usd") or 0), 5)
    return {"by_caller": by_caller, "by_model": by_model,
            "cost_usd_total": round(sum(t["cost_usd"] for t in by_model.values()), 5)}
