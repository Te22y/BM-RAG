"""Chat calls made by the build stages, and the token usage they report.

Paper extraction and the optional LLM entity alignment call the OpenAI or Anthropic SDK directly,
because they send page images and need JSON mode. The paper summaries use LangChain; their usage
reaches the same UsageLog through a LangChain callback.
"""

from __future__ import annotations

import logging
import os
import threading
from collections import defaultdict
from typing import Any

from bmrag.config import load_environment
from bmrag.events import PRICES

log = logging.getLogger(__name__)


def is_anthropic(model: str) -> bool:
    return model.startswith("claude-")


def openai_client() -> Any:
    load_environment()
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set in the environment or .env")
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def anthropic_client() -> Any:
    load_environment()
    try:
        import anthropic
    except ImportError:
        raise SystemExit("Anthropic models need the anthropic package: pip install anthropic") from None
    key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("ClAUDE_API_KEY")  # noqa: SIM112  (misspelling used by older .env files)
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY is not set in the environment or .env")
    return anthropic.Anthropic(api_key=key)


def chat_kwargs(
    model: str, messages: list[dict], *, max_tokens: int, temperature: float | None, min_gpt5_completion_tokens: int
) -> dict[str, Any]:
    """Chat completion arguments in JSON mode.

    gpt-5 models take max_completion_tokens (raised to at least min_gpt5_completion_tokens, since their
    reasoning tokens count against it) and reject a temperature.
    """
    kwargs: dict[str, Any] = {"model": model, "messages": messages}
    if model.startswith("gpt-5"):
        kwargs["max_completion_tokens"] = max(max_tokens, min_gpt5_completion_tokens)
    else:
        kwargs["max_tokens"] = max_tokens
        if temperature is not None:
            kwargs["temperature"] = temperature
    kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def complete(client: Any, kwargs: dict[str, Any], usage: UsageLog | None = None, stage: str = "") -> Any:
    """Create a chat completion, retrying once per argument a model rejects.

    Models differ in whether they accept max_tokens, temperature and JSON mode; when the error
    names one of them, it is renamed or dropped and the call is repeated, at most three times.
    """
    kwargs = dict(kwargs)
    for _ in range(3):
        try:
            response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            if not _adapt_to_error(kwargs, str(exc).lower()):
                raise
    else:
        response = client.chat.completions.create(**kwargs)
    if usage is not None:
        usage.add_openai(stage, kwargs["model"], getattr(response, "usage", None))
    return response


def _adapt_to_error(kwargs: dict[str, Any], message: str) -> bool:
    changed = False
    if "max_tokens" in kwargs and ("max_tokens" in message or "max_completion_tokens" in message):
        kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        changed = True
    if "temperature" in kwargs and "temperature" in message and "unsupported" in message:
        kwargs.pop("temperature")
        changed = True
    if "response_format" in kwargs and ("response_format" in message or "response format" in message):
        kwargs.pop("response_format")
        changed = True
    return changed


class UsageLog:
    """Token counts per stage and model, safe to update from several threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._chat: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
        )
        self._embeddings: dict[tuple[str, str], dict[str, int]] = defaultdict(
            lambda: {"calls": 0, "texts": 0, "characters": 0}
        )

    def add_chat(
        self,
        stage: str,
        model: str,
        input_tokens: int = 0,
        cached_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        with self._lock:
            row = self._chat[(stage, model)]
            row["calls"] += 1
            row["input_tokens"] += input_tokens or 0
            row["cached_tokens"] += cached_tokens or 0
            row["output_tokens"] += output_tokens or 0
            row["reasoning_tokens"] += reasoning_tokens or 0

    def add_openai(self, stage: str, model: str, usage: Any) -> None:
        if usage is None:
            self.add_chat(stage, model)
            return
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        self.add_chat(
            stage,
            model,
            getattr(usage, "prompt_tokens", 0),
            getattr(prompt_details, "cached_tokens", 0) if prompt_details else 0,
            getattr(usage, "completion_tokens", 0),
            getattr(completion_details, "reasoning_tokens", 0) if completion_details else 0,
        )

    def add_anthropic(self, stage: str, model: str, usage: Any) -> None:
        self.add_chat(
            stage,
            model,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "cache_read_input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0),
        )

    def add_langchain(self, stage: str, usage_by_model: dict[str, Any]) -> None:
        """Totals from langchain_core's UsageMetadataCallbackHandler (one entry per model, calls unknown)."""
        for model, meta in usage_by_model.items():
            details = meta.get("input_token_details") or {}
            out_details = meta.get("output_token_details") or {}
            with self._lock:
                row = self._chat[(stage, model)]
                row["input_tokens"] += meta.get("input_tokens", 0)
                row["cached_tokens"] += details.get("cache_read", 0) or 0
                row["output_tokens"] += meta.get("output_tokens", 0)
                row["reasoning_tokens"] += out_details.get("reasoning", 0) or 0

    def add_langchain_calls(self, stage: str, model: str, calls: int) -> None:
        with self._lock:
            self._chat[(stage, model)]["calls"] += calls

    def add_embeddings(self, stage: str, model: str, texts: list[str]) -> None:
        with self._lock:
            row = self._embeddings[(stage, model)]
            row["calls"] += 1
            row["texts"] += len(texts)
            row["characters"] += sum(len(t) for t in texts)

    def summary(self) -> dict[str, Any]:
        """Usage rows with a list-price estimate. Prices are assumptions (bmrag.events.PRICES), not invoices."""
        with self._lock:
            chat = [
                {"stage": s, "model": m, **row, "estimated_usd": _estimate(m, row)}
                for (s, m), row in sorted(self._chat.items())
            ]
            embeddings = [{"stage": s, "model": m, **row} for (s, m), row in sorted(self._embeddings.items())]
        known = [row["estimated_usd"] for row in chat if row["estimated_usd"] is not None]
        return {
            "chat": chat,
            "embeddings": embeddings,
            "estimated_chat_usd": round(sum(known), 4) if known else None,
            "unpriced_models": sorted({row["model"] for row in chat if row["estimated_usd"] is None}),
            "note": "Embedding tokens are not counted. Prices are assumed list prices.",
        }


def _estimate(model: str, row: dict[str, int]) -> float | None:
    key = next((k for k in sorted(PRICES, key=len, reverse=True) if model.startswith(k)), None)
    if key is None:
        return None
    price_in, price_cached, price_out = PRICES[key]
    fresh = row["input_tokens"] - row["cached_tokens"]
    return round((fresh * price_in + row["cached_tokens"] * price_cached + row["output_tokens"] * price_out) / 1e6, 6)
