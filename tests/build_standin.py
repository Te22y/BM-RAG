"""Run a build command with stand-ins for the OpenAI chat and embedding APIs.

Usage: python tests/build_standin.py REQUESTS_JSONL REPLIES_JSON -m MODULE [ARGS...]

Every request is appended to REQUESTS_JSONL as a digest of its arguments, so two code
versions can be compared without storing any paper text. Replies are deterministic:
- paper extraction requests get the reply that REPLIES_JSON gives for the PDF named in the request;
- entity alignment requests get an empty mapping;
- any other chat request (element summaries) gets a summary derived from the request digest;
- embeddings are unit vectors derived from the input text.
Nothing reaches the network: the patched methods never call the real client.
"""

from __future__ import annotations

import hashlib
import json
import os
import runpy
import sys
import threading

import openai.resources.chat.completions as chat
import openai.resources.embeddings as embeddings
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.completion_usage import CompletionUsage
from openai.types.create_embedding_response import Usage

EMBEDDING_DIMENSIONS = 16
_lock = threading.Lock()


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def _text_of(messages) -> str:
    return json.dumps(messages, ensure_ascii=False, default=str)


class StandIn:
    def __init__(self, requests_path: str, replies: dict):
        self.requests_path = requests_path
        self.extraction_replies: dict[str, str] = replies.get("extraction", {})

    def record(self, kind: str, kwargs: dict) -> str:
        key = digest(kwargs)
        with _lock, open(self.requests_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": kind, "model": kwargs.get("model"), "digest": key}) + "\n")
        return key

    def chat_reply(self, kwargs: dict) -> ChatCompletion:
        key = self.record("chat", kwargs)
        text = _text_of(kwargs.get("messages"))
        if "extract the paper-level record" in text:
            matches = [reply for name, reply in self.extraction_replies.items() if name in text]
            if len(matches) != 1:
                raise RuntimeError("stand-in: no single extraction reply matches this request")
            content = matches[0]
        elif "You align entity names" in text:
            content = json.dumps({"mapping": {}, "notes": []})
        else:
            content = f"Stand-in summary {key[:12]}."
        message = ChatCompletionMessage(role="assistant", content=content)
        return ChatCompletion(
            id=f"chatcmpl-{key[:12]}",
            object="chat.completion",
            created=0,
            model=kwargs.get("model") or "stand-in",
            choices=[Choice(index=0, finish_reason="stop", message=message)],
            usage=CompletionUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )

    def embedding_reply(self, kwargs: dict) -> CreateEmbeddingResponse:
        self.record("embedding", kwargs)
        inputs = kwargs.get("input")
        inputs = inputs if isinstance(inputs, list) else [inputs]
        data = [Embedding(object="embedding", index=i, embedding=_unit_vector(item)) for i, item in enumerate(inputs)]
        return CreateEmbeddingResponse(
            object="list",
            model=kwargs.get("model") or "stand-in",
            data=data,
            usage=Usage(prompt_tokens=0, total_tokens=0),
        )


def _unit_vector(item) -> list[float]:
    raw = hashlib.sha256(json.dumps(item, default=str).encode()).digest()
    values = [b - 127.5 for b in raw[:EMBEDDING_DIMENSIONS]]
    norm = sum(v * v for v in values) ** 0.5
    return [v / norm for v in values]


class RawResponse:
    """What client.with_raw_response.create returns: LangChain calls .parse() on it."""

    def __init__(self, parsed):
        self._parsed = parsed
        self.headers: dict[str, str] = {}

    def parse(self):
        return self._parsed


def _maybe_raw(kwargs: dict, parsed):
    headers = kwargs.get("extra_headers") or {}
    return RawResponse(parsed) if headers.get("X-Stainless-Raw-Response") else parsed


def install(stand_in: StandIn) -> None:
    def chat_create(self, *args, **kwargs):
        return _maybe_raw(kwargs, stand_in.chat_reply(kwargs))

    async def chat_create_async(self, *args, **kwargs):
        return _maybe_raw(kwargs, stand_in.chat_reply(kwargs))

    def embedding_create(self, *args, **kwargs):
        return _maybe_raw(kwargs, stand_in.embedding_reply(kwargs))

    async def embedding_create_async(self, *args, **kwargs):
        return _maybe_raw(kwargs, stand_in.embedding_reply(kwargs))

    chat.Completions.create = chat_create
    chat.AsyncCompletions.create = chat_create_async
    embeddings.Embeddings.create = embedding_create
    embeddings.AsyncEmbeddings.create = embedding_create_async


def main() -> None:
    requests_path, replies_path, flag, module, *args = sys.argv[1:]
    if flag != "-m":
        raise SystemExit(__doc__)
    with open(replies_path, encoding="utf-8") as f:
        install(StandIn(requests_path, json.load(f)))
    os.environ.setdefault("OPENAI_API_KEY", "stand-in")
    sys.argv = [module, *args]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
