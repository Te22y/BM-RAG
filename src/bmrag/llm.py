"""Chat models and embeddings for every provider, chosen from the model name.

| Model name                    | Provider          | Key variable             | Extra package          |
|-------------------------------|-------------------|--------------------------|------------------------|
| gpt-*, o1*, o3*, o4*          | OpenAI            | OPENAI_API_KEY           | (base dependency)      |
| deepseek-*                    | DeepSeek          | DEEPSEEK_API_KEY         | (base dependency)      |
| openai-compatible:<model>     | any OpenAI-style  | OPENAI_COMPATIBLE_API_KEY, OPENAI_COMPATIBLE_BASE_URL | (base dependency) |
| claude-*                      | Anthropic         | ANTHROPIC_API_KEY        | langchain-anthropic    |
| gemini-*                      | Google            | GEMINI_API_KEY           | langchain-google-genai |
| meta-llama/*, *llama*         | Hugging Face      | HUGGINGFACEHUB_API_TOKEN | langchain-huggingface  |

An explicit api_key takes priority over the environment. Embeddings stay on OpenAI
whatever the chat provider is: querying a vector store needs the model that built it.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from bmrag.config import EMBEDDING_MODEL, load_environment

OPENAI_PREFIXES = ("gpt-", "o1", "o3", "o4")
COMPATIBLE_PREFIX = "openai-compatible:"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _require_key(api_key: Optional[str], variable: str) -> str:
    key = api_key or os.environ.get(variable)
    if not key:
        raise RuntimeError(f"{variable} is not set; add it to .env or pass api_key")
    return key


def _import_chat_class(module: str, class_name: str, package: str) -> Any:
    try:
        return getattr(__import__(module, fromlist=[class_name]), class_name)
    except ImportError as exc:
        raise RuntimeError(f"This model needs the {package} package: pip install {package}") from exc


def _openai_style(model: str, temperature: float, max_retries: int, **client: Any) -> Any:
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model, temperature=temperature, max_retries=max_retries, **client)


def _huggingface(model: str, temperature: float, api_key: Optional[str], max_new_tokens: int) -> Any:
    endpoint_class = _import_chat_class("langchain_huggingface", "HuggingFaceEndpoint", "langchain-huggingface")
    chat_class = _import_chat_class("langchain_huggingface", "ChatHuggingFace", "langchain-huggingface")
    endpoint = endpoint_class(
        repo_id=model,
        task="text-generation",
        temperature=max(temperature, 0.01),   # the endpoint rejects exactly 0
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=_require_key(api_key, "HUGGINGFACEHUB_API_TOKEN"),
    )
    return chat_class(llm=endpoint)


def create_chat_model(model: str, temperature: float = 0.0, max_retries: int = 2, *,
                      api_key: Optional[str] = None, max_new_tokens: int = 4096) -> Any:
    """A LangChain chat model for this name. max_new_tokens applies to Hugging Face only."""
    load_environment()
    if model.startswith(OPENAI_PREFIXES):
        client = {"api_key": api_key} if api_key else {}
        return _openai_style(model, temperature, max_retries, **client)
    if model.startswith("deepseek-"):
        return _openai_style(model, temperature, max_retries, base_url=DEEPSEEK_BASE_URL,
                             api_key=_require_key(api_key, "DEEPSEEK_API_KEY"))
    if model.startswith(COMPATIBLE_PREFIX):
        return _openai_style(model.removeprefix(COMPATIBLE_PREFIX), temperature, max_retries,
                             base_url=_require_key(None, "OPENAI_COMPATIBLE_BASE_URL"),
                             api_key=_require_key(api_key, "OPENAI_COMPATIBLE_API_KEY"))
    if model.startswith("claude-"):
        chat_class = _import_chat_class("langchain_anthropic", "ChatAnthropic", "langchain-anthropic")
        return chat_class(model=model, temperature=temperature, max_retries=max_retries,
                          api_key=_require_key(api_key, "ANTHROPIC_API_KEY"))
    if model.startswith("gemini-"):
        chat_class = _import_chat_class("langchain_google_genai", "ChatGoogleGenerativeAI", "langchain-google-genai")
        return chat_class(model=model, temperature=temperature, max_retries=max_retries,
                          google_api_key=_require_key(api_key, "GEMINI_API_KEY"))
    if "llama" in model.lower():
        return _huggingface(model, temperature, api_key, max_new_tokens)
    raise ValueError(f"Unknown LLM model: {model!r}")


def create_embeddings(api_key: Optional[str] = None) -> Any:
    load_environment()
    from langchain_openai import OpenAIEmbeddings
    client = {"api_key": api_key} if api_key else {}
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, **client)


def complete(chat_model: Any, prompt: str, caller: str) -> str:
    """Run one prompt and return the reply text. `caller` labels the call in traces."""
    reply = chat_model.invoke(prompt, config={"metadata": {"bmrag_caller": caller}})
    content = reply.content
    if isinstance(content, list):   # some providers return a list of content blocks
        content = "".join(block if isinstance(block, str) else block.get("text", "") for block in content)
    return content.strip()


def strip_code_fence(reply: str) -> str:
    """Remove a markdown code fence that a model may wrap around JSON."""
    if not reply.startswith("```"):
        return reply
    body = reply.split("```", 2)[1]
    if body.lstrip().lower().startswith("json"):
        body = body.split("\n", 1)[1]
    return body.rsplit("```", 1)[0].strip()
