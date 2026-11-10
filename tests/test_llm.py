"""Provider routing: model names reach the right client with the right key. No network calls."""

from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

from tests import harness  # noqa: F401  (sets the import path)

from bmrag import llm


class RecordingClient:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def fake_module(class_name: str) -> types.ModuleType:
    module = types.ModuleType(class_name)
    setattr(module, class_name, RecordingClient)
    return module


def environment(**values: str):
    return mock.patch.dict(os.environ, values, clear=False)


class ProviderRouting(unittest.TestCase):
    def setUp(self):
        openai_module = types.ModuleType("langchain_openai")
        openai_module.ChatOpenAI = RecordingClient
        patcher = mock.patch.dict(sys.modules, {
            "langchain_openai": openai_module,
            "langchain_anthropic": fake_module("ChatAnthropic"),
            "langchain_google_genai": fake_module("ChatGoogleGenerativeAI"),
        })
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_openai_reads_its_own_key_from_the_environment(self):
        for model in ("gpt-5", "gpt-5-mini", "o3-mini"):
            client = llm.create_chat_model(model, temperature=0, max_retries=2)
            self.assertEqual(client.kwargs, {"model": model, "temperature": 0, "max_retries": 2})

    def test_explicit_key_is_passed_to_openai(self):
        self.assertEqual(llm.create_chat_model("gpt-5", api_key="user-key").kwargs["api_key"], "user-key")

    def test_deepseek_uses_the_openai_client_with_its_endpoint(self):
        with environment(DEEPSEEK_API_KEY="deepseek-key"):
            client = llm.create_chat_model("deepseek-chat")
        self.assertEqual(client.kwargs["base_url"], llm.DEEPSEEK_BASE_URL)
        self.assertEqual(client.kwargs["api_key"], "deepseek-key")

    def test_any_openai_compatible_endpoint(self):
        with environment(OPENAI_COMPATIBLE_API_KEY="k", OPENAI_COMPATIBLE_BASE_URL="http://localhost:11434/v1"):
            client = llm.create_chat_model("openai-compatible:qwen3")
        self.assertEqual(client.kwargs["model"], "qwen3")
        self.assertEqual(client.kwargs["base_url"], "http://localhost:11434/v1")

    def test_anthropic_and_gemini(self):
        with environment(ANTHROPIC_API_KEY="a-key", GEMINI_API_KEY="g-key"):
            self.assertEqual(llm.create_chat_model("claude-sonnet-x").kwargs["api_key"], "a-key")
            self.assertEqual(llm.create_chat_model("gemini-x").kwargs["google_api_key"], "g-key")

    def test_missing_key_names_the_variable(self):
        with mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
            with self.assertRaisesRegex(RuntimeError, "ANTHROPIC_API_KEY"):
                llm.create_chat_model("claude-sonnet-x")

    def test_missing_package_says_what_to_install(self):
        with mock.patch.dict(sys.modules, {"langchain_anthropic": None}), environment(ANTHROPIC_API_KEY="a"):
            with self.assertRaisesRegex(RuntimeError, "langchain-anthropic"):
                llm.create_chat_model("claude-sonnet-x")

    def test_unknown_model_is_rejected(self):
        with self.assertRaises(ValueError):
            llm.create_chat_model("mystery-model")


class Replies(unittest.TestCase):
    def test_complete_joins_content_blocks_and_labels_the_call(self):
        seen = {}

        def invoke(prompt, config=None):
            seen.update(config)
            return types.SimpleNamespace(content=[{"type": "text", "text": " a"}, "b "])
        self.assertEqual(llm.complete(types.SimpleNamespace(invoke=invoke), "p", caller="analyst"), "ab")
        self.assertEqual(seen["metadata"]["bmrag_caller"], "analyst")

    def test_strip_code_fence(self):
        self.assertEqual(llm.strip_code_fence('```json\n{"a": 1}\n```'), '{"a": 1}')
        self.assertEqual(llm.strip_code_fence('{"a": 1}'), '{"a": 1}')


if __name__ == "__main__":
    unittest.main()
