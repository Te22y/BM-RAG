"""With the real stores and fake embeddings, prompts must match those recorded from commit 2514c33.

This shows the data moved intact and evidence is packed as before. Needs the local
data/ and runs/ folders, which are not in Git; skips itself without them. Stores
and compares hashes only, so no paper text enters the repository.
"""

from __future__ import annotations

import json
import types
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import scenarios as sc
from tests import harness

from bmrag.agents import reasoner
from bmrag.config import get_paths
from bmrag.profiles import PAPER_V1, STAGE_GRAPH_CLIMATE, STAGE_GRAPH_RULES, STAGE_KG_CASES, STAGE_KG_RULES
from bmrag.schemas import AnalystOutput

RUN_DIR = get_paths().runs_dir / "20261102_005635_full"
LOCAL_BASELINE = harness.GOLDEN / "baseline_local_data.json"
# Paper search is left out: a random query vector has no clear neighbours among 711
# documents, so the approximate search can order ties differently between processes.
KG_STAGES = frozenset({STAGE_KG_CASES, STAGE_KG_RULES, STAGE_GRAPH_CLIMATE, STAGE_GRAPH_RULES})


class HashEmbeddings:
    """Deterministic stand-in for the embedding API; same vectors as the recorder used."""

    def _vector(self, text: str) -> list[float]:
        seed = int(harness.sha(text)[:16], 16) % (2 ** 32)
        vector = np.random.default_rng(seed).normal(size=1536)
        return (vector / np.linalg.norm(vector)).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]


@unittest.skipUnless(RUN_DIR.exists() and get_paths().case_store.exists() and LOCAL_BASELINE.exists(),
                     "needs the local data/ and runs/ folders")
class RealStores(unittest.TestCase):
    def test_prompts_match_recorded_hashes(self):
        recorded = json.loads(LOCAL_BASELINE.read_text(encoding="utf-8"))
        captured: list[dict] = []
        replies = [sc.reasoner_reply(4, 20.0, [])]
        profile = replace(PAPER_V1, retrieval_stages=KG_STAGES)

        def create_chat_model(model, temperature=0.0, max_retries=2, **_):
            def invoke(prompt, *args, **kwargs):
                captured.append({"model": model, "prompt_sha256": harness.sha(prompt), "prompt_chars": len(prompt)})
                return types.SimpleNamespace(content=sc.reply_to(prompt, replies))
            return types.SimpleNamespace(invoke=invoke)

        recorder = harness.Recorder(sc.SCENARIOS["pass_first_attempt"])
        with mock.patch("bmrag.llm.create_embeddings", lambda api_key=None: HashEmbeddings()), \
                mock.patch("bmrag.llm.create_chat_model", create_chat_model), \
                mock.patch("bmrag.retrieval.graph._get_driver", recorder.driver):
            for city, expected in recorded["cities"].items():
                with self.subTest(city):
                    run = json.loads((RUN_DIR / f"{city}.json").read_text(encoding="utf-8"))
                    captured.clear()
                    result = reasoner.reason(AnalystOutput(**run["pipeline_result"]["analyst"]), profile)
                    self.assertEqual(captured, expected["llm_calls"])
                    self.assertEqual(harness.normalized(result.diagnostics["stages"]), expected["stages"])


if __name__ == "__main__":
    unittest.main()
