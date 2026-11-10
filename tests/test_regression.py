"""Behaviour must match the baseline recorded from commit 2514c33 (the code behind the paper).

Compares every prompt hash, LLM setting, Cypher query, retrieval query and final
result. If one of these fails, the change altered `paper_v1` behaviour.
"""

from __future__ import annotations

import dataclasses
import unittest

import scenarios as sc
from tests import harness

from bmrag import events, pipeline
from bmrag.agents import reasoner
from bmrag.evaluation import lookup_scoring
from bmrag.profiles import get_profile
from bmrag.prompts import load_prompt
from bmrag.retrieval import graph


def _strip_timing(trace_events: list[dict]) -> list[dict]:
    return harness.normalized([{k: v for k, v in e.items() if k not in ("t", "elapsed_sec")} for e in trace_events])


def _without_elapsed(result: dict) -> dict:
    return {k: v for k, v in result.items() if k != "elapsed_seconds"}


class PromptsAndTemplates(unittest.TestCase):
    def test_prompt_files_match_recorded_hashes(self):
        import common
        import runners
        recorded = harness.BASELINE["constants"]["texts"]
        current = {
            "analyst": load_prompt("analyst"),
            "reasoner": load_prompt("reasoner"),
            "case_query_rewrite": load_prompt("case_query_rewrite"),
            "paper_query": load_prompt("paper_query"),
            "paper_rerank": load_prompt("paper_rerank"),
            "ablation_task_spec": common.TASK_OUTPUT_SPEC,
            "ablation_only_llm": runners.ONLY_LLM_PROMPT,
            "ablation_naive_rag": runners.NAIVE_RAG_PROMPT,
            "ablation_graph_rag": runners.GRAPH_RAG_PROMPT,
            "position_semantics": reasoner.POSITION_SEMANTICS,
            "mechanism_case_query": reasoner.MECHANISM_CASE_QUERY,
            "mechanism_rule_query": reasoner.MECHANISM_RULE_QUERY,
        }
        self.assertEqual({k: harness.sha(v) for k, v in current.items()}, recorded)

    def test_cypher_templates_match(self):
        recorded = harness.BASELINE["constants"]
        self.assertEqual({k: harness.sha(v) for k, v in graph.CYPHER_TEMPLATES.items()}, recorded["cypher"])
        self.assertEqual(harness.normalized(graph._TEMPLATE_DEFAULTS), recorded["template_defaults"])


class Pipeline(unittest.TestCase):
    def test_every_scenario_matches_baseline(self):
        for name, scenario in sc.SCENARIOS.items():
            with self.subTest(name), harness.fake_backends(scenario) as recorder:
                overrides = {"max_retries": scenario["max_retries"]}
                if scenario["sources"]:
                    overrides["retrieval_stages"] = frozenset(scenario["sources"])
                result = pipeline.run(dict(sc.TASK), get_profile("paper_v1", **overrides))
                recorded = harness.BASELINE["pipeline"][name]
                self.assertEqual(harness.normalized(recorder.llm_calls), recorded["llm_calls"])
                self.assertEqual(harness.normalized(recorder.graph_calls), recorded["graph_calls"])
                self.assertEqual(recorder.searches, recorded["searches"])
                self.assertEqual(harness.normalized(_without_elapsed(result.to_dict())), recorded["result"])

    def test_trace_events_match_baseline(self):
        for name in ("pass_first_attempt", "soft_retry_exhausted", "rewrite_and_all_stages"):
            with self.subTest(name), harness.fake_backends(sc.SCENARIOS[name]):
                trace = events.Trace()
                with events.capture(trace):
                    pipeline.run(dict(sc.TASK), get_profile())
                self.assertEqual(_strip_timing(trace.events), harness.BASELINE["events"][name])

    def test_scan_mode_matches_baseline(self):
        import trace_pipeline
        recorded = harness.BASELINE["events"]["scan"]
        with harness.fake_backends(sc.SCENARIOS["rewrite_and_all_stages"]):
            scan = trace_pipeline.run_scan({"city": "testville", "zone": "synthetic", "analyst_input": dict(sc.TASK)})
        self.assertEqual(_strip_timing(scan["events"]), recorded["events"])
        self.assertEqual(scan["stages_attempt1"], recorded["stages_attempt1"])
        self.assertEqual(scan["stages_retry"], recorded["stages_retry"])
        self.assertEqual({k: v.get("n", "error") for k, v in scan["template_check"].items()},
                         recorded["template_check"])


class Experiments(unittest.TestCase):
    def test_every_ablation_experiment_matches_baseline(self):
        import runners
        record = {"city": "testville", "zone": "synthetic", "analyst_input": dict(sc.TASK)}
        for name, recorded in harness.BASELINE["experiments"].items():
            with self.subTest(name), harness.fake_backends(sc.SCENARIOS["rewrite_and_all_stages"]) as recorder:
                result = runners.EXPERIMENTS[name](dict(record))
                result.get("raw_pipeline", {}).pop("elapsed_sec", None)
                # The recorder could not see the baselines' max_retries in the old code: it was fixed at 2.
                expected_calls = [{**c, "max_retries": c["max_retries"] if c["max_retries"] is not None else 2}
                                  for c in recorded["llm_calls"]]
                self.assertEqual(harness.normalized(recorder.llm_calls), expected_calls)
                self.assertEqual(harness.normalized(recorder.graph_calls), recorded["graph_calls"])
                self.assertEqual(recorder.searches, recorded["searches"])
                self.assertEqual(harness.normalized(result), recorded["result"])


class Scoring(unittest.TestCase):
    def test_lookup_scoring_matches_baseline(self):
        import tempfile
        from pathlib import Path
        recorded = harness.BASELINE["scoring"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "summary.csv"
            path.write_text(sc.SYNTHETIC_LOOKUP_CSV, encoding="utf-8")
            lookup = lookup_scoring.build_lookup(path)
        scored = [lookup_scoring.score_one("alpha", pos, tm, lookup) for pos, tm in sc.SCORING_CASES]
        self.assertEqual(harness.normalized(scored), recorded["scored"])
        block = dict(lookup["alpha"])
        grid = {f"{k[0]}/{k[1]}": v for k, v in block.pop("grid").items()}
        self.assertEqual(harness.normalized({**block, "grid": grid}), recorded["lookup"])


class AcceptableThreshold(unittest.TestCase):
    """The synthetic scoring cases do not straddle this threshold, so it gets its own check."""

    def test_capture_loss_of_ten_percent_is_the_limit(self):
        block = {"gt_pos": 4, "gt_tm": 18.0, "gt_savings_pct": 10.0, "gt_total_kWh": 900.0, "tied_tm_set": [18.0],
                 "grid": {(3, 20.0): {"total_kWh": 901.0, "pct_savings": 9.05},
                          (3, 22.0): {"total_kWh": 902.0, "pct_savings": 8.95}}}
        lookup = {"alpha": block}
        self.assertTrue(lookup_scoring.score_one("alpha", 3, 20.0, lookup)["is_acceptable"])
        self.assertFalse(lookup_scoring.score_one("alpha", 3, 22.0, lookup)["is_acceptable"])


class Profiles(unittest.TestCase):
    def test_paper_profile_is_what_the_paper_used(self):
        profile = dataclasses.asdict(get_profile("paper_v1", analyst_model="gpt-5-mini", reasoner_model="gpt-5",
                                                 paper_rerank_model="gpt-5.2"))
        self.assertEqual({k: profile[k] for k in ("max_retries", "first_top_k_cases", "first_top_k_rules",
                                                  "retry_top_k_cases", "retry_top_k_rules", "sufficient_cases",
                                                  "sufficient_rules", "weak_top1_score", "weak_score_spread",
                                                  "paper_top_k", "context_paper_query", "use_analyst", "use_judge")},
                         {"max_retries": 2, "first_top_k_cases": 5, "first_top_k_rules": 3, "retry_top_k_cases": 6,
                          "retry_top_k_rules": 4, "sufficient_cases": 6, "sufficient_rules": 4,
                          "weak_top1_score": 0.45, "weak_score_spread": 0.02, "paper_top_k": 3,
                          "context_paper_query": False, "use_analyst": True, "use_judge": True})


if __name__ == "__main__":
    unittest.main()
