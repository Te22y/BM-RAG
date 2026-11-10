"""Build web/demo_samples/*.json: invented example runs for the public demo.

Nothing here comes from a paper or a real run. The real pipeline code runs on invented
evidence with a scripted LLM, so the page shows the true report format. Tasks, evidence,
reasoning, timings, tokens, the optimum and the energy figures are all made up, and the
page labels these runs as synthetic. No API call is made.

Usage (from the repo root):
  python web/build_demo_samples.py
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from typing import Any, Callable
from unittest import mock

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from report import build_report  # noqa: E402

from bmrag import events, pipeline  # noqa: E402
from bmrag.agents.reasoner import MECHANISM_CASE_QUERY, MECHANISM_RULE_QUERY  # noqa: E402
from bmrag.evaluation.ground_truth import score  # noqa: E402
from bmrag.profiles import get_profile  # noqa: E402

OUTPUT = HERE / "demo_samples"
PAPER_ELEMENT = "0123456789abcdef__text__{:04d}"
BUILDING = ("small office, exterior wall; HVAC dual-setpoint thermostat - occupied: cooling setpoint = 24°C, "
            "heating setpoint = 21°C; setback: cooling = 29°C, heating = 16°C.")
WALL = "stucco 25 mm | gypsum 16 mm | insulation R=1.6 | gypsum 16 mm (interior)"
PCM = "organic PCM, latent heat=200 kJ/kg, thickness=20 mm, phase-transition band = 4°C"
# Invented durations per kind of step, in seconds
STEP_SECONDS = {"analyst": 20.0, "reasoner": 70.0, "paper_rerank": 2.0, "retrieval": 0.3, "default": 0.1}
OUTPUT_TOKENS = {"analyst": 1800, "reasoner": 7000, "paper_rerank": 100}


def _case(index: int, known_optimum: bool) -> dict[str, Any]:
    return {
        "case_id": f"case_demo_{index}", "score": round(0.66 - index * 0.02, 3),
        "source_file": f"Invented paper {'ABC'[index % 3]}.pdf",
        "case_profile": {
            "case_summary": f"Invented case {index}: a simulated office wall where an inner PCM layer lowered annual load.",
            "conditional_context": {
                "climate": f"Invented climate {index}", "wall_assembly": "Invented four-layer wall",
                "pcm_position": "interior", "optimal_pcm_position": "interior" if known_optimum else None,
                "performance_metrics": f"Invented saving of {index + 1} percent",
            },
            "raw_case": {},
        },
        "internal_kg": {"relations_text": [f"Case {index} --FAVORS--> interior position"]},
    }


def _rule(index: int) -> dict[str, Any]:
    return {
        "rule_id": f"rule_demo_{index}", "score": round(0.62 - index * 0.02, 3),
        "source_file": f"Invented paper {'ABC'[index % 3]}.pdf",
        "rule_profile": {
            "case_summary": f"Invented rule context {index}.",
            "kg_rule": {"relation": "FAVORS", "subject": "PCM layer", "object": "interior position",
                        "condition": "invented condition",
                        "evidence_sentence": f"Invented statement {index}: PCM closer to the room couples better to it."},
        },
        "internal_kg": {"relations_text": [f"Rule {index} --FAVORS--> interior position"]},
    }


def _reply(pos: int, tm: float, position: str, cited: list[str], confidence: float, why: str) -> str:
    return json.dumps({
        "position": position, "pcm_interface_pos": pos, "pcm_tm_c": tm,
        "tm_reasoning": [f"Invented reasoning: the interface temperature crosses {tm} °C during a typical day."],
        "joint_rationale": f"Invented example, not a real result. {why}",
        "reasoning_chain": [f"Invented step: {cited[0]} has a similar wall, so its mechanism applies.",
                            f"Invented step: {cited[-1]} points to the same coupling argument."],
        "evidence_ids": cited, "confidence": confidence,
        "alternatives": [{"pcm_interface_pos": 2, "pcm_tm_c": 28.0, "evidence_ids": [],
                          "why_not_preferred": "Invented reason: insulation separates this position from the room."}],
    })


ANALYST_REPLY = json.dumps({
    "climate_query": "invented temperate climate", "construction_query": "stucco gypsum insulation gypsum wall",
    "pcm_query": "organic PCM latent heat 200 kJ/kg thickness 20 mm",
    "composite_query": "invented temperate climate, four-layer wall, organic PCM, minimize annual energy",
    "materials": ["stucco", "gypsum", "insulation"], "objective_kw": "energy saving", "climate_kw": "invented",
    "pcm_tm_c": None, "pcm_thickness_mm": 20, "wall_layer_count": 4,
    "notes": ["Invented note: this run is a synthetic example."],
})

SAMPLES = {
    "sample-city-a": {
        "title": "Sample City A (synthetic)", "cases_with_optimum": 8,
        "replies": [_reply(4, 20.0, "interior side of the insulation", ["case_demo_0", "rule_demo_0"], 0.7,
                           "The inner position couples to the room and still cycles daily.")],
        "truth": {"gt_pos": 4, "gt_tm": 18.0, "gt_savings_pct": 3.0, "baseline_total_kWh": 10000.0,
                  "gt_total_kWh": 9700.0, "tied_tm_set": [18.0]},
        "cells": {(4, 20.0): 9712.0},
    },
    "sample-city-b": {
        "title": "Sample City B (synthetic)", "cases_with_optimum": 2,
        "replies": [_reply(3, 26.0, "midpoint of the insulation", ["case_demo_1"], 0.5,
                           "A first estimate favors the middle of the insulation."),
                    _reply(3, 24.0, "midpoint of the insulation", ["case_demo_1", "rule_demo_1"], 0.55,
                           "The retry keeps the position and lowers the melting temperature."),
                    _reply(3, 24.0, "midpoint of the insulation",
                           ["case_demo_1", "rule_demo_1", PAPER_ELEMENT.format(1)], 0.55,
                           "The Judge's reminder is addressed, and the last attempt is accepted.")],
        "truth": {"gt_pos": 4, "gt_tm": 22.0, "gt_savings_pct": 4.0, "baseline_total_kWh": 10000.0,
                  "gt_total_kWh": 9600.0, "tied_tm_set": [22.0]},
        "cells": {(3, 24.0): 9780.0},
    },
}


class _ScriptedChatModel(BaseChatModel):
    """A LangChain chat model that answers from a script, so tracing sees ordinary LLM calls."""

    model_label: str
    script: Callable[[str], tuple[str, str]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(self, messages: list[Any], stop: Any = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        prompt = messages[0].content
        text, caller = self.script(prompt)
        input_tokens, output_tokens = len(prompt) // 4, OUTPUT_TOKENS[caller]
        message = AIMessage(content=text, response_metadata={"model_name": self.model_label},
                            usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens,
                                            "total_tokens": input_tokens + output_tokens})
        return ChatResult(generations=[ChatGeneration(message=message)])


def _script(replies: list[str]) -> Callable[[str], tuple[str, str]]:
    remaining = list(replies)

    def answer(prompt: str) -> tuple[str, str]:
        if prompt.startswith("You are the ANALYST"):
            return ANALYST_REPLY, "analyst"
        if prompt.startswith("You are an expert relevance"):
            return json.dumps([{"index": i, "score": round(0.9 - i * 0.1, 1)} for i in range(3)]), "paper_rerank"
        return (remaining.pop(0) if len(remaining) > 1 else remaining[0]), "reasoner"
    return answer


def _graph_session(rows_for: Callable[[str, dict[str, Any]], list[dict[str, Any]]]) -> Any:
    session = mock.MagicMock()
    session.__enter__.return_value = session
    session.run.side_effect = lambda cypher, **params: rows_for(cypher, params)
    return types.SimpleNamespace(session=lambda database=None: session)


def _graph_rows(cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    if "c.case_id IN $case_ids" in cypher:
        return [{"case_id": case_id, "paper": "Invented paper", "favored_positions": ["interior side"],
                 "tested_positions": ["exterior", "interior"], "climate": "Invented climate",
                 "supporting_rules": []} for case_id in params["case_ids"]]
    return []


def _fix_timings(trace_events: list[dict[str, Any]]) -> float:
    """Replace measured times with invented ones so the output is reproducible."""
    clock = 0.0
    started: dict[Any, float] = {}
    for event in trace_events:
        kind = event["kind"]
        if kind == "llm":
            seconds = STEP_SECONDS[event["caller"]]
        else:
            seconds = STEP_SECONDS["retrieval"] if kind == "retrieval" else STEP_SECONDS["default"]
        clock = round(clock + seconds, 1)
        event["t"] = clock
        if kind == "agent_start":
            started[event["agent"]] = clock
        elif kind == "agent_end":
            event["elapsed_sec"] = round(clock - started[event["agent"]], 1)
        elif "elapsed_sec" in event:
            event["elapsed_sec"] = seconds
    return clock


def _build(sample_id: str, sample: dict[str, Any]) -> dict[str, Any]:
    cases = [_case(i, i < sample["cases_with_optimum"]) for i in range(8)]
    rules = [_rule(i) for i in range(6)]

    def search_cases(query: str, top_k: int) -> list[dict[str, Any]]:
        return cases[-top_k:] if query == MECHANISM_CASE_QUERY else cases[:top_k]

    def search_rules(query: str, top_k: int) -> list[dict[str, Any]]:
        return rules[-top_k:] if query == MECHANISM_RULE_QUERY else rules[:top_k]

    def paper_candidates() -> Any:
        documents = [Document(page_content="Invented element.",
                              metadata={"doc_id": PAPER_ELEMENT.format(i), "element_type": "text",
                                        "source_file": "Invented paper C.pdf"}) for i in range(1, 4)]
        return types.SimpleNamespace(invoke=lambda query: documents)

    script = _script(sample["replies"])
    task = {"climate": f"{sample['title'].split(' (')[0]}, an invented temperate climate", "building_type": BUILDING,
            "wall_construction": WALL, "pcm_parameters": PCM, "goal": "minimize total annual energy"}
    trace = events.Trace()
    with mock.patch("bmrag.llm.create_chat_model",
                    lambda model, *a, **k: _ScriptedChatModel(model_label=model, script=script)), \
            mock.patch("bmrag.retrieval.kg_index.search_case_bundles", search_cases), \
            mock.patch("bmrag.retrieval.kg_index.search_rule_bundles", search_rules), \
            mock.patch("bmrag.retrieval.kg_index.load_rule_store", lambda: {r["rule_id"]: r for r in rules}), \
            mock.patch("bmrag.retrieval.graph._get_driver", lambda: _graph_session(_graph_rows)), \
            mock.patch("bmrag.retrieval.papers._get_retriever", paper_candidates), \
            mock.patch("bmrag.retrieval.papers._load_summary", lambda root, doc_id: f"Invented summary of {doc_id}."), \
            events.capture(trace):
        result = pipeline.run(task, get_profile("paper_v1", analyst_model="gpt-5-mini", reasoner_model="gpt-5",
                                                paper_rerank_model="gpt-5.2")).to_dict()

    result["elapsed_seconds"] = _fix_timings(trace.events)
    meta = {"id": sample_id, "title": sample["title"], "zone": None, "source": "synthetic", "synthetic": True}
    stores = {"case_store.json": {c["case_id"]: c for c in cases}, "rule_store.json": {r["rule_id"]: r for r in rules}}
    with mock.patch("report._load_store", stores.__getitem__):
        report = build_report(meta, result, trace.events, events.summarize_llm_usage(trace.events), None)

    truth = sample["truth"]
    final = result["final_prediction"]
    report["score"] = score(sample_id, final["pcm_interface_pos"], final["pcm_tm_c"], {sample_id: truth},
                            {sample_id: sample["cells"]}, {})
    report["truth"] = {"gt_pos": truth["gt_pos"], "gt_tm": truth["gt_tm"], "tied_tm_set": truth["tied_tm_set"],
                       "savings_pct": truth["gt_savings_pct"], "baseline_kWh": truth["baseline_total_kWh"],
                       "optimum_kWh": truth["gt_total_kWh"]}
    return report


def build_all() -> dict[str, dict[str, Any]]:
    return {sample_id: _build(sample_id, sample) for sample_id, sample in SAMPLES.items()}


def render(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, indent=1) + "\n"


def main() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for sample_id, report in build_all().items():
        (OUTPUT / f"{sample_id}.json").write_text(render(report), encoding="utf-8", newline="\n")
        print(f"{sample_id}: {len(report['attempts'])} attempt(s), capture loss {report['score']['capture_loss']:.3f}")


if __name__ == "__main__":
    main()
