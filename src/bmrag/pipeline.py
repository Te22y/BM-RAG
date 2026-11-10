"""The pipeline: Analyst, then Reasoner and Judge in a retry loop.

Command line (from the repo root, one run costs about US$0.15-0.40):
  python -m bmrag.pipeline --task task.json --out result.json
It prints the estimated cost; --out also saves token usage per caller and model, and the event trace.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional

from bmrag import events
from bmrag.agents.analyst import analyze, passthrough
from bmrag.agents.judge import judge
from bmrag.agents.reasoner import reason
from bmrag.profiles import PROFILES, Profile, get_profile
from bmrag.retrieval.graph import close_driver
from bmrag.schemas import AnalystInput, AnalystOutput, JudgeOutput, PipelineResult, ReasonerOutput

FINAL_FIELDS = ("position", "pcm_interface_pos", "pcm_tm_c", "tm_reasoning", "joint_rationale", "confidence",
                "reasoning_chain", "evidence_ids", "alternatives")
DEFAULT_RETRY_HINT = "Re-examine evidence and tighten the chain."


def _unchecked_verdict() -> JudgeOutput:
    return JudgeOutput(verdict="pass", fatal_issues=[], soft_issues=[], evidence_audit={}, rubric_scores={},
                       overall_score=1.0, retry_hint=None)


def _retry_instruction(hint: Optional[str], previous: ReasonerOutput) -> str:
    return ("PREVIOUS ATTEMPT WAS REJECTED BY THE JUDGE.\n"
            f"Judge hint: {hint or DEFAULT_RETRY_HINT}\n"
            "Previous prediction (for reference, do NOT just repeat it):\n"
            f"{json.dumps(previous.to_dict(), ensure_ascii=False, indent=2)}\n"
            "Address the hint specifically. Improve mechanism clarity and "
            "ensure every citation appears in the evidence bundle.")


def _run_analyst(task: AnalystInput, profile: Profile) -> AnalystOutput:
    with events.agent_span("analyst"):
        return analyze(task, profile) if profile.use_analyst else passthrough(task)


def _run_reasoner(analyst_out: AnalystOutput, profile: Profile,
                  previous: Optional[tuple[ReasonerOutput, JudgeOutput]]) -> ReasonerOutput:
    with events.agent_span("reasoner") as span:
        if previous is None:
            result = reason(analyst_out, profile)
        else:
            result = reason(analyst_out, profile, profile.retry_top_k_cases, profile.retry_top_k_rules,
                            _retry_instruction(previous[1].retry_hint, previous[0]))
        span.summary = {"pos": result.pcm_interface_pos, "tm": result.pcm_tm_c, "confidence": result.confidence}
    return result


def _run_judge(analyst_out: AnalystOutput, prediction: ReasonerOutput, profile: Profile) -> JudgeOutput:
    with events.agent_span("judge") as span:
        result = judge(analyst_out, prediction) if profile.use_judge else _unchecked_verdict()
        span.summary = {"verdict": result.verdict, "n_fatal": len(result.fatal_issues),
                        "n_soft": len(result.soft_issues)}
    return result


def run(task: AnalystInput | dict[str, Any], profile: Optional[Profile] = None) -> PipelineResult:
    """Run one design task. When retries run out, the last attempt is returned as it is."""
    started = time.time()
    profile = profile or get_profile()
    if isinstance(task, dict):
        task = AnalystInput(**task)

    analyst_out = _run_analyst(task, profile)
    attempts: list[dict[str, Any]] = []
    previous: Optional[tuple[ReasonerOutput, JudgeOutput]] = None
    for attempt in range(profile.max_retries + 1):
        prediction = _run_reasoner(analyst_out, profile, previous)
        verdict = _run_judge(analyst_out, prediction, profile)
        attempts.append({"attempt": attempt + 1, "reasoner": prediction.to_dict(), "judge": verdict.to_dict()})
        previous = (prediction, verdict)
        if verdict.verdict == "pass":
            break

    final = attempts[-1]
    return PipelineResult(
        user_input=task.to_dict(),
        analyst=analyst_out.to_dict(),
        reasoner_attempts=attempts,
        final_prediction={name: final["reasoner"][name] for name in FINAL_FIELDS},
        verdict=final["judge"]["verdict"],
        elapsed_seconds=round(time.time() - started, 2),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one BM-RAG design task")
    parser.add_argument("--task", type=Path, required=True, help="JSON file with the AnalystInput fields")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="paper_v1")
    parser.add_argument("--out", type=Path, help="write the full result to this JSON file")
    args = parser.parse_args()

    trace = events.Trace()
    with events.capture(trace):
        result = run(json.loads(args.task.read_text(encoding="utf-8")), get_profile(args.profile))
    close_driver()
    usage = events.summarize_llm_usage(trace.events)
    final = result.final_prediction
    print(f"position {final['pcm_interface_pos']} ({final['position']}), melting temperature {final['pcm_tm_c']} C")
    print(f"verdict {result.verdict}, {len(result.reasoner_attempts)} attempt(s), {result.elapsed_seconds} s")
    calls = sum(totals["calls"] for totals in usage["by_model"].values())
    print(f"{calls} LLM call(s), about US${usage['cost_usd_total']:.3f} at list prices (embeddings not counted)")
    if args.out:
        # the result plus what the web demo records: token usage and cost, and the event trace
        report = {**result.to_dict(), "llm_usage": usage, "events": trace.events}
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
