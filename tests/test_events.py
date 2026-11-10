"""Tracing: LLM calls are labelled and priced, and concurrent runs do not mix."""

from __future__ import annotations

import threading
import unittest

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from tests import harness  # noqa: F401  (sets the import path)

from bmrag import events, llm


class TracedModel(GenericFakeChatModel):
    pass


def _model(text: str = "reply") -> TracedModel:
    return TracedModel(messages=iter([AIMessage(content=text, usage_metadata={
        "input_tokens": 1000, "output_tokens": 100, "total_tokens": 1100}, response_metadata={"model_name": "gpt-5"})]))


class LLMRecording(unittest.TestCase):
    def test_call_is_labelled_counted_and_priced(self):
        trace = events.Trace()
        with events.capture(trace):
            self.assertEqual(llm.complete(_model(), "hello", caller="reasoner"), "reply")
        event = next(e for e in trace.events if e["kind"] == "llm")
        self.assertEqual((event["caller"], event["model"], event["prompt_chars"]), ("reasoner", "gpt-5", 5))
        self.assertAlmostEqual(event["cost_usd"], (1000 * 1.25 + 100 * 10) / 1e6)
        usage = events.summarize_llm_usage(trace.events)
        self.assertEqual(usage["by_caller"]["reasoner"]["calls"], 1)

    def test_nothing_is_recorded_without_a_trace(self):
        trace = events.Trace()
        llm.complete(_model(), "hello", caller="reasoner")
        events.emit("retrieval", fn="x")
        self.assertEqual(trace.events, [])

    def test_concurrent_runs_keep_separate_traces(self):
        traces = {name: events.Trace() for name in ("a", "b")}
        barrier = threading.Barrier(2)

        def work(name):
            with events.capture(traces[name]), events.agent_span("analyst"):
                barrier.wait()
                llm.complete(_model(), name, caller=name)
                barrier.wait()

        threads = [threading.Thread(target=work, args=(name,)) for name in traces]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        for name, trace in traces.items():
            callers = [e["caller"] for e in trace.events if e["kind"] == "llm"]
            self.assertEqual(callers, [name])


if __name__ == "__main__":
    unittest.main()


class PipelineCommandLine(unittest.TestCase):
    def test_reports_cost_and_saves_usage(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest import mock

        from bmrag import pipeline
        from bmrag.schemas import PipelineResult

        def fake_run(task, profile):
            llm.complete(_model(), "hello", caller="reasoner")  # one priced call inside the trace
            final = {"pcm_interface_pos": 4, "position": "inner interface", "pcm_tm_c": 18.0}
            return PipelineResult(user_input=task, analyst={}, reasoner_attempts=[{}], final_prediction=final,
                                  verdict="pass", elapsed_seconds=1.0)

        with tempfile.TemporaryDirectory() as tmp:
            task, out = Path(tmp) / "task.json", Path(tmp) / "out.json"
            task.write_text("{}", encoding="utf-8")
            argv = ["pipeline", "--task", str(task), "--out", str(out)]
            with mock.patch.object(pipeline, "run", fake_run), mock.patch.object(pipeline, "close_driver"), \
                    mock.patch.object(pipeline, "get_profile"), mock.patch("sys.argv", argv), \
                    mock.patch("builtins.print") as printed:
                pipeline.main()
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertIn("US$0.002", printed.call_args_list[-1].args[0])  # 1000 in + 100 out tokens of gpt-5
        self.assertEqual(report["llm_usage"]["by_caller"]["reasoner"]["calls"], 1)
        self.assertTrue(any(e["kind"] == "llm" for e in report["events"]))
        self.assertEqual(report["final_prediction"]["pcm_tm_c"], 18.0)
