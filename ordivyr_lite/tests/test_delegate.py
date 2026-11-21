"""Bounded delegation: isolated child context, no nesting, a hard child budget."""

from __future__ import annotations

import threading
import time

from ordivyr_lite.delegate import DelegateLimits, Delegator
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox


def _call(name: str, args: dict, call_id: str = "call_0") -> dict:
    return openai_reply(text="", calls=[(name, args)])


def test_child_context_is_isolated_and_cannot_delegate(tmp_path):
    transport = FakeTransport([
        _call("delegate", {"task": "Reply with the single word alpha."}),
        openai_reply(text="alpha", prompt_tokens=40, completion_tokens=2),
        openai_reply(text="parent-done", prompt_tokens=80, completion_tokens=3),
    ])
    box = ToolBox(tmp_path, profile="workspace")
    router = make_router(transport, tmp_path=tmp_path)
    result = Runner(router, box, "Split the work.",
                    limits=RunLimits(max_turns=4, delegate=True, delegate_turns=2)).run()
    assert result.stop_reason == "done"
    assert result.text == "parent-done"
    assert "delegate" in box.names()
    child = transport.requests[1]
    parent_follow = transport.requests[2]
    assert all(m.get("content") != "Split the work." for m in child["messages"])
    child_user = "\n".join(str(m.get("content") or "") for m in child["messages"])
    assert "Task contract:" in child_user
    assert "Budget:" in child_user
    assert "Reply with the single word alpha." in child_user
    child_tools = [t["function"]["name"] for t in child.get("tools") or []]
    assert "delegate" not in child_tools
    blob = json_messages(parent_follow)
    assert "alpha" in blob
    assert result.stats.input_tokens >= 40
    assert result.stats.model_calls >= 3


def test_child_budget_stops_a_second_child(tmp_path):
    transport = FakeTransport([
        openai_reply(text="", calls=[
            ("delegate", {"task": "one"}),
            ("delegate", {"task": "two"}),
        ]),
        openai_reply(text="only-one"),
        openai_reply(text="parent-done"),
    ])
    # The scripted reply above uses one combined call list via openai_reply's calls
    # parameter, but the helper here passed calls incorrectly. Rebuild.
    transport.responses = [
        openai_reply(text="", calls=[("delegate", {"task": "one"}), ("delegate", {"task": "two"})]),
        openai_reply(text="only-one"),
        openai_reply(text="parent-done"),
    ]
    box = ToolBox(tmp_path, profile="read-only")
    router = make_router(transport, tmp_path=tmp_path)
    result = Runner(router, box, "Two children.",
                    limits=RunLimits(max_turns=3, delegate=True, delegate_children=1,
                                     delegate_turns=1)).run()
    assert result.stop_reason == "done"
    rendered = json_messages(transport.requests[-1])
    assert "only-one" in rendered
    assert "child budget exhausted" in rendered


def test_read_only_batch_runs_concurrently():
    current = {"n": 0}
    lock = threading.Lock()
    seen = []

    class Child:
        def __init__(self, task):
            self.task = task

        def run(self):
            with lock:
                current["n"] += 1
                seen.append(current["n"])
            time.sleep(0.05)
            with lock:
                current["n"] -= 1

            class Result:
                text = self.task
                stop_reason = "done"
                error = None
                artifacts = []

                class stats:
                    input_tokens = 1
                    cached_input_tokens = 0
                    output_tokens = 1
                    model_calls = 1
                    tool_calls = 0
                    tool_errors = 0

            return Result()

    delegator = Delegator(
        spawn=lambda task, allow: Child(task),
        limits=DelegateLimits(max_children=4, max_parallel=2))
    reports = delegator.run_many([("a", ["read_file"]), ("b", ["read_file"])])
    assert [r.text for r in reports] == ["a", "b"]
    assert max(seen) == 2


def test_a_duplicate_task_in_one_batch_runs_once():
    calls = []

    class Child:
        def __init__(self, task):
            calls.append(task)

        def run(self):
            class Result:
                text = "once"
                stop_reason = "done"
                error = None
                artifacts = ["a.txt"]

                class stats:
                    input_tokens = 5
                    cached_input_tokens = 1
                    output_tokens = 1
                    model_calls = 1
                    tool_calls = 0
                    tool_errors = 0

            return Result()

    delegator = Delegator(spawn=lambda task, allow: Child(task),
                          limits=DelegateLimits(max_children=4, max_parallel=2))
    reports = delegator.run_many([("same", ["read_file"]), ("same", ["read_file"])])
    assert calls == ["same"]
    assert reports[0].stop_reason == "done"
    assert reports[0].input_tokens == 5
    assert reports[1].stop_reason == "duplicate"
    assert reports[1].input_tokens == 0
    assert "not run again" in reports[1].text


def json_messages(payload: dict) -> str:
    return "\n".join(str(m.get("content") or "") for m in payload["messages"])
