"""Efficiency regressions: repeated reads, validation reuse, token allocation,
task-aware compression and concurrency-safe spending.

Every assertion is tied to a measured counter, not to a code path.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from ordivyr_lite import context as CTX
from ordivyr_lite.config import Route
from ordivyr_lite.errors import BudgetError
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.providers import HttpTransport
from ordivyr_lite.router import ModelRouter
from ordivyr_lite.tests.support import FakeTransport, local_route, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.usage import UsageLedger, estimate_cost

REPO = Path(__file__).resolve().parents[2]


def paid_route(**caps) -> Route:
    return Route(provider="p", protocol="openai", base_url="https://api.example.test/v1",
                 model="m", api_key="k", paid=True, authorized=True,
                 price=(1.0, 2.0, 0.1, 1.0),
                 caps={"per_request_usd": 10.0, "daily_usd": 10.0, "total_usd": 10.0, **caps})


def runner_for(tmp_path, responses, *, limits=None, boxes=None, validator=None):
    box = ToolBox(tmp_path, profile="workspace")
    router = make_router(FakeTransport(responses), tmp_path=tmp_path)
    runner = Runner(router, box, "do the task", limits=limits or RunLimits(max_turns=8),
                    validator=validator)
    return runner, box


def test_reservation_blocks_the_second_paid_call(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    route = paid_route(total_usd=1.0, daily_usd=1.0)
    first = ledger.reserve(run_id="r1", route=route, estimate_usd=0.6)
    assert first and first.startswith("res_")
    with pytest.raises(BudgetError):
        ledger.reserve(run_id="r2", route=route, estimate_usd=0.6)
    ledger.release(first)
    assert ledger.spent() == 0.0
    assert ledger.reserve(run_id="r3", route=route, estimate_usd=0.6)


def test_settlement_replaces_the_reservation_with_the_real_cost(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    route = paid_route()
    rid = ledger.reserve(run_id="r1", route=route, estimate_usd=2.0)
    assert ledger.spent() == 2.0
    ledger.record(run_id="r1", route=route, usage={"input": 1000, "output": 1000},
                  cost_usd=0.003, latency_ms=12.0, reserve_id=rid)
    assert ledger.spent() == pytest.approx(0.003)
    assert ledger.reserve(run_id="r2", route=route, estimate_usd=9.0)


def test_an_expired_reservation_stops_counting(tmp_path):
    path = tmp_path / "usage.jsonl"
    ledger = UsageLedger(path)
    route = paid_route(total_usd=1.0, daily_usd=1.0)
    old = {"ts": 1.0, "provider": "p", "model": "m", "status": "reserved",
           "reserve_id": "res_old", "usd": 5.0, "cost_usd": 5.0}
    path.write_text(json.dumps(old) + "\n", "utf-8")
    assert ledger.spent() == 0.0


def test_two_processes_cannot_both_pass_the_cap(tmp_path):
    ledger_path = tmp_path / "usage.jsonl"
    script = tmp_path / "reserver.py"
    script.write_text(textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(REPO)!r})
        from ordivyr_lite.config import Route
        from ordivyr_lite.errors import BudgetError
        from ordivyr_lite.usage import UsageLedger
        route = Route(provider="p", protocol="openai", base_url="https://x.test/v1", model="m",
                      api_key="k", paid=True, authorized=True, price=(1.0, 2.0, 0.1, 1.0),
                      caps={{"per_request_usd": 10.0, "daily_usd": 1.0, "total_usd": 1.0}})
        ledger = UsageLedger({str(ledger_path)!r})
        try:
            rid = ledger.reserve(run_id=sys.argv[1], route=route, estimate_usd=0.75)
        except BudgetError:
            print("deny")
        else:
            print("ok" if rid else "free")
    """), "utf-8")
    procs = [subprocess.Popen([sys.executable, str(script), "r" + str(i)],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(6)]
    outs = [p.communicate(timeout=60)[0].strip() for p in procs]
    assert outs.count("ok") == 1, outs
    assert outs.count("deny") == 5, outs


def test_cap_estimate_takes_the_stricter_of_two_estimators():
    route = paid_route()
    messages = [{"role": "user", "content": "x" * 4000}]
    est = estimate_cost(route, messages, 1000)
    strict = (4000 / 3.5 * route.price[0] + 1000 * route.price[1]) / 1_000_000
    assert est >= round(strict, 10) - 1e-9


def test_a_repeated_read_is_served_without_touching_the_tool(tmp_path):
    (tmp_path / "a.txt").write_text("hello world" + "\n" * 400, "utf-8")
    responses = [
        openai_reply(calls=[("read_file", {"path": "a.txt"})]),
        openai_reply(calls=[("read_file", {"path": "a.txt"})]),
        openai_reply(text="done"),
    ]
    runner, box = runner_for(tmp_path, responses)
    result = runner.run()
    assert result.stats.reused_reads == 1
    assert box.stats.calls == 1, "the repeated read must not touch the filesystem tool"
    contents = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert len(contents) == 2 and "identical to the result" in contents[1]


def test_a_changed_file_is_read_again(tmp_path):
    (tmp_path / "a.txt").write_text("one" + "\n", "utf-8")
    responses = [
        openai_reply(calls=[("read_file", {"path": "a.txt"})]),
        openai_reply(calls=[("write_file", {"path": "a.txt", "content": "two" + "\n"})]),
        openai_reply(calls=[("read_file", {"path": "a.txt"})]),
        openai_reply(text="done"),
    ]
    runner, box = runner_for(tmp_path, responses)
    result = runner.run()
    assert result.stats.reused_reads == 0
    contents = [m["content"] for m in result.messages if m.get("role") == "tool"]
    assert "two" in contents[-1]


def test_identical_validation_is_not_run_twice(tmp_path):
    calls = {"n": 0}

    class V:
        kind = "tests"
        tests_run = True
        ok = False
        report = "assert 1 == 2"
        command = "pytest"

        def __call__(self, box, wrote):
            calls["n"] += 1
            return self

        def as_dict(self):
            return {"ok": self.ok, "kind": self.kind, "report": self.report}

    responses = [openai_reply(text="done"), openai_reply(text="done")]
    runner, box = runner_for(tmp_path, responses, validator=V())
    result = runner.run()
    assert calls["n"] == 1
    assert result.stats.skipped_validations == 1
    assert result.stop_reason == "repeat_failure"


def test_a_cut_off_answer_is_retried_with_a_smaller_budget(tmp_path):
    responses = [
        openai_reply(text="a" * 50, finish="length"),
        openai_reply(text="short answer"),
    ]
    runner, box = runner_for(tmp_path, responses, limits=RunLimits(max_turns=4))
    result = runner.run()
    assert result.stop_reason == "done"
    assert result.stats.model_calls == 2
    assert result.stats.output_limit == 8192


def test_output_allocation_respects_the_context_window(tmp_path):
    transport = FakeTransport([openai_reply(text="done")])
    route = local_route()
    route.context_window = 2000
    router = ModelRouter([route], ledger=UsageLedger(tmp_path / "usage.jsonl"),
                         transport=HttpTransport(http=transport))
    box = ToolBox(tmp_path, profile="workspace")
    Runner(router, box, "task", limits=RunLimits(max_turns=2, output_tokens=16384)).run()
    assert transport.requests[0]["max_tokens"] <= 2000 - 256


def test_sandbox_session_is_opened_once_and_closed(tmp_path, monkeypatch):
    from ordivyr_lite import tools as T

    opened = {"n": 0, "closed": 0, "runs": 0}

    class FakeSession:
        def run(self, command, timeout=60):
            opened["runs"] += 1

            class R:
                code, timed_out, isolated = 0, False, True
                out = "ok"
                err = ""

                def text(self):
                    return "ok"
            return R()

        def close(self):
            opened["closed"] += 1

    def fake_session(*a, **k):
        opened["n"] += 1
        return FakeSession()

    monkeypatch.setattr(T.SB, "session", fake_session)
    box = ToolBox(tmp_path, profile="trusted", sandbox_mode="bwrap")
    for _ in range(3):
        assert box.call("run_command", {"command": "echo ok"}) == "ok"
    assert opened["n"] == 1 and opened["runs"] == 3
    box.close()
    assert opened["closed"] == 1


def test_a_small_tree_never_builds_a_search_index(tmp_path, monkeypatch):
    from ordivyr_lite import memory as MEM

    def boom(*a, **k):
        raise AssertionError("the index must not be built for a small tree")

    for i in range(3):
        (tmp_path / ("f%d.py" % i)).write_text("x = 1" + "\n", "utf-8")
    monkeypatch.setattr(MEM, "CodeIndex", boom)
    box = ToolBox(tmp_path, profile="workspace")
    runner = Runner(make_router(FakeTransport([openai_reply(text="done")]), tmp_path=tmp_path),
                    box, "task")
    assert runner._context_hint().count("Likely relevant files") == 0


def test_a_reservation_is_released_when_the_paid_call_fails(tmp_path):
    from ordivyr_lite.errors import ProviderError
    from ordivyr_lite.router import ModelRouter as MR

    ledger = UsageLedger(tmp_path / "usage.jsonl")
    route = paid_route()
    route.max_retries = 0
    transport = FakeTransport([(500, {"error": {"message": "boom"}})])
    router = MR([route], ledger=ledger, transport=HttpTransport(http=transport))
    with pytest.raises(ProviderError):
        router.chat([{"role": "user", "content": "x"}], max_tokens=10, run_id="r1")
    assert ledger.spent() == 0.0
    assert any(r.get("status") == "settled" for r in ledger.records())


def test_compression_keeps_the_result_the_task_is_about():
    messages = [{"role": "system", "content": "sys"},
                {"role": "user", "content": "fix the invoice vat total"}]
    for i in range(10):
        body = ("invoice totals need a vat fix " if i == 0 else "unrelated colours palette ")
        body = body * 48
        messages.append({"role": "tool", "tool_call_id": "c" + str(i), "content": body})
    messages.append({"role": "assistant", "content": "done"})
    relevant_before = messages[2]["content"]
    stale_before = messages[4]["content"]
    cut, compacted = CTX.maybe_compact(messages, 10_000, turns=6, trail=[], errors=[],
                                       threshold=100, query="fix the invoice vat total")
    assert compacted and cut > 0
    assert messages[2]["content"] == relevant_before, "the task-relevant result must survive"
    assert messages[4]["content"] != stale_before, "an irrelevant old result is elided"


def test_compression_without_a_query_falls_back_to_recency():
    messages = [{"role": "system", "content": "s"}]
    for i in range(6):
        messages.append({"role": "tool", "tool_call_id": str(i), "content": "line" * 200})
    cut, compacted = CTX.maybe_compact(messages, 10_000, turns=6, trail=[], errors=[],
                                       threshold=100)
    assert compacted and cut > 0
    tools_msgs = [m for m in messages if m.get("role") == "tool"]
    assert tools_msgs[-1]["content"] == "line" * 200, "the tail is never elided"
    assert messages[0]["content"] == "s", "the head must never be rewritten"
    assert any(str(m.get("content", "")).startswith(CTX.CKPT) for m in messages[1:]), \
        "the checkpoint is appended at the cut boundary"
