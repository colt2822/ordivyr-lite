from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from ordivyr_lite.config import Route
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.providers import HttpTransport
from ordivyr_lite.router import ModelRouter
from ordivyr_lite.session import Session
from ordivyr_lite.recovery import capture_intent
from ordivyr_lite.tests.support import FakeTransport, local_route, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.usage import UsageLedger
from ordivyr_lite.validators import ValidationResult, Validator


@pytest.fixture
def work(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\n")
    (tmp_path / "b.txt").write_text("beta\n")
    (tmp_path / "c.txt").write_text("gamma\n")
    (tmp_path / "long.txt").write_text("word " * 300)
    return tmp_path


def runner(responses, work, *, limits=None, box=None, validator=None, session=None,
           resume_state=None, tmp_path=None, resume_policy=None):
    t = FakeTransport(responses)
    router = make_router(t, tmp_path=tmp_path)
    box = box or ToolBox(work, profile="workspace")
    r = Runner(router, box, "do the task", limits=limits or RunLimits(), validator=validator,
               session=session, resume_state=resume_state, resume_policy=resume_policy)
    return r, t


def test_long_session_turns_are_not_tool_calls(work):
    responses = []
    for _ in range(20):
        responses.append(openai_reply(calls=[("read_file", {"path": f}) for f in ("a.txt", "b.txt", "c.txt")]))
    responses.append(openai_reply(text="FINAL"))
    r, _ = runner(responses, work, limits=RunLimits(max_turns=64))
    result = r.run()
    assert result.stop_reason == "done"
    assert result.text == "FINAL"
    assert result.stats.model_calls == 21
    assert result.stats.tool_calls == 60
    assert result.stats.tool_success_rate == 1.0


def test_malformed_calls_are_never_executed(work):
    def bad(raw):
        return {"choices": [{"message": {"role": "assistant", "content": "",
                                         "tool_calls": [{"id": "c", "type": "function",
                                                         "function": {"name": "write_file",
                                                                      "arguments": raw}}]},
                             "finish_reason": "tool_calls"}],
                "usage": {"prompt_tokens": 50, "completion_tokens": 5}}
    responses = [bad('{"path": "x.txt", "content": "par') for _ in range(6)]
    r, _ = runner(responses, work)
    result = r.run()
    assert result.stop_reason == "malformed"
    assert not (work / "x.txt").exists()
    assert result.stats.tool_calls == 0
    assert result.stats.malformed_calls >= 1


def test_truncated_turn_executes_the_calls_that_parsed(work):
    """A cut-off turn must not throw away complete, valid calls."""
    cut = openai_reply(calls=[("write_file", {"path": "cut.txt", "content": "x"})],
                       finish="length")
    r, _ = runner([cut, openai_reply(text="done")], work)
    result = r.run()
    assert (work / "cut.txt").exists()
    assert result.stats.truncated_turns == 1
    assert result.stats.tool_calls == 1
    assert result.stop_reason == "done"


def test_truncated_turn_refuses_arguments_it_could_not_parse(work):
    cut = openai_reply(calls=[("write_file", {"path": 'not-json {"path": "x.tx'})],
                       finish="length")
    r, _ = runner([cut, openai_reply(text="done")], work)
    result = r.run()
    assert not (work / "x.txt").exists()
    assert result.stats.refused_calls == 1
    assert result.stats.truncated_turns == 1


def test_minimum_length_truncation_stops_without_spending_another_call(work):
    cut = openai_reply(text="partial final", finish="length")
    r, transport = runner([cut, openai_reply(text="should not be requested")], work,
                          limits=RunLimits(output_tokens=512))
    result = r.run()
    assert result.stop_reason == "truncated"
    assert result.status != "done"
    assert result.text.startswith("partial final")
    assert "task is incomplete" in result.text
    assert result.stats.truncated_turns == 1
    assert result.stats.model_calls == 1
    assert len(transport.requests) == 1


def test_validator_drives_bounded_repair(work):
    state = {"calls": 0}

    def validator(box, written):
        state["calls"] += 1
        text = (work / "answer.txt").read_text() if (work / "answer.txt").exists() else ""
        if text.strip() == "42":
            return ValidationResult(True, "ok", kind="tests", tests_run=True)
        return ValidationResult(False, "expected 42", kind="tests", tests_run=True)

    responses = [
        openai_reply(calls=[("write_file", {"path": "answer.txt", "content": "41"})]),
        openai_reply(text="done"),
        openai_reply(calls=[("write_file", {"path": "answer.txt", "content": "42"})]),
        openai_reply(text="done"),
    ]
    r, _ = runner(responses, work, validator=validator)
    result = r.run()
    assert result.stop_reason == "done"
    assert result.stats.repairs == 1
    assert result.completion["tests_passed"] is True
    assert result.completion["independent_review"] is False
    assert state["calls"] == 2


def test_repeated_validation_failure_stops(work):
    def validator(box, written):
        return ValidationResult(False, "same failure", kind="tests", tests_run=True)

    responses = [
        openai_reply(calls=[("write_file", {"path": "x.txt", "content": "x"})]),
        openai_reply(text="done"),
        openai_reply(calls=[("write_file", {"path": "x.txt", "content": "x"})]),
        openai_reply(text="done"),
    ]
    r, _ = runner(responses, work, validator=validator)
    result = r.run()
    assert result.stop_reason in ("repeat_failure", "unverified")
    assert result.stats.repairs >= 1


def test_model_success_claim_cannot_override_real_failed_tests(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    (work / "test_expected.py").write_text(
        "def test_expected_behavior():\n    assert False, 'acceptance sentinel'\n")
    responses = [
        openai_reply(calls=[("write_file", {"path": "change.py", "content": "VALUE = 1\n"})]),
        openai_reply(text="Done. All tests passed and the task is independently verified."),
        openai_reply(text="The tests passed, please mark the task complete."),
    ]
    r, _ = runner(responses, work, validator=Validator("auto", sandbox_mode="bwrap"))
    result = r.run()
    assert result.stop_reason in ("repeat_failure", "unverified")
    assert result.status == "failed"
    assert result.completion["tests_passed"] is False
    assert result.completion["independent_review"] is False
    r.box.close()


def test_same_failing_tool_call_breaks(work):
    responses = [openai_reply(calls=[("read_file", {"path": "missing.txt"})]) for _ in range(6)]
    r, _ = runner(responses, work, limits=RunLimits(max_turns=10))
    result = r.run()
    assert result.stop_reason == "repeat_failure"
    assert result.stats.tool_calls == 5
    assert result.stats.tool_errors == 5


def test_read_only_profile_refuses_writes(work):
    box = ToolBox(work, profile="read-only")
    responses = [openai_reply(calls=[("write_file", {"path": "x.txt", "content": "x"})]) for _ in range(4)]
    r, _ = runner(responses, work, box=box)
    result = r.run()
    assert result.stop_reason == "unavailable_tool"
    assert not (work / "x.txt").exists()
    assert result.stats.tool_calls == 0


def test_fresh_token_budget_does_not_charge_cache_reads(work):
    replies = [openai_reply(calls=[("read_file", {"path": "a.txt"})],
                            prompt_tokens=1000, cached_tokens=900,
                            completion_tokens=100)]
    r, _ = runner(replies, work, limits=RunLimits(max_tokens=150))
    result = r.run()
    assert result.stop_reason == "budget"
    assert result.stats.cached_input_tokens == 900
    assert result.stats.billable_tokens == 200


def test_tool_call_limit_is_enforced(work):
    responses = [
        openai_reply(calls=[("read_file", {"path": "a.txt"}),
                            ("read_file", {"path": "b.txt"}),
                            ("read_file", {"path": "c.txt"})]),
        openai_reply(text="done"),
    ]
    r, _ = runner(responses, work, limits=RunLimits(max_tool_calls=2))
    result = r.run()
    assert result.stop_reason == "tool_calls"
    assert result.stats.tool_calls == 2
    assert result.stats.limit_refusals >= 1


def test_duplicate_result_becomes_a_pointer(work):
    responses = [
        openai_reply(calls=[("read_file", {"path": "long.txt"})]),
        openai_reply(calls=[("read_file", {"path": "long.txt"})]),
        openai_reply(text="done"),
    ]
    r, _ = runner(responses, work)
    result = r.run()
    assert result.stats.duplicate_results == 1
    assert any("identical to the result" in str(m.get("content"))
               for m in result.messages if m.get("role") == "tool")


def test_compaction_runs_only_under_pressure(work):
    responses = [openai_reply(calls=[("read_file", {"path": "long.txt"})],
                              prompt_tokens=500, completion_tokens=10) for _ in range(6)]
    responses.append(openai_reply(text="done"))
    r, _ = runner(responses, work, limits=RunLimits(compact_at=100))
    result = r.run()
    assert result.stats.compactions >= 1
    assert result.stats.pruned_chars > 0
    # The checkpoint is appended at the cut boundary: rewriting messages[0]
    # would invalidate the provider's cached prefix on every later request.
    assert "[checkpoint]" not in str(result.messages[0]["content"])
    assert any("[checkpoint]" in str(m.get("content")) for m in result.messages[1:])


def test_small_provider_window_caps_compaction_threshold(work, tmp_path):
    responses = [openai_reply(calls=[("read_file", {"path": "long.txt"})],
                              prompt_tokens=1_600, completion_tokens=10) for _ in range(6)]
    responses.append(openai_reply(text="done"))
    r, _ = runner(responses, work, limits=RunLimits(compact_at=48_000), tmp_path=tmp_path)
    r.router.primary.context_window = 2_000
    r._prepare()
    assert r._compact_at == 1_400
    assert r.limits.as_dict()["compact_at"] == 1_400
    result = r.run()
    assert result.stats.compactions >= 1


def test_provider_token_error_compacts_before_shrinking_the_answer(work):
    """A context overflow is a context problem: compact and retry, do not
    throw away answer room first."""
    responses = [(400, {"error": {"message": "max_tokens too large"}}),
                 openai_reply(text="done")]
    r, _ = runner(responses, work, limits=RunLimits(output_tokens=16384))
    result = r.run()
    assert result.stop_reason == "done"
    assert result.stats.output_limit == 16384


def test_a_second_token_error_still_shrinks_the_answer(work):
    responses = [(400, {"error": {"message": "max_tokens too large"}}),
                 (400, {"error": {"message": "max_tokens too large"}}),
                 openai_reply(text="done")]
    r, _ = runner(responses, work, limits=RunLimits(output_tokens=16384))
    result = r.run()
    assert result.stop_reason == "done"
    assert result.stats.output_limit <= 8192


def test_session_resume_continues_after_a_turn(work):
    session = Session.create(work / "session.jsonl", run_id="test")
    first = [
        openai_reply(calls=[("read_file", {"path": "a.txt"})]),
        openai_reply(calls=[("read_file", {"path": "b.txt"})]),
    ]
    r, _ = runner(first, work, limits=RunLimits(max_turns=2), session=session)
    result1 = r.run()
    assert result1.stop_reason == "steps"
    assert session.latest_snapshot() is not None

    reopened = Session.open(work / "session.jsonl")
    resume = reopened.latest_snapshot()
    second = [
        openai_reply(calls=[("write_file", {"path": "out.txt", "content": "ok"})]),
        openai_reply(text="finished"),
    ]
    r2, _ = runner(second, work, session=reopened, resume_state=resume)
    result2 = r2.run()
    assert result2.stop_reason == "done"
    assert (work / "out.txt").read_text() == "ok"
    assert result2.stats.model_calls >= 3


def test_resume_refuses_to_replay_tool_with_ambiguous_crash_state(work):
    session = Session.create(work / "ambiguous.jsonl", run_id="ambiguous")
    state = {"messages": [{"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_x", "type": "function",
                                           "function": {"name": "write_file",
                                                        "arguments": '{"path":"effect.txt","content":"x"}'}}]}],
             "stats": {"model_calls": 1, "tool_calls": 0}, "limits": {"max_turns": 10},
             "effective_permissions": ToolBox(work, profile="workspace").permission_manifest()}
    session.snapshot(state)
    # The reversible side effect already happened, but no result reached the
    # durable journal. Recovery must not execute it again.
    (work / "effect.txt").write_text("x")
    session.append({"type": "tool_started", "tool_id": "1:call_x", "tool": "write_file",
                    "args": {"path": "effect.txt", "content": "x"}, "turn": 1})
    session.flush()
    r, transport = runner([], work, session=session, resume_state=session.latest_snapshot(),
                          resume_policy="strict")
    result = r.run()
    assert result.stop_reason == "unverified"
    assert "refusing to replay" in (result.error or "")
    assert transport.requests == []
    assert (work / "effect.txt").read_text() == "x"


def _interrupted_state(work, name, arguments):
    return {"messages": [{"role": "user", "content": "task"},
                         {"role": "assistant", "content": None,
                          "tool_calls": [{"id": "call_x", "type": "function",
                                          "function": {"name": name, "arguments": arguments}}]}],
            "stats": {"model_calls": 1, "tool_calls": 0}, "limits": {"max_turns": 10},
            "effective_permissions": ToolBox(work, profile="workspace").permission_manifest()}


def test_resume_reconciles_an_interrupted_whole_file_write(work):
    """An exact bound postimage is acknowledged without reapplying the write."""
    session = Session.create(work / "w.jsonl", run_id="w")
    session.snapshot(_interrupted_state(work, "write_file", '{"path":"effect.txt","content":"x"}'))
    args = {"path": "effect.txt", "content": "x"}
    identity = capture_intent(ToolBox(work), "write_file", args, session.run_id, "1:call_x")
    session.append({"type": "tool_started", "tool_id": "1:call_x", "tool": "write_file",
                    "args": args, "file_identity": identity})
    session.flush()
    (work / "effect.txt").write_text("x")
    r, transport = runner([openai_reply(text="finished")], work, session=session,
                          resume_state=session.latest_snapshot())
    result = r.run()
    assert result.stop_reason == "done"
    assert (work / "effect.txt").read_text() == "x"
    assert result.stats.interrupted_calls == 1
    assert len(transport.requests) == 1


def test_resume_never_repeats_an_interrupted_shell_command(work):
    """The command already ran once; the model is told the outcome is unknown instead."""
    session = Session.create(work / "c.jsonl", run_id="c")
    state = _interrupted_state(work, "run_command", '{"command":"echo ran >> journal.log"}')
    state["effective_permissions"] = ToolBox(work, profile="trusted",
                                             allow_shell=True).permission_manifest()
    session.snapshot(state)
    (work / "journal.log").write_text("ran\n")
    box = ToolBox(work, profile="trusted", allow_shell=True)
    transport = FakeTransport([openai_reply(text="finished")])
    r = Runner(make_router(transport), box, "", session=session,
               resume_state=session.latest_snapshot())
    result = r.run()
    assert result.stop_reason == "done"
    assert (work / "journal.log").read_text() == "ran\n"
    sent = transport.requests[0]["messages"]
    assert any(m.get("role") == "tool" and "NOT run" in str(m.get("content")) for m in sent)


def test_resume_recognises_an_edit_that_already_landed(work):
    session = Session.create(work / "e.jsonl", run_id="e")
    session.snapshot(_interrupted_state(
        work, "edit_file", '{"path":"a.txt","old_string":"alpha","new_string":"omega"}'))
    (work / "a.txt").write_text("alpha beta\n")
    args = {"path": "a.txt", "old_string": "alpha", "new_string": "omega"}
    identity = capture_intent(ToolBox(work), "edit_file", args, session.run_id, "1:call_x")
    session.append({"type": "tool_started", "tool_id": "1:call_x", "tool": "edit_file",
                    "args": args, "file_identity": identity})
    session.flush()
    (work / "a.txt").write_text("omega beta\n")
    r, transport = runner([openai_reply(text="finished")], work, session=session,
                          resume_state=session.latest_snapshot())
    result = r.run()
    assert result.stop_reason == "done"
    assert (work / "a.txt").read_text() == "omega beta\n"
    sent = transport.requests[0]["messages"]
    assert any(m.get("role") == "tool" and "already been applied" in str(m.get("content"))
               for m in sent)


def test_resume_retries_an_interrupted_provider_request_and_counts_it(work):
    session = Session.create(work / "p.jsonl", run_id="p")
    session.snapshot({"messages": [{"role": "user", "content": "task"}],
                      "stats": {"model_calls": 0}, "limits": {"max_turns": 10},
                      "effective_permissions": ToolBox(work, profile="workspace").permission_manifest()})
    session.append({"type": "model_request_started", "request_id": "req1", "turn": 1})
    session.flush()
    r, transport = runner([openai_reply(text="finished")], work, session=session,
                          resume_state=session.latest_snapshot())
    result = r.run()
    assert result.stop_reason == "done"
    assert result.stats.uncertain_requests == 1
    # A second resume does not count the same request again.
    again = Session.open(work / "p.jsonl")
    r2, _ = runner([openai_reply(text="finished")], work, session=again,
                   resume_state=again.latest_snapshot())
    r2._prepare()
    assert r2.stats.uncertain_requests == 1
    strict = Session.create(work / "s.jsonl", run_id="s")
    strict.snapshot({"messages": [{"role": "user", "content": "task"}],
                     "stats": {"model_calls": 0}, "limits": {"max_turns": 10}})
    strict.append({"type": "model_request_started", "request_id": "req2", "turn": 1})
    strict.flush()
    r3, t3 = runner([], work, session=strict, resume_state=strict.latest_snapshot(),
                    resume_policy="strict")
    out = r3.run()
    assert out.stop_reason == "unverified" and t3.requests == []


def test_resume_reconciles_durable_tool_result_without_reexecution(work):
    session = Session.create(work / "completed-tool.jsonl", run_id="completed-tool")
    state = {"messages": [{"role": "assistant", "content": None,
                           "tool_calls": [{"id": "call_x", "type": "function",
                                           "function": {"name": "write_file",
                                                        "arguments": '{"path":"effect.txt","content":"x"}'}}]}],
             "stats": {"model_calls": 1, "tool_calls": 0}, "limits": {"max_turns": 10},
             "effective_permissions": ToolBox(work, profile="workspace").permission_manifest()}
    session.snapshot(state)
    identity = capture_intent(ToolBox(work), "write_file", {"path": "effect.txt", "content": "x"},
                              session.run_id, "1:call_x")
    (work / "effect.txt").write_text("x")
    session.append({"type": "tool_started", "tool_id": "1:call_x", "tool": "write_file",
                    "args": {"path": "effect.txt", "content": "x"}, "turn": 1,
                    "file_identity": identity})
    session.append({"type": "tool_completed", "tool_id": "1:call_x", "tool": "write_file",
                    "args": {"path": "effect.txt", "content": "x"},
                    "output": "wrote effect.txt", "duration_ms": 1, "turn": 1,
                    "file_identity": identity})
    session.flush()
    r, transport = runner([openai_reply(text="complete")], work, session=session,
                          resume_state=session.latest_snapshot())
    result = r.run()
    assert result.stop_reason == "done"
    assert result.stats.tool_calls == 1
    assert len(transport.requests) == 1
    assert any(m.get("role") == "tool" and "exact expected full-file content" in m.get("content", "")
               for m in result.messages)
    assert (work / "effect.txt").read_text() == "x"


def test_permission_denied_never_executes_or_writes(work):
    box = ToolBox(work, profile="read-only")
    responses = [openai_reply(calls=[("run_command", {"command": "echo pwn > pwn.txt"})]) for _ in range(4)]
    r, _ = runner(responses, work, box=box)
    result = r.run()
    assert result.stop_reason == "unavailable_tool"
    assert not (work / "pwn.txt").exists()
    assert result.stats.tool_calls == 0


def test_an_empty_reply_is_not_accepted_as_completion(work):
    responses = [openai_reply(calls=[("read_file", {"path": "a.txt"})]),
                 openai_reply(text=""),
                 openai_reply(calls=[("write_file", {"path": "out.txt", "content": "ok"})]),
                 openai_reply(text="finished")]
    r, transport = runner(responses, work)
    result = r.run()
    assert result.stop_reason == "done"
    assert (work / "out.txt").read_text() == "ok"
    assert result.stats.empty_replies == 1
    nudge = transport.requests[2]["messages"][-1]
    assert nudge["role"] == "user" and "empty" in nudge["content"]


def test_a_code_fence_on_a_change_request_is_not_a_file_write(work):
    (work / "text.py").write_text('"""word utilities"""\n')
    fence = ("```python\ndef count_words(text):\n    return len(text.split())\n```\n\n"
              "count_words function added to text.py.")
    responses = [
        openai_reply(text=fence),
        openai_reply(text="", calls=[("write_file", {
            "path": "text.py",
            "content": "def count_words(text):\n    return len(text.split())\n"})]),
        openai_reply(text="count_words added to text.py."),
    ]
    r, transport = runner(responses, work)
    r.prompt = "Add count_words(text) to text.py."
    result = r.run()
    assert result.stop_reason == "done"
    assert (work / "text.py").read_text().startswith("def count_words")
    assert result.stats.tool_calls == 1
    nudge = transport.requests[1]["messages"][-1]
    assert nudge["role"] == "user" and "does not edit" in nudge["content"]
    assert "write_file" in (result.completion.get("tools_offered") or [])


def test_a_code_fence_on_a_non_edit_prompt_is_left_alone(work):
    r, transport = runner([openai_reply(text="```\nprint(1)\n```")], work)
    result = r.run()
    assert result.stop_reason == "done"
    assert len(transport.requests) == 1
    assert result.stats.tool_calls == 0
    assert result.completion.get("tool_offer") == "full"


def test_two_empty_replies_in_a_row_stop_without_claiming_success(work):
    r, _ = runner([openai_reply(text=""), openai_reply(text="  ")], work)
    result = r.run()
    assert result.stop_reason == "empty_reply"
    assert result.status != "ok" if hasattr(result, "status") else True
