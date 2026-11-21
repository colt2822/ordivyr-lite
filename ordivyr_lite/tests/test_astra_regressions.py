"""Audit reproductions using injected transports, never model inference."""
import pytest
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply, local_route
from ordivyr_lite.tests.test_efficiency import paid_route
from ordivyr_lite.errors import BudgetError


def run(tmp_path, responses, **kwargs):
    router = make_router(FakeTransport(responses), tmp_path=tmp_path)
    return Runner(router, ToolBox(tmp_path, profile="workspace"), "inspect and fix", **kwargs).run()


def test_read_after_write_sees_new_content(tmp_path):
    (tmp_path / "a").write_text("old")
    result = run(tmp_path, [openai_reply(calls=[
        ("read_file", {"path": "a"}),
        ("write_file", {"path": "a", "content": "new"}),
        ("read_file", {"path": "a"}),
    ]), openai_reply(text="done")])
    outputs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert "old" in outputs[0] and "new" in outputs[2]


def test_long_repeated_errors_trip_breaker(tmp_path, monkeypatch):
    monkeypatch.setattr(ToolBox, "call", lambda *a: "ERROR: " + "failure " * 100)
    result = run(tmp_path, [openai_reply(calls=[("read_file", {"path": "a"})])] * 6,
                 limits=RunLimits(max_turns=6))
    assert result.stop_reason == "repeat_failure"
    assert result.stats.model_calls == 5
    assert all(m["content"].startswith("ERROR") for m in result.messages if m["role"] == "tool")


def test_validator_exception_cannot_succeed(tmp_path):
    def validator(*args):
        raise OSError("checker unavailable")
    result = run(tmp_path, [openai_reply(text="done")], validator=validator)
    assert result.status == "failed" and result.stop_reason == "unverified"
    assert result.completion["tests_passed"] is not True


def test_read_available_again_after_compaction(tmp_path):
    (tmp_path / "a").write_text("start\n" + "data\n" * 500 + "NEEDED_AT_END\n")
    for name in ("b", "c"):
        (tmp_path / name).write_text("other\n" * 500)
    reads = [("read_file", {"path": name, "limit": 1000}) for name in ("a", "b", "c")]
    result = run(tmp_path, [openai_reply(calls=reads),
        (400, {"error": {"message": "maximum context length exceeded"}}),
        openai_reply(calls=[reads[0]]), openai_reply(text="done")])
    outputs = [m["content"] for m in result.messages if m["role"] == "tool"]
    assert "elided to save tokens" in outputs[0]
    assert "NEEDED_AT_END" in outputs[-1]


def test_schema_cost_is_reserved_before_paid_dispatch(tmp_path):
    route = paid_route(per_request_usd=0.002)
    transport = FakeTransport([openai_reply(text="done")])
    router = make_router(transport, routes=[route], tmp_path=tmp_path)
    tools = [{"type": "function", "function": {"name": "read", "description": "large " * 10000,
                "parameters": {"type": "object", "properties": {}}}}]
    with pytest.raises(BudgetError):
        router.chat([{"role": "user", "content": "hello"}], tools, max_tokens=512)
    assert transport.requests == []


def test_truncation_respects_turn_budget(tmp_path):
    result = run(tmp_path, [openai_reply(text="partial", finish="length"), openai_reply(text="done")],
                 limits=RunLimits(max_turns=1))
    assert result.stats.model_calls == 1
    assert result.status == "failed"


def test_provider_timeout_is_bounded_by_monotonic_deadline(monkeypatch):
    from ordivyr_lite.providers import ModelClient
    import time
    client = ModelClient(local_route())
    observed = []
    def request(messages, tools, max_tokens, temperature, timeout, **kwargs):
        observed.append(timeout)
        return object()
    monkeypatch.setattr(client, '_request', request)
    client.chat([], timeout=30, deadline=time.monotonic() + 0.2)
    assert 0 < observed[0] <= 0.2


def test_resume_retains_window_aware_compaction(tmp_path):
    route = local_route(context_window=2000)
    runner = Runner(make_router(FakeTransport([]), routes=[route], tmp_path=tmp_path),
                    ToolBox(tmp_path, profile='workspace'), 'task',
                    resume_state={'messages': [{'role': 'user', 'content': 'task'}]})
    runner._prepare()
    assert runner._compact_at <= 1400


def test_resume_restores_stricter_saved_compaction_threshold(tmp_path):
    route = local_route(context_window=16_000)
    runner = Runner(make_router(FakeTransport([]), routes=[route], tmp_path=tmp_path),
                    ToolBox(tmp_path, profile='workspace'), 'task',
                    resume_state={'messages': [{'role': 'user', 'content': 'task'}],
                                  'limits': {'compact_at': 800}})
    runner._prepare()
    assert runner._compact_at == 800


def test_progress_nudge_follows_all_tool_results(tmp_path):
    (tmp_path / 'a').write_text('data')
    result = run(tmp_path, [openai_reply(calls=[('read_file', {'path': 'a'})])] * 8
                          + [openai_reply(text='done')])
    pending = set()
    for m in result.messages:
        if m['role'] == 'assistant':
            pending = {c['id'] for c in m.get('tool_calls', [])}
        elif m['role'] == 'tool':
            pending.discard(m['tool_call_id'])
        elif m['role'] == 'user':
            assert not pending


def test_large_repetitive_exact_edit_keeps_precise_feedback(tmp_path):
    old = 'same\n' * 5000 + 'old\n' + 'tail\n' * 5000
    new = 'same\n' * 5000 + 'new\n' + 'tail\n' * 5000
    (tmp_path / 'a').write_text(old)
    box = ToolBox(tmp_path, profile='workspace')
    out = box.call('edit_file', {'path': 'a', 'old_string': old, 'new_string': new})
    assert out.endswith('-old\n+new')
    assert (tmp_path / 'a').read_text() == new
