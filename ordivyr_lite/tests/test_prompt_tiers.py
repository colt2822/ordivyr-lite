"""Prompt tiers, the stream flag, and selective memory.

A completion with no tools must not pay for the agent essay. A tool run must
not pay for a catalogue of tools that are turned off.
"""

from __future__ import annotations

import json

from ordivyr_lite.context import task_needs_tools
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.memory import ContextStore
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox


def test_no_tools_sends_only_the_user_message(tmp_path):
    transport = FakeTransport([openai_reply(text="pong", prompt_tokens=36, completion_tokens=2)])
    box = ToolBox(tmp_path, profile="workspace", allow_tools=[])
    router = make_router(transport, tmp_path=tmp_path)
    result = Runner(router, box, "Reply with the single word pong.",
                    limits=RunLimits(max_turns=2)).run()
    assert result.stop_reason == "done"
    assert result.text == "pong"
    sent = transport.requests[0]
    assert [m["role"] for m in sent["messages"]] == ["user"]
    assert "tools" not in sent
    blob = json.dumps(sent)
    assert "Unavailable" not in blob
    assert "ORDIVYR Lite" not in blob


def test_tool_prompt_does_not_list_disabled_tools(tmp_path):
    transport = FakeTransport([openai_reply(text="done")])
    box = ToolBox(tmp_path, profile="workspace")
    router = make_router(transport, tmp_path=tmp_path)
    Runner(router, box, "Look around.", limits=RunLimits(max_turns=1)).run()
    system = transport.requests[0]["messages"][0]["content"]
    assert "Unavailable tools" not in system
    assert "Tools enabled:" in system
    assert "run_command" not in system
    assert "read_file" in system


def test_no_stream_omits_the_sse_flag(tmp_path):
    transport = FakeTransport([openai_reply(text="pong")])
    box = ToolBox(tmp_path, profile="read-only", allow_tools=[])
    router = make_router(transport, tmp_path=tmp_path)
    Runner(router, box, "pong", limits=RunLimits(max_turns=1), stream=False).run()
    assert "stream" not in transport.requests[0]


def test_stream_requests_sse_by_default(tmp_path):
    transport = FakeTransport([openai_reply(text="pong")])
    box = ToolBox(tmp_path, profile="read-only", allow_tools=[])
    router = make_router(transport, tmp_path=tmp_path)
    Runner(router, box, "pong", limits=RunLimits(max_turns=1), stream=True).run()
    assert transport.requests[0]["stream"] is True


def test_unrelated_memory_stays_out_of_the_prompt(tmp_path):
    store = ContextStore(tmp_path, home=tmp_path / "home")
    store.remember(run_id="run-invoice", task="fix invoice vat total rounding",
                   files=["src/invoices.py"], errors=["KeyError: total"], outcome="resolved")
    assert store.recall("Reply with the single word pong.") == ""
    assert "run-invoice" in store.recall("fix invoice vat rounding")


def test_a_reply_does_not_need_tools_and_a_file_task_does():
    assert task_needs_tools("Reply with the single word pong.") is False
    assert task_needs_tools("hello") is False
    assert task_needs_tools("what is 2 + 2?") is False
    assert task_needs_tools("read NOTE.txt and change the total") is True
    assert task_needs_tools("fix the failing test") is True
    assert task_needs_tools("") is False


def test_compact_schemas_keep_the_call_contract(tmp_path):
    box = ToolBox(tmp_path, profile="workspace")
    by_name = {d.name: d for d in box.defs}
    assert by_name["read_file"].schema["required"] == ["path"]
    assert "spill:N" in by_name["read_file"].description
    assert by_name["edit_file"].schema["required"] == ["path", "old_string", "new_string"]
    assert "replace_all" in by_name["edit_file"].description
    spec = json.dumps(box.spec(), separators=(",", ":"))
    # The pre-change workspace schema was 2,689 characters. Stay under that.
    assert len(spec) < 2689


def test_delegate_is_absent_until_enabled(tmp_path):
    box = ToolBox(tmp_path, profile="workspace")
    assert "delegate" not in box.names()
    transport = FakeTransport([openai_reply(text="done")])
    router = make_router(transport, tmp_path=tmp_path)
    Runner(router, box, "hi", limits=RunLimits(max_turns=1)).run()
    names = [t["function"]["name"] for t in transport.requests[0].get("tools") or []]
    assert "delegate" not in names
