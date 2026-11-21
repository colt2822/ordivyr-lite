"""Advertised tools are a subset. Permissions stay on the toolbox."""

from __future__ import annotations

import json

from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.toolselect import select_tools


def _names(payload: dict) -> list[str]:
    return [t["function"]["name"] for t in payload.get("tools") or []]


def test_a_read_does_not_advertise_write_tools():
    names, reason = select_tools(
        "Read NOTE.txt and reply with only the integer in that file.",
        ["list_files", "read_file", "grep", "find_files", "edit_file", "write_file"])
    assert reason == "read"
    assert names == ["list_files", "read_file", "grep", "find_files"]


def test_a_fix_keeps_every_permitted_tool():
    names, reason = select_tools(
        "Fix the failing test in calc.py",
        ["list_files", "read_file", "grep", "find_files", "edit_file", "write_file"])
    assert reason == "full"
    assert "edit_file" in names and "write_file" in names


def test_a_read_on_a_trusted_profile_keeps_shell():
    names, reason = select_tools(
        "inspect the input",
        ["list_files", "read_file", "grep", "find_files", "edit_file", "write_file",
         "run_command", "fetch_url"])
    assert reason == "read"
    assert "run_command" in names and "fetch_url" in names
    assert "edit_file" not in names and "write_file" not in names


def test_custom_tools_stay_on_a_read():
    names, reason = select_tools(
        "Read NOTE.txt",
        ["read_file", "edit_file", "delegate", "mcp_docs_search"])
    assert reason == "read"
    assert names == ["read_file", "delegate", "mcp_docs_search"]


def test_an_allow_list_is_not_widened(tmp_path):
    transport = FakeTransport([openai_reply(text="done")])
    box = ToolBox(tmp_path, profile="workspace", allow_tools=["read_file"])
    router = make_router(transport, tmp_path=tmp_path)
    Runner(router, box, "Fix the failing test.", limits=RunLimits(max_turns=1)).run()
    assert "edit_file" not in _names(transport.requests[0])
    assert "read_file" in _names(transport.requests[0])


def test_wire_schema_drops_property_descriptions_and_keeps_required(tmp_path):
    box = ToolBox(tmp_path, profile="workspace")
    full = json.dumps(box.spec(), separators=(",", ":"))
    wire = json.dumps(box.request_spec(), separators=(",", ":"))
    assert len(wire) < len(full)
    by_name = {t["function"]["name"]: t["function"] for t in box.request_spec()}
    read = by_name["read_file"]
    assert read["parameters"]["required"] == ["path"]
    assert "description" not in read["parameters"]["properties"]["path"]
    assert "spill:N" in read["description"]
    edit = by_name["edit_file"]
    assert edit["parameters"]["required"] == ["path", "old_string", "new_string"]
    assert "replace_all" in edit["description"]


def test_read_request_omits_write_schemas_but_a_write_still_runs(tmp_path):
    (tmp_path / "NOTE.txt").write_text("42\n", encoding="utf-8")
    transport = FakeTransport([
        openai_reply(text="", calls=[("edit_file", {
            "path": "NOTE.txt", "old_string": "42\n", "new_string": "43\n"})]),
        openai_reply(text="43"),
    ])
    box = ToolBox(tmp_path, profile="workspace")
    router = make_router(transport, tmp_path=tmp_path)
    result = Runner(router, box, "Read NOTE.txt and reply with only the integer.",
                    limits=RunLimits(max_turns=3)).run()
    assert "edit_file" not in _names(transport.requests[0])
    assert "read_file" in _names(transport.requests[0])
    assert result.stats.tool_selection_misses == 1
    assert (tmp_path / "NOTE.txt").read_text(encoding="utf-8") == "43\n"


def test_read_only_profile_still_refuses_a_write(tmp_path):
    transport = FakeTransport([
        openai_reply(text="", calls=[("write_file", {"path": "a.txt", "content": "x"})]),
        openai_reply(text="no"),
    ])
    box = ToolBox(tmp_path, profile="read-only")
    router = make_router(transport, tmp_path=tmp_path)
    result = Runner(router, box, "Fix the bug by writing a.txt",
                    limits=RunLimits(max_turns=2)).run()
    assert "write_file" not in _names(transport.requests[0])
    blob = json.dumps(transport.requests[1])
    assert "not available" in blob or "not allowed" in blob or "unknown tool" in blob
    assert result.stats.tool_calls == 0 or result.stats.refused_calls >= 0
    assert not (tmp_path / "a.txt").exists()
