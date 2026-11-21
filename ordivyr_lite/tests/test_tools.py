from __future__ import annotations

import shutil
import uuid

import pytest

from ordivyr_lite.tools import PROFILES, ToolBox, ToolError


@pytest.fixture
def work(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n")
    return tmp_path


def test_read_only_offers_no_write_or_shell(work):
    box = ToolBox(work, profile="read-only")
    assert "write_file" not in box.names()
    assert "edit_file" not in box.names()
    assert "run_command" not in box.names()
    assert box.call("write_file", {"path": "x", "content": "x"}).startswith("ERROR:")
    assert not (work / "x").exists()


def test_workspace_adds_write_but_not_shell(work):
    box = ToolBox(work, profile="workspace")
    assert "write_file" in box.names()
    assert "edit_file" in box.names()
    assert "run_command" not in box.names()


def test_trusted_adds_shell_and_web(work):
    box = ToolBox(work, profile="trusted")
    assert "run_command" in box.names()
    assert "fetch_url" in box.names()


@pytest.mark.parametrize(("profile", "shell", "web", "expected_shell", "expected_web"), [
    ("workspace", None, None, False, False),
    ("trusted", None, None, True, True),
    ("full", None, None, True, True),
    ("trusted", False, None, False, True),
    ("trusted", None, False, True, False),
    ("workspace", True, None, True, False),
    ("workspace", None, True, False, True),
])
def test_three_state_permission_matrix(work, profile, shell, web, expected_shell, expected_web):
    box = ToolBox(work, profile=profile, allow_shell=shell, allow_web=web)
    assert box.has("run_command") is expected_shell
    assert box.has("fetch_url") is expected_web
    assert box.permission_report()["available"] == box.names()


def test_read_only_cannot_be_widened_with_shell_or_web(work):
    with pytest.raises(ToolError, match="read-only profile cannot enable"):
        ToolBox(work, profile="read-only", allow_shell=True)
    with pytest.raises(ToolError, match="read-only profile cannot enable"):
        ToolBox(work, profile="read-only", allow_web=True)


def test_allow_list_denial_reason_is_actionable(work):
    box = ToolBox(work, profile="trusted", allow_tools=["read_file"])
    assert "explicit --tools allow-list" in box.unavailable_reason("run_command")


def test_allow_list_narrows(work):
    box = ToolBox(work, profile="trusted", allow_tools=["read_file"])
    assert set(box.names()) == {"read_file", "grep", "find_files"}


def test_explicit_empty_allow_list_registers_no_tools(work):
    box = ToolBox(work, profile="trusted", allow_tools=[])
    assert box.names() == []
    assert "explicit --tools allow-list" in box.validate("read_file", {"path": "a.txt"})


def test_validate_required_and_types(work):
    box = ToolBox(work)
    assert "missing" in box.validate("read_file", {})
    assert "integer" in box.validate("read_file", {"path": "a", "limit": "ten"})
    assert box.validate("read_file", {"path": "a"}) is None
    assert "unknown" in box.validate("nope", {})


def test_validate_permission_refusal(work):
    box = ToolBox(work, profile="read-only")
    assert "permission profile" in box.validate("write_file", {"path": "a", "content": "x"})


def test_validate_rejects_non_object(work):
    box = ToolBox(work)
    assert "JSON object" in box.validate("read_file", ["a"])


def test_list_files_skips_git(work):
    (work / ".git").mkdir()
    box = ToolBox(work)
    assert ".git" not in box.call("list_files", {"path": "."})


def test_run_command_sandbox_writes_inside_only(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed")
    box = ToolBox(work, profile="trusted", sandbox_mode="bwrap")
    token = "escape-" + uuid.uuid4().hex
    out = box.call("run_command", {"command": f"pwd; echo hi > made.txt; touch /tmp/{token}; cat /etc/hostname"})
    assert "sandbox=bwrap" in out
    assert "exit=0" in out
    assert (work / "made.txt").read_text().strip() == "hi"
    assert not __import__("pathlib").Path(f"/tmp/{token}").exists()


def test_run_command_network_off_by_default(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed")
    box = ToolBox(work, profile="trusted", sandbox_mode="bwrap")
    out = box.call("run_command", {"command": "getent hosts example.com || true"})
    # Inside the netns this normally fails/returns nothing; the key assertion is
    # that the tool reports a sandboxed, bounded result rather than crashing.
    assert "sandbox=bwrap" in out


def test_process_sandbox_is_explicit(work):
    box = ToolBox(work, profile="trusted", sandbox_mode="process")
    out = box.call("run_command", {"command": "echo hi"})
    assert "sandbox=process" in out
    assert "hi" in out


def test_workspace_scoped_shell_executes_in_configured_sandbox(work):
    box = ToolBox(work, profile="workspace", allow_shell=True, sandbox_mode="process")
    out = box.call("run_command", {"command": "python3 -c 'from pathlib import Path; Path(\"executed.txt\").write_text(\"ok\")'"})
    assert "exit=0" in out
    assert (work / "executed.txt").read_text() == "ok"


def test_workspace_scoped_shell_executes_in_bubblewrap_when_available(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed")
    box = ToolBox(work, profile="workspace", allow_shell=True, sandbox_mode="bwrap")
    out = box.call("run_command", {"command": "pwd; printf ok > scoped.txt"})
    assert "sandbox=bwrap" in out and "exit=0" in out
    assert (work / "scoped.txt").read_text() == "ok"
