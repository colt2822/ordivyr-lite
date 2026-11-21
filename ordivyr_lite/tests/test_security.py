from __future__ import annotations

import shutil

import pytest

from ordivyr_lite.session import Session, list_sessions
from ordivyr_lite.tools import ToolBox, ToolError, fetch_url


def test_fetch_url_rejects_private_and_non_http():
    with pytest.raises(ToolError, match="private or local"):
        fetch_url("http://127.0.0.1/")
    with pytest.raises(ToolError, match="only plain http"):
        fetch_url("file:///etc/passwd")
    with pytest.raises(ToolError, match="credentials"):
        fetch_url("http://user:pass@example.com/")


def test_absolute_and_parent_paths_are_refused(tmp_path):
    box = ToolBox(tmp_path)
    assert "relative" in box.call("read_file", {"path": "/etc/passwd"})
    assert "escapes" in box.call("read_file", {"path": "../x"})


def test_command_timeout_is_reported(tmp_path):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    box = ToolBox(tmp_path, profile="trusted", sandbox_mode="bwrap")
    out = box.call("run_command", {"command": "sleep 5", "timeout": 1})
    assert "timed_out=True" in out
    assert "exit=" in out


def test_session_round_trip(tmp_path):
    s = Session.create(tmp_path / "s.jsonl", run_id="r1")
    s.snapshot({"messages": [{"role": "user", "content": "hi"}], "stats": {}})
    s.close()
    reopened = Session.open(tmp_path / "s.jsonl")
    assert reopened.latest_snapshot()["messages"][0]["content"] == "hi"
    assert reopened.info()["snapshots"] == 1
    assert list_sessions(tmp_path)[0]["run_id"] == "r1"
