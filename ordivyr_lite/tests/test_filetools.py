from __future__ import annotations

import os
from pathlib import Path

import pytest

from ordivyr_lite.tools import ToolBox, ToolError


@pytest.fixture
def work(tmp_path):
    (tmp_path / "a.txt").write_text("alpha\nbeta\ngamma\n")
    (tmp_path / "big.py").write_text("\n".join(f"value_{i} = {i}" for i in range(600)) + "\n")
    return tmp_path


def test_read_file_is_paginated_and_line_numbered(work):
    box = ToolBox(work)
    out = box.call("read_file", {"path": "a.txt", "offset": 2, "limit": 1})
    assert "     2\tbeta" in out
    assert "offset=3" in out


def test_read_file_rejects_path_escape(work):
    box = ToolBox(work)
    assert box.call("read_file", {"path": "../etc/passwd"}).startswith("ERROR:")


def test_read_file_rejects_symlink(work):
    (work / "link").symlink_to("/etc/passwd")
    box = ToolBox(work)
    assert "symlinks" in box.call("read_file", {"path": "link"}).lower()


def test_edit_file_unique_match(work):
    box = ToolBox(work, profile="workspace")
    out = box.call("edit_file", {"path": "a.txt", "old_string": "beta", "new_string": "BETA"})
    assert "edited" in out
    assert (work / "a.txt").read_text() == "alpha\nBETA\ngamma\n"


def test_edit_file_ambiguous_refused(work):
    (work / "dup.txt").write_text("x\nx\n")
    box = ToolBox(work, profile="workspace")
    out = box.call("edit_file", {"path": "dup.txt", "old_string": "x", "new_string": "y"})
    assert out.startswith("ERROR:")
    assert "matches 2 places" in out


def test_edit_file_replace_all(work):
    (work / "dup.txt").write_text("x\nx\n")
    box = ToolBox(work, profile="workspace")
    out = box.call("edit_file", {"path": "dup.txt", "old_string": "x", "new_string": "y",
                                 "replace_all": True})
    assert "2 exact match" in out
    assert (work / "dup.txt").read_text() == "y\ny\n"


def test_edit_file_tolerant_only_when_unique(work):
    (work / "t.txt").write_text("  a = 1\n  b = 2\n")
    box = ToolBox(work, profile="workspace")
    out = box.call("edit_file", {"path": "t.txt", "old_string": "a = 1\nb = 2",
                                 "new_string": "a = 3\nb = 4"})
    assert "whitespace-tolerant" in out
    assert "a = 3" in (work / "t.txt").read_text()


def test_edit_file_preserves_crlf(work):
    (work / "crlf.txt").write_bytes(b"one\r\ntwo\r\n")
    box = ToolBox(work, profile="workspace")
    box.call("edit_file", {"path": "crlf.txt", "old_string": "two", "new_string": "TWO"})
    assert (work / "crlf.txt").read_bytes() == b"one\r\nTWO\r\n"


def test_grep_and_find(work):
    box = ToolBox(work)
    hits = box.call("grep", {"pattern": "gamma"})
    assert "a.txt:3" in hits
    files = box.call("find_files", {"glob": "*.py"})
    assert "big.py" in files


def test_spill_and_page(work):
    box = ToolBox(work)
    huge = "x" * 20_000
    out = box.call("grep", {"pattern": "x"})  # no spill expected
    assert isinstance(out, str)
    n = box.spill("hello\nworld\n")
    assert "hello" in box.call("read_file", {"path": f"spill:{n}"})


def test_write_file_is_confined(work):
    box = ToolBox(work, profile="workspace")
    assert box.call("write_file", {"path": "../x", "content": "x"}).startswith("ERROR:")
    assert not (work.parent / "x").exists()


def test_write_file_does_not_follow_symlinked_parent(work):
    (work / "real").mkdir()
    (work / "linkdir").symlink_to(work / "real")
    box = ToolBox(work, profile="workspace")
    out = box.call("write_file", {"path": "linkdir/x", "content": "x"})
    assert out.startswith("ERROR:")
