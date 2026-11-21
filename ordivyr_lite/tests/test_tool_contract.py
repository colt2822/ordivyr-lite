"""Tool-layer contract: a failed command is a failed call, salvaged output on a
timeout, schema-named refusals with unambiguous coercion, ambiguity reported as
ambiguity, a bounded diff after a successful edit, and true search totals.

Every test here pins one item of the tool contract that loop.py derives ok from.
"""

from __future__ import annotations

import os
import re
import shutil
import threading

import pytest

from ordivyr_lite import filetools as FT
from ordivyr_lite import sandbox as SB
from ordivyr_lite.tools import ToolBox, _coerce


@pytest.fixture
def work(tmp_path):
    return tmp_path


def box_for(work, profile="workspace"):
    return ToolBox(work, profile=profile)


def put(work, name, text):
    p = work / name
    p.write_bytes(text.encode("utf-8"))
    return p


def edit(work, name, old, new, **kw):
    return box_for(work).call("edit_file", {"path": name, "old_string": old,
                                            "new_string": new, **kw})


# --------------------------------------------------------------- 1. exit status
def test_a_command_that_exits_non_zero_is_a_failed_tool_call(work):
    box = ToolBox(work, profile="trusted", sandbox_mode="process")
    out = box.call("run_command", {"command": "echo stdout-line; echo stderr-line >&2; exit 1"})
    assert out.startswith("ERROR: command failed (exit=1)")
    rest = out.split("\n", 1)[1]
    assert "stdout-line" in rest
    assert "stderr-line" in rest


def test_a_command_that_exits_zero_keeps_the_old_result_shape(work):
    box = ToolBox(work, profile="trusted", sandbox_mode="process")
    out = box.call("run_command", {"command": "echo hi"})
    assert not out.startswith("ERROR:")
    assert "exit=0" in out and "sandbox=process" in out and "hi" in out


def test_a_refused_command_leaves_the_sandbox_fields_visible(work):
    box = ToolBox(work, profile="trusted", sandbox_mode="process")
    out = box.call("run_command", {"command": "echo before; sleep 5", "timeout": 1})
    assert out.startswith("ERROR: command timed out after 1s (exit=-9)")
    assert "before" in out
    # test_security.py and operators still read these two markers
    assert "timed_out=True" in out and "sandbox=process" in out


def test_a_failing_command_is_an_error_through_a_real_sandbox(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap not installed")
    box = ToolBox(work, profile="trusted", sandbox_mode="bwrap")
    try:
        out = box.call("run_command", {"command": "echo sandbox-stdout; exit 3"})
        assert out.startswith("ERROR: command failed (exit=3)")
        assert "sandbox-stdout" in out.split("\n", 1)[1]
    finally:
        box.close()


# --------------------------------------------------------------- 2. salvage
def _bare_session(tmp_path, fake_proc=None):
    """A SandboxSession with no bwrap process, for protocol-level tests."""
    sess = object.__new__(SB.SandboxSession)
    sess.workdir, sess.network, sess.mode, sess.memory_mb = tmp_path, False, "bwrap", 4096
    sess.fallback, sess._persistent, sess._closed = False, True, False
    sess._proc, sess._buffer = fake_proc, b""
    sess._lock = threading.Lock()
    return sess


class _Pipe:
    def __init__(self):
        self.written = b""

    def write(self, data):
        self.written += data

    def flush(self):
        pass

    def close(self):
        pass


def test_a_framed_timeout_keeps_what_the_driver_streamed(tmp_path):
    r, w = os.pipe()
    os.write(w, b"compile said no")
    reader = os.fdopen(r, "rb", 0)

    class FakeProc:
        pid, returncode, stdout, stderr = 999_999, -9, reader, None

        def __init__(self):
            self.stdin = _Pipe()

        def poll(self):
            return None

        def kill(self):
            pass

        def wait(self, timeout=None):
            return -9

    sess = _bare_session(tmp_path, FakeProc())
    try:
        result = sess.run("sleep 30", timeout=1)
    finally:
        os.close(w)
    assert result.timed_out is True
    assert result.ok is False
    assert "compile said no" in result.stdout


def test_salvage_strips_control_bytes_and_the_stale_frame_header(tmp_path):
    sess = _bare_session(tmp_path)
    sess._buffer = b"0123456789ab 0 5 0\nkeep\x00\x07me"
    text, truncated = sess._salvage("0123456789ab")
    assert text == "keepme"
    assert truncated is False


def test_salvage_caps_what_it_returns(tmp_path):
    sess = _bare_session(tmp_path)
    sess._buffer = b"x" * 60_000
    text, truncated = sess._salvage("nope")
    assert truncated is True
    assert len(text) <= SB._SALVAGE_CAP
    assert text.endswith("[... output truncated ...]")


# --------------------------------------------------------------- 3. validation
def test_validate_coerces_an_integer_string_in_place(work):
    box = box_for(work)
    args = {"path": "a.txt", "limit": "50"}
    assert box.validate("read_file", args) is None
    assert args["limit"] == 50 and isinstance(args["limit"], int)


@pytest.mark.parametrize("value", ["1,50", "50.5", "fifty", ""])
def test_validate_refuses_an_ambiguous_integer(work, value):
    box = box_for(work)
    args = {"path": "a.txt", "limit": value}
    refusal = box.validate("read_file", args)
    assert refusal and refusal.startswith("parameter 'limit' must be integer")
    assert args["limit"] == value


@pytest.mark.parametrize("kind, value, expected", [
    ("number", "0.5", 0.5),
    ("number", "3", 3.0),
    ("integer", " 50 ", 50),
    ("integer", 7.0, 7),
    ("boolean", "true", True),
    ("boolean", "false", False),
    ("boolean", 1, True),
    ("boolean", 0, False),
    ("string", 50, "50"),
    ("string", 0.5, "0.5"),
])
def test_unambiguous_scalars_are_coerced(kind, value, expected):
    accepted, coerced = _coerce(kind, value)
    assert accepted is True
    assert coerced == expected and type(coerced) is type(expected)


@pytest.mark.parametrize("kind, value", [
    ("number", "1,5"),
    ("number", "half"),
    ("boolean", "yes"),
    ("boolean", "True"),
    ("boolean", 2),
    ("string", True),
    ("integer", True),
])
def test_ambiguous_scalars_are_refused_not_guessed(kind, value):
    accepted, coerced = _coerce(kind, value)
    assert accepted is False
    assert coerced == value


def test_boolean_and_string_coercion_reach_the_payload(work):
    box = box_for(work)
    args = {"pattern": "x", "regex": "true", "path": 7}
    assert box.validate("grep", args) is None
    assert args["regex"] is True and args["path"] == "7"
    bad = {"pattern": "x", "regex": "yes"}
    refusal = box.validate("grep", bad)
    assert refusal.startswith("parameter 'regex' must be boolean")
    assert "true or false" in refusal


def test_missing_parameter_lists_the_required_set_and_the_accepted_types(work):
    refusal = box_for(work).validate("read_file", {})
    assert refusal.startswith("missing required parameter(s): path")
    assert "path (string, required)" in refusal
    assert "offset (integer, optional)" in refusal
    assert "limit (integer, optional)" in refusal


def test_a_bad_type_names_what_the_parameter_accepts(work):
    refusal = box_for(work).validate("read_file", {"path": "a", "limit": "ten"})
    assert refusal.startswith("parameter 'limit' must be integer")
    assert "whole number such as 50" in refusal
    assert "'ten'" in refusal


def test_an_unknown_tool_names_the_tools_this_profile_offers(work):
    refusal = box_for(work).validate("nope", {})
    assert refusal.startswith("unknown tool 'nope'")
    assert "available tools:" in refusal
    assert "list_files" in refusal and "edit_file" in refusal


def test_a_profile_refusal_names_the_tools_that_profile_offers(work):
    refusal = ToolBox(work, profile="read-only").validate("write_file",
                                                         {"path": "a", "content": "x"})
    assert refusal.startswith("tool 'write_file' is not available in the read-only permission profile")
    assert "read_file" in refusal and "write_file" not in refusal.split("offers:")[1]


# --------------------------------------------------------------- 4. ambiguity
@pytest.fixture(autouse=True)
def clean_match_stats():
    for key in FT.MATCH_STATS:
        FT.MATCH_STATS[key] = 0
    yield
    for key in FT.MATCH_STATS:
        FT.MATCH_STATS[key] = 0


def test_a_tolerant_multi_match_is_reported_as_ambiguity(work):
    p = put(work, "amb.txt", "a = 1  \nb = 2\n---\na = 1\t\nb = 2\n")
    out = edit(work, "amb.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert out.startswith("ERROR:")
    assert "matches 2 places" in out
    assert "lines 1, 4" in out
    assert "replace_all" in out
    assert "was not found" not in out
    assert p.read_text() == "a = 1  \nb = 2\n---\na = 1\t\nb = 2\n"
    assert FT.MATCH_STATS["fail"] == 1


def test_ambiguity_names_at_most_four_lines(work):
    body = "---\n".join(f"{pad}a = 1\n{pad}b = 2\n" for pad in ("  ", "    ", "\t", "   ", " "))
    put(work, "many.txt", body)
    out = edit(work, "many.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert out.startswith("ERROR:")
    assert "matches 5 places" in out
    assert "lines 1, 4, 7, 10" in out
    assert "line 13" not in out and ", 13" not in out


def test_text_that_is_truly_absent_still_says_not_found(work):
    body = "".join(f"value_{i} = {i}\n" for i in range(12))
    put(work, "z.txt", body)
    out = edit(work, "z.txt", "value_5 = 55", "value_5 = 5")
    assert out.startswith("ERROR:") and "was not found" in out


# --------------------------------------------------------------- 5. edit diff
def test_a_successful_edit_echoes_the_changed_region(work):
    p = put(work, "d.txt", "alpha\nbeta\ngamma\n")
    out = edit(work, "d.txt", "alpha\nbeta\ngamma", "one\ntwo\nthree")
    head, *rows = out.split("\n")
    assert head.startswith("edited d.txt at line 1") and "1 exact match" in head
    assert rows == ["-alpha", "-beta", "-gamma", "+one", "+two", "+three"]
    assert len(out) <= 1200
    assert p.read_text() == "one\ntwo\nthree\n"


def test_the_diff_does_not_change_what_is_written(work):
    targeted = put(work, "a.txt", "alpha\nbeta\ngamma\n")
    exact = put(work, "b.txt", "alpha\nbeta\ngamma\n")
    edit(work, "a.txt", "alpha\nbeta\ngamma", "one\ntwo\nthree")
    FT.atomic_write(exact, "one\ntwo\nthree\n")
    assert targeted.read_bytes() == exact.read_bytes()


def test_a_tolerant_match_still_names_the_stage_before_the_diff(work):
    p = put(work, "t.txt", "  a = 1\n  b = 2\n")
    out = edit(work, "t.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    head, *rows = out.split("\n")
    assert "indentation-insensitive" in head
    assert rows == ["-  a = 1", "-  b = 2", "+  a = 3", "+  b = 4"]
    assert p.read_text() == "  a = 3\n  b = 4\n"


def test_a_long_diff_is_capped_and_says_how_much_is_hidden(work):
    old = "\n".join(f"line_{i} = {i}" for i in range(40))
    new = "\n".join(f"value_{i} = -{i}" for i in range(40))
    put(work, "big.txt", old + "\n")
    out = edit(work, "big.txt", old, new)
    head, *rows = out.split("\n")
    assert head.startswith("edited big.txt")
    assert len(rows) == 9, rows                        # 8 diff lines + the notice
    assert re.fullmatch(r"\+\d+ -\d+ lines not shown", rows[-1]), rows[-1]
    assert len(out) <= 900
    assert "value_39 = -39" in (work / "big.txt").read_text()


# --------------------------------------------------------------- search totals
def test_grep_reports_the_true_total_when_it_stops_at_the_cap(work):
    for name in ("a.txt", "b.txt", "c.txt"):
        put(work, name, "".join(f"needle {i}\n" for i in range(40)))
    out = box_for(work).call("grep", {"pattern": "needle"})
    assert out.startswith("a.txt:1: needle 0")
    assert f"showing {FT.GREP_MAX_MATCHES} of 120 matches in 3 files" in out
    assert len(out.split("\n")) == FT.GREP_MAX_MATCHES + 1


def test_an_under_cap_grep_result_carries_no_notice(work):
    put(work, "a.txt", "needle\nother\n")
    out = box_for(work).call("grep", {"pattern": "needle"})
    assert out == "a.txt:1: needle"


def test_a_grep_deadline_is_announced(work, monkeypatch):
    (work / "sub").mkdir()
    (work / "sub" / "a.txt").write_text("needle\n", "utf-8")
    monkeypatch.setattr(FT, "SEARCH_SECONDS", -1)
    out = box_for(work).call("grep", {"pattern": "needle"})
    assert "no matches" in out
    assert FT.DEADLINE_NOTE in out


def test_a_find_deadline_is_announced(work, monkeypatch):
    (work / "sub").mkdir()
    (work / "sub" / "a.py").write_text("x = 1\n", "utf-8")
    monkeypatch.setattr(FT, "SEARCH_SECONDS", -1)
    out = box_for(work).call("find_files", {"glob": "*.py"})
    assert "no files matched" in out
    assert FT.DEADLINE_NOTE in out


def test_grep_and_find_still_answer_when_the_walk_finishes(work):
    put(work, "a.py", "needle\n")
    box = box_for(work)
    assert FT.DEADLINE_NOTE not in box.call("grep", {"pattern": "needle"})
    assert FT.DEADLINE_NOTE not in box.call("find_files", {"glob": "*.py"})
