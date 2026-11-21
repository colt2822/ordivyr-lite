"""edit_file robustness: the exact -> rstrip -> indent -> collapse ladder, its
uniqueness rule, the zero-match hint and the MATCH_STATS counters."""

from __future__ import annotations

import re
import time

import pytest

from ordivyr_lite.filetools import MATCH_STATS
from ordivyr_lite.tools import ToolBox

KEYS = ("exact", "rstrip", "indent", "collapse", "fail")


@pytest.fixture
def work(tmp_path):
    return tmp_path


@pytest.fixture(autouse=True)
def clean_stats():
    for k in KEYS:
        MATCH_STATS[k] = 0
    yield MATCH_STATS
    for k in KEYS:
        MATCH_STATS[k] = 0


def put(work, name: str, text: str):
    p = work / name
    p.write_bytes(text.encode("utf-8"))
    return p


def edit(work, name: str, old: str, new: str, **kw) -> str:
    box = ToolBox(work, profile="workspace")
    return box.call("edit_file", {"path": name, "old_string": old, "new_string": new, **kw})


def test_exact_hit_is_stage_one(work):
    p = put(work, "a.txt", "alpha\nbeta\ngamma\n")
    out = edit(work, "a.txt", "beta", "BETA")
    assert "1 exact match" in out and out.startswith("edited ")
    assert p.read_text() == "alpha\nBETA\ngamma\n"
    assert MATCH_STATS == {"exact": 1, "rstrip": 0, "indent": 0, "collapse": 0, "fail": 0}


def test_exact_hit_inside_a_line(work):
    p = put(work, "a.txt", "call(alpha, beta)\n")
    assert "1 exact match" in edit(work, "a.txt", "alpha, beta", "gamma")
    assert p.read_text() == "call(gamma)\n"


def test_rstrip_stage_drops_trailing_whitespace(work):
    p = put(work, "r.txt", "  a = 1   \n  b = 2\t\n")
    out = edit(work, "r.txt", "  a = 1\n  b = 2", "  a = 9\n  b = 8")
    assert "trailing-whitespace-tolerant" in out
    assert p.read_text() == "  a = 9\n  b = 8\n"
    assert MATCH_STATS["rstrip"] == 1 and MATCH_STATS["exact"] == 0


def test_indent_stage_reindents_to_the_block(work):
    p = put(work, "i.txt", "  a = 1\n  b = 2\n")
    out = edit(work, "i.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert "indentation-insensitive" in out and "whitespace-tolerant" in out
    assert p.read_text() == "  a = 3\n  b = 4\n"
    assert MATCH_STATS["indent"] == 1


def test_collapse_stage_folds_internal_runs(work):
    p = put(work, "c.txt", "result = f(a,b)\n")
    out = edit(work, "c.txt", "result  =  f(a,b)", "result = g(a,b)")
    assert "collapsed-whitespace" in out
    assert p.read_text() == "result = g(a,b)\n"
    assert MATCH_STATS["collapse"] == 1


def test_exact_ambiguity_is_refused(work):
    p = put(work, "d.txt", "x\nx\n")
    out = edit(work, "d.txt", "x", "y")
    assert out.startswith("ERROR:") and "matches 2 places" in out and "lines 1, 2" in out
    assert p.read_text() == "x\nx\n"
    assert MATCH_STATS["exact"] == 0 and MATCH_STATS["fail"] == 1


def test_rstrip_ambiguity_is_refused(work):
    p = put(work, "r2.txt", "a = 1  \nb = 2\n---\na = 1\t\nb = 2\n")
    out = edit(work, "r2.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert out.startswith("ERROR:")
    assert p.read_text() == "a = 1  \nb = 2\n---\na = 1\t\nb = 2\n"
    assert MATCH_STATS["fail"] == 1 and MATCH_STATS["rstrip"] == 0


def test_indent_ambiguity_is_refused(work):
    p = put(work, "i2.txt", "\ta = 1\n\tb = 2\n---\n    a = 1\n    b = 2\n")
    out = edit(work, "i2.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert out.startswith("ERROR:")
    assert p.read_text() == "\ta = 1\n\tb = 2\n---\n    a = 1\n    b = 2\n"
    assert MATCH_STATS["indent"] == 0 and MATCH_STATS["fail"] == 1


def test_collapse_ambiguity_is_refused(work):
    p = put(work, "c2.txt", "x = f(1,  2)\nx = f(1, 2)\n")
    out = edit(work, "c2.txt", "x  =  f(1, 2)", "x = g(1, 2)")
    assert out.startswith("ERROR:")
    assert p.read_text() == "x = f(1,  2)\nx = f(1, 2)\n"
    assert MATCH_STATS["collapse"] == 0 and MATCH_STATS["fail"] == 1


def test_zero_match_names_the_closest_line(work):
    body = "".join(f"value_{i} = {i}\n" for i in range(12))
    p = put(work, "z.txt", body)
    out = edit(work, "z.txt", "value_5 = 55", "value_5 = 5")
    assert out.startswith("ERROR:") and "was not found" in out
    assert re.search(r"line 6: ", out)
    assert "read_file" in out and "offset=6" in out
    assert len(out) < 720
    assert p.read_text() == body
    assert MATCH_STATS["fail"] == 1


def test_zero_match_hint_truncates_long_snippets(work):
    long_line = "value_5 = " + "9" * 400
    body = "".join(f"value_{i} = {i}\n" for i in range(12) if i != 5) + long_line + "\n"
    put(work, "z2.txt", body)
    out = edit(work, "z2.txt", "value_5 = 55", "value_5 = 5")
    assert out.startswith("ERROR:") and len(out) < 720
    assert "9" * 130 not in out


def test_crlf_file_is_preserved(work):
    p = work / "crlf.txt"
    p.write_bytes(b"one\r\ntwo\r\n")
    assert "1 exact match" in edit(work, "crlf.txt", "two", "TWO")
    assert p.read_bytes() == b"one\r\nTWO\r\n"


def test_crlf_survives_a_tolerant_stage(work):
    p = work / "crlf2.txt"
    p.write_bytes(b"  a = 1\r\n  b = 2\r\n")
    out = edit(work, "crlf2.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert "indentation-insensitive" in out
    assert p.read_bytes() == b"  a = 3\r\n  b = 4\r\n"


def test_missing_trailing_newline_is_preserved(work):
    p = work / "n.txt"
    p.write_bytes(b"a\nb")
    edit(work, "n.txt", "b", "B")
    assert p.read_bytes() == b"a\nB"


def test_tab_block_vs_space_old_string(work):
    p = put(work, "t.txt", "\treturn 1\n")
    out = edit(work, "t.txt", "    return 1", "    return 2")
    assert "indentation-insensitive" in out
    assert p.read_text() == "\treturn 2\n"


def test_dedented_old_string_matches_an_indented_block(work):
    p = put(work, "m.py", "class C:\n    def m(self):\n        return 1\n")
    out = edit(work, "m.py", "def m(self):\n    return 1", "def m(self):\n    return 2")
    assert "indentation-insensitive" in out
    assert p.read_text() == "class C:\n    def m(self):\n        return 2\n"


def test_replace_all_applies_at_the_exact_stage(work):
    p = put(work, "ra.txt", "x\nx\n")
    out = edit(work, "ra.txt", "x", "y", replace_all=True)
    assert "2 exact match(es)" in out
    assert p.read_text() == "y\ny\n"
    assert MATCH_STATS == {"exact": 1, "rstrip": 0, "indent": 0, "collapse": 0, "fail": 0}


def test_replace_all_still_needs_a_unique_tolerant_match(work):
    p = put(work, "ra2.txt", "  a = 1\n  b = 2\n")
    out = edit(work, "ra2.txt", "a = 1\nb = 2", "a = 3\nb = 4", replace_all=True)
    assert "indentation-insensitive" in out
    assert p.read_text() == "  a = 3\n  b = 4\n"
    assert MATCH_STATS["indent"] == 1


def test_empty_old_string_is_refused(work):
    p = put(work, "e.txt", "abc\n")
    out = edit(work, "e.txt", "", "x")
    assert out.startswith("ERROR:") and "empty" in out
    assert p.read_text() == "abc\n"
    assert MATCH_STATS == {"exact": 0, "rstrip": 0, "indent": 0, "collapse": 0, "fail": 0}


def test_identical_strings_are_refused(work):
    p = put(work, "same.txt", "abc\n")
    out = edit(work, "same.txt", "abc", "abc")
    assert out.startswith("ERROR:") and "identical" in out
    assert p.read_text() == "abc\n"


def test_unicode_content_round_trips(work):
    p = put(work, "u.txt", "naive = 'cafe'\nemoji = '\U0001f600'\n")
    assert "1 exact match" in edit(work, "u.txt", "naive = 'cafe'", "naive = 'caf\u00e9'")
    assert p.read_text() == "naive = 'caf\u00e9'\nemoji = '\U0001f600'\n"
    assert p.read_bytes() == "naive = 'caf\u00e9'\nemoji = '\U0001f600'\n".encode("utf-8")


def test_one_megabyte_file_matches_fast(work):
    body = "".join(f"value_{i} = {i}\n" for i in range(60_000))
    assert len(body) > 1_000_000
    p = put(work, "big.py", body)
    box = ToolBox(work, profile="workspace")
    t0 = time.perf_counter()
    out = box.call("edit_file", {"path": "big.py", "old_string": "value_30000 = 30000",
                                 "new_string": "value_30000 = -1"})
    elapsed = time.perf_counter() - t0
    assert "1 exact match" in out
    assert elapsed < 0.3, elapsed
    assert "value_30000 = -1\n" in p.read_text()
    assert MATCH_STATS["exact"] == 1


def test_large_file_tolerant_stage_stays_off_when_ambiguous(work):
    body = "    v = 1\n" * 20_000
    p = put(work, "big2.py", body)
    box = ToolBox(work, profile="workspace")
    out = box.call("edit_file", {"path": "big2.py", "old_string": "v = 1\nv = 1",
                                 "new_string": "v = 2\nv = 2"})
    assert out.startswith("ERROR:")
    assert p.read_text() == body


def test_match_stats_count_every_stage(work):
    put(work, "e.txt", "alpha\nbeta\n")
    put(work, "r.txt", "  a = 1  \n  b = 2\n")
    put(work, "i.txt", "    a = 1\n    b = 2\n")
    put(work, "c.txt", "result = f(a,b)\n")
    put(work, "z.txt", "nothing here\n")
    put(work, "d.txt", "x\nx\n")
    edit(work, "e.txt", "beta", "BETA")
    edit(work, "r.txt", "  a = 1\n  b = 2", "  a = 3\n  b = 4")
    edit(work, "i.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    edit(work, "c.txt", "result  =  f(a,b)", "result = g(a,b)")
    edit(work, "z.txt", "elsewhere", "x")
    edit(work, "d.txt", "x", "y")
    assert MATCH_STATS == {"exact": 1, "rstrip": 1, "indent": 1, "collapse": 1, "fail": 2}


def test_match_stats_keys_are_fixed():
    assert set(MATCH_STATS) == set(KEYS)
    assert all(isinstance(v, int) for v in MATCH_STATS.values())


@pytest.mark.parametrize("name, body, old, new", [
    ("m1.txt", "abc\ndef\n", "abc\nefg", "x"),
    ("m2.txt", "x\nx\n", "x", "y"),
    ("m3.txt", "a = 1  \nb = 2\n---\na = 1\t\nb = 2\n", "a = 1\nb = 2", "a = 3\nb = 4"),
    ("m4.txt", "\ta = 1\n\tb = 2\n---\n    a = 1\n    b = 2\n", "a = 1\nb = 2", "a = 3\nb = 4"),
    ("m5.txt", "x = f(1,  2)\nx = f(1, 2)\n", "x  =  f(1, 2)", "x = g(1, 2)"),
])
def test_failed_edit_never_writes(work, name, body, old, new):
    p = put(work, name, body)
    out = edit(work, name, old, new)
    assert out.startswith("ERROR:")
    assert p.read_text() == body
    assert MATCH_STATS["fail"] == 1


def test_tolerant_stages_never_match_across_a_line_boundary(work):
    p = put(work, "x.txt", "zz a = 1  \nzz b = 2  \n")
    out = edit(work, "x.txt", "a = 1\nzz b", "c")
    assert out.startswith("ERROR:")
    assert p.read_text() == "zz a = 1  \nzz b = 2  \n"
    assert MATCH_STATS["fail"] == 1


def test_tolerant_stage_leaves_surrounding_lines_alone(work):
    body = "header = 1\n\n  a = 1\n  b = 2\n\ntrailer = 2\n"
    p = put(work, "s.txt", body)
    out = edit(work, "s.txt", "a = 1\nb = 2", "a = 3\nb = 4")
    assert "indentation-insensitive" in out
    assert p.read_text() == "header = 1\n\n  a = 3\n  b = 4\n\ntrailer = 2\n"
