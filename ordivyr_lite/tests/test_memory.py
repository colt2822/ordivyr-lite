from __future__ import annotations

import copy
import hashlib
import json
import os
import time

import pytest

from ordivyr_lite.memory import CodeIndex, ContextStore, compress, rank_texts
from ordivyr_lite.util import messages_chars


@pytest.fixture
def tree(tmp_path):
    root = tmp_path / "ws"
    (root / "src").mkdir(parents=True)
    (root / "src" / "invoices.py").write_text("\n".join([
        '"""Invoice handling: VAT and totals."""',
        "",
        "VAT_RATE = 0.2",
        "",
        "",
        "def invoice_total(items):",
        "    total = 0.0",
        "    for item in items:",
        "        total += item",
        "    return total * (1 + VAT_RATE)",
        "",
    ]))
    (root / "src" / "colors.py").write_text("\n".join([
        '"""Color palette helpers: named colours and nearest-match lookup."""',
        "",
        'PALETTE = {"red": "#f00", "green": "#0f0"}',
        "",
        "",
        "def nearest(rgb):",
        "    return min(PALETTE, key=lambda name: abs(len(name) - len(rgb)))",
        "",
    ]))
    (root / "README.md").write_text("\n".join([
        "# Example project",
        "",
        "Licensed under the MIT license. See the handbook for details.",
        "",
    ]))
    return root


def make_index(root, tmp_path, **kwargs) -> CodeIndex:
    return CodeIndex(root, cache_dir=tmp_path / "cache", **kwargs)


def test_search_ranks_relevant_file_first(tree, tmp_path):
    index = make_index(tree, tmp_path)
    index.refresh()
    hits = index.search("invoice vat total")
    paths = [hit["path"] for hit in hits]
    assert paths[0] == "src/invoices.py"
    assert "README.md" not in paths
    assert "src/colors.py" not in paths
    assert index.search("color palette")[0]["path"] == "src/colors.py"


def test_search_returns_path_score_line_snippet(tree, tmp_path):
    index = make_index(tree, tmp_path)
    index.refresh()
    hit = index.search("invoice vat total")[0]
    assert set(hit) == {"path", "score", "line", "snippet"}
    assert isinstance(hit["score"], float) and 0.0 < hit["score"] <= 1.0
    assert hit["line"] >= 1
    assert "invoice" in hit["snippet"].lower()


def test_source_has_no_result_for_unrelated_words(tree, tmp_path):
    index = make_index(tree, tmp_path)
    index.refresh()
    assert index.search("kubernetes helm chart") == []


def test_empty_or_whitespace_query_returns_empty(tree, tmp_path):
    index = make_index(tree, tmp_path)
    index.refresh()
    assert index.search("") == []
    assert index.search("   ") == []
    assert index.search("\t ") == []
    assert index.search("invoice", k=0) == []


def test_search_k_bounds_results(tree, tmp_path):
    index = make_index(tree, tmp_path)
    index.refresh()
    assert len(index.search("invoice vat total color palette", k=1)) == 1
    assert len(index.search("invoice vat total color palette", k=2)) == 2


def test_incremental_refresh_reindexes_only_changed_files(tree, tmp_path):
    index = make_index(tree, tmp_path)
    first = index.refresh()
    total = first["files"]
    assert first == {"files": total, "reindexed": total, "skipped": 0}
    assert index.refresh() == {"files": total, "reindexed": 0, "skipped": total}
    target = tree / "src" / "colors.py"
    target.write_text("changed color palette helpers\n")
    stamp = time.time_ns() + 1_000_000_000
    os.utime(target, ns=(stamp, stamp))
    assert index.refresh() == {"files": total, "reindexed": 1, "skipped": total - 1}
    assert index.search("changed palette")[0]["path"] == "src/colors.py"


def test_refresh_picks_up_new_and_deleted_files(tree, tmp_path):
    index = make_index(tree, tmp_path)
    total = index.refresh()["files"]
    extra = tree / "src" / "shipping.py"
    extra.write_text("def shipping_cost(weight):\n    return weight * 3\n")
    assert index.refresh() == {"files": total + 1, "reindexed": 1, "skipped": total}
    assert index.search("shipping cost weight")[0]["path"] == "src/shipping.py"
    extra.unlink()
    assert index.refresh() == {"files": total, "reindexed": 0, "skipped": total}
    assert "shipping.py" not in [hit["path"] for hit in index.search("shipping cost")]


def test_cache_reload_produces_identical_search_results(tree, tmp_path):
    cache = tmp_path / "cache"
    first = CodeIndex(tree, cache_dir=cache)
    first.refresh()
    expected = first.search("invoice vat total", k=5)
    expected_outline = first.outline()
    second = CodeIndex(tree, cache_dir=cache)
    assert second.stats()["cache_loaded"] is True
    assert second.search("invoice vat total", k=5) == expected
    assert second.outline() == expected_outline
    assert second.refresh()["reindexed"] == 0
    assert second.search("invoice vat total", k=5) == expected


def test_index_cache_is_private_and_named_by_root_hash(tree, tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path / "home"))
    index = CodeIndex(tree)
    index.refresh()
    key = hashlib.sha256(str(tree.resolve()).encode("utf-8")).hexdigest()[:16]
    assert index.cache_path == tmp_path / "home" / "index" / f"{key}.json"
    assert index.cache_path.exists()
    assert index.cache_path.stat().st_mode & 0o777 == 0o600


def test_corrupt_index_cache_is_tolerated(tree, tmp_path):
    index = make_index(tree, tmp_path)
    total = index.refresh()["files"]
    index.cache_path.write_text("{ this is not json")
    reloaded = make_index(tree, tmp_path)
    assert reloaded.refresh() == {"files": total, "reindexed": total, "skipped": 0}
    index.cache_path.write_text(json.dumps({"version": 1, "root": str(tree.resolve()),
                                            "files": {"../etc/passwd": {"mtime": 1, "size": 1,
                                                      "counts": {"invoice": 3}}}}))
    poisoned = make_index(tree, tmp_path)
    assert "etc/passwd" not in str(poisoned.stats())
    assert poisoned.refresh()["files"] == total


def test_symlink_outside_root_is_never_indexed(tree, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("auditsecret token vault\n")
    (tree / "linked.py").symlink_to(outside / "secret.py")
    (tree / "linkdir").symlink_to(outside)
    index = make_index(tree, tmp_path)
    index.refresh()
    assert index.search("auditsecret token vault") == []
    assert "linked.py" not in index.outline()
    assert "linkdir" not in index.outline()
    assert (outside / "secret.py").read_text() == "auditsecret token vault\n"


def test_oversized_and_binary_files_are_skipped(tree, tmp_path):
    (tree / "huge.py").write_bytes(b"x = 1\n" * 400)
    (tree / "blob.bin").write_bytes(b"invoice vat total" + bytes([0, 1, 2, 3]))
    index = make_index(tree, tmp_path, max_bytes=1000)
    result = index.refresh()
    stats = index.stats()
    assert result["files"] == 3
    assert stats["skipped_large"] == 1
    assert stats["skipped_binary"] == 1
    assert index.search("blob") == []
    assert "blob.bin" not in index.outline()
    assert "huge.py" not in index.outline()


def test_build_output_and_vcs_dirs_are_skipped(tree, tmp_path):
    for rel in (".git/config", "node_modules/pkg/index.js", "build/out.py", "__pycache__/x.pyc"):
        path = tree / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("skippedmarker payload\n")
    index = make_index(tree, tmp_path)
    index.refresh()
    assert index.search("skippedmarker payload") == []
    outline = index.outline()
    for rel in ("config", "node_modules", "build/", "__pycache__"):
        assert rel not in outline


def test_outline_is_bounded_and_lists_project_layout(tree, tmp_path):
    for i in range(30):
        package = tree / "pkg" / f"mod{i:02d}"
        package.mkdir(parents=True)
        for j in range(6):
            (package / f"file_{j}.py").write_text(f"def helper_{i}_{j}():\n    return {j}\n")
    index = make_index(tree, tmp_path)
    index.refresh()
    outline = index.outline()
    assert outline.startswith("ws: ")
    assert "files indexed" in outline
    assert "pkg/" in outline
    assert len(outline) <= 1200
    assert len(index.outline(limit=2)) <= 1200
    assert len(index.outline(limit=2)) < len(outline)


def test_two_identical_runs_order_identically(tree, tmp_path):
    left = CodeIndex(tree, cache_dir=tmp_path / "left")
    right = CodeIndex(tree, cache_dir=tmp_path / "right")
    left.refresh()
    right.refresh()
    for query in ("invoice vat total", "color palette", "license handbook"):
        assert left.search(query, k=8) == right.search(query, k=8)
    assert left.outline() == right.outline()
    assert left.search("invoice vat total") == left.search("invoice vat total")


def test_rank_texts_prefers_full_match_and_is_deterministic():
    items = [("full", "invoice vat total amount due"),
             ("partial", "invoice template header"),
             ("noise", "colour palette table")]
    ranked = rank_texts("invoice vat total", items, 3)
    assert [key for key, _text, _score in ranked][0] == "full"
    assert "noise" not in [key for key, _text, _score in ranked]
    assert all(0.0 < score <= 1.0 for _key, _text, score in ranked)
    assert ranked == rank_texts("invoice vat total", items, 3)
    assert len(rank_texts("invoice", items, 1)) == 1
    assert rank_texts("", items, 3) == []
    assert rank_texts("invoice", [], 3) == []
    assert rank_texts("invoice", items, 0) == []


def history(chars: int = 3000, tools: int = 5) -> list[dict]:
    messages: list[dict] = [{"role": "system", "content": "system prompt"}]
    for i in range(tools):
        messages.append({"role": "assistant", "content": f"step {i}", "tool_calls": [
            {"id": f"call_{i}", "type": "function",
             "function": {"name": "read_file", "arguments": "{}"}}]})
        messages.append({"role": "tool", "tool_call_id": f"call_{i}",
                         "content": f"result {i}\n" + ("x" * chars)})
    messages.append({"role": "user", "content": "finish it"})
    messages.append({"role": "assistant", "content": "done"})
    messages.append({"role": "tool", "tool_call_id": "tail", "content": "z" * 40})
    return messages


def test_compress_shrinks_old_tool_results_only():
    messages = history()
    before = copy.deepcopy(messages)
    removed = compress(messages, budget_tokens=500, protect=3)
    assert removed > 0
    assert messages[0] == before[0]
    assert messages[-3:] == before[-3:]
    for old, new in zip(before, messages):
        if old["role"] != "tool":
            assert old == new
    tool_before = sum(len(m["content"]) for m in before if m["role"] == "tool")
    tool_after = sum(len(m["content"]) for m in messages if m["role"] == "tool")
    assert tool_after == tool_before - removed
    assert messages_chars(messages) <= 500 * 4
    assert messages_chars(messages) < messages_chars(before)
    assert [m["tool_call_id"] for m in messages if m["role"] == "tool"] == [
        "call_0", "call_1", "call_2", "call_3", "call_4", "tail"]


def test_compress_respects_first_and_protected_messages():
    messages = history(tools=2)
    before = copy.deepcopy(messages)
    assert compress(messages, budget_tokens=10, protect=len(messages)) == 0
    assert messages == before
    assert compress(messages, budget_tokens=10, protect=1) > 0
    assert messages[0] == before[0]
    assert messages[-1] == before[-1]


def test_compress_is_a_noop_within_budget():
    messages = history(chars=10, tools=1)
    before = copy.deepcopy(messages)
    assert compress(messages, budget_tokens=100_000, protect=3) == 0
    assert messages == before
    assert compress([], budget_tokens=100, protect=3) == 0


def test_compress_families_stay_bounded_and_never_touch_protected():
    for family in ("generic", "shell", "test", "json", "not-a-family"):
        messages = history(chars=2000, tools=3)
        before = copy.deepcopy(messages)
        removed = compress(messages, budget_tokens=1500, protect=3, family=family)
        assert removed > 0
        assert messages[0] == before[0]
        assert messages[-3:] == before[-3:]
        assert messages_chars(messages) <= 1500 * 4


def test_context_store_recall_is_relevant(tree, tmp_path):
    store = ContextStore(tree, home=tmp_path / "home")
    store.remember(run_id="run-invoice", task="fix invoice vat total rounding",
                   files=["src/invoices.py"], errors=["KeyError: total"], outcome="resolved")
    store.remember(run_id="run-colors", task="rename colour palette entries",
                   files=["src/colors.py"], errors=[], outcome="resolved")
    text = store.recall("invoice vat rounding")
    assert "run-invoice" in text
    assert "src/invoices.py" in text
    assert "run-colors" not in text
    assert store.recall("zzz qqq unknown") == ""
    assert store.recall("") == ""
    assert store.recall("invoice", limit=0) == ""


def test_context_store_recall_is_bounded(tree, tmp_path):
    store = ContextStore(tree, home=tmp_path / "home")
    for i in range(40):
        store.remember(run_id=f"run-{i}", task="invoice vat total rounding followup " * 4,
                       files=[f"src/invoices_{i}.py"], errors=["KeyError: total"],
                       outcome="resolved")
    text = store.recall("invoice vat total", limit=3)
    assert 0 < len(text) <= 900
    assert text.count("- run-") <= 3
    assert store.as_dict()["count"] == 40


def test_context_store_tolerates_corrupt_store(tree, tmp_path):
    store = ContextStore(tree, home=tmp_path / "home")
    store.remember(run_id="run-1", task="invoice vat total", files=["src/invoices.py"],
                   errors=[], outcome="resolved")
    store.path.write_text("{ not json at all")
    fresh = ContextStore(tree, home=tmp_path / "home")
    assert fresh.as_dict()["count"] == 0
    assert fresh.recall("invoice vat total") == ""
    fresh.remember(run_id="run-2", task="invoice vat total", files=[], errors=[],
                   outcome="resolved")
    assert fresh.as_dict()["count"] == 1
    store.path.write_text(json.dumps({"version": 1, "root": "/some/other/root",
                                      "entries": [{"run_id": "x", "task": "invoice"}]}))
    assert ContextStore(tree, home=tmp_path / "home").as_dict()["count"] == 0


def test_context_store_does_not_leak_across_roots(tree, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    (other / "a.py").write_text("other project\n")
    home = tmp_path / "home"
    first = ContextStore(tree, home=home)
    first.remember(run_id="run-1", task="invoice vat total", files=["src/invoices.py"],
                   errors=[], outcome="resolved")
    second = ContextStore(other, home=home)
    assert second.path != first.path
    assert second.as_dict()["count"] == 0
    assert second.recall("invoice vat total") == ""
    assert first.recall("invoice vat total") != ""


def test_context_store_home_and_payload_shape(tree, tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path / "envhome"))
    store = ContextStore(tree)
    assert store.path.parent == tmp_path / "envhome" / "context"
    store.remember(run_id="run-1", task="invoice vat total", files="src/invoices.py",
                   errors=None, outcome=None)
    payload = store.as_dict()
    assert set(payload) == {"root", "store", "count", "entries"}
    assert payload["count"] == 1
    entry = payload["entries"][0]
    assert entry["run_id"] == "run-1"
    assert entry["files"] == ["src/invoices.py"]
    assert entry["errors"] == []
    assert store.path.exists()
    assert store.path.stat().st_mode & 0o777 == 0o600
