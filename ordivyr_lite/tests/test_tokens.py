from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from ordivyr_lite.tokens import (
    CACHE_BLOCK,
    FAMILIES,
    Calibrator,
    align_cache,
    count_messages,
    count_request,
    count_text,
    family_for,
    load_tokenizer,
    tokenizer_info,
)

PROSE = ("The quick brown fox jumps over the lazy dog while the engineer reads the log, "
         "checks the invariant and writes one short sentence about what actually happened. ")
SOURCE = ("def add(a, b):\n"
          "    total = a + b\n"
          "    if total > 0:\n"
          "        return {'value': total, 'ok': True}\n"
          "    raise ValueError('negative total: %r' % (total,))\n"
          "\n")
CJK = "上下文窗口的令牌估算需要保留前缀缓存边界并记录真实的用量数据。"


def repeat_to(text: str, size: int) -> str:
    return (text * (size // len(text) + 1))[:size]


MESSAGES = [{"role": "user", "content": "list the files"}]
TOOL = {"type": "function",
        "function": {"name": "list_files", "description": "List files in a directory.",
                     "parameters": {"type": "object",
                                    "properties": {"path": {"type": "string"}},
                                    "required": ["path"]}}}


def test_contract_constants():
    assert FAMILIES == ("openai", "claude", "deepseek", "qwen", "generic")
    assert CACHE_BLOCK == {"openai": 128, "deepseek": 64, "claude": 1024, "qwen": 32, "generic": 64}


@pytest.mark.parametrize("model,family", [
    ("deepseek-chat", "deepseek"),
    ("deepseek-reasoner", "deepseek"),
    ("qwen2.5-coder", "qwen"),
    ("Qwen3-Coder", "qwen"),
    ("claude-sonnet-4", "claude"),
    ("gpt-4.1", "openai"),
    ("o3-mini", "openai"),
    ("llama-3.1-8b", "generic"),
    ("totally-unknown", "generic"),
    ("", "generic"),
])
def test_family_detection_from_model(model, family):
    assert family_for(model) == family


def test_family_detection_from_provider_and_base_url():
    assert family_for("mystery", provider="openai") == "openai"
    assert family_for("mystery", provider="anthropic") == "claude"
    assert family_for("mystery", provider="deepseek") == "deepseek"
    assert family_for("mystery", provider="dashscope") == "qwen"
    assert family_for("mystery", base_url="https://api.anthropic.com") == "claude"
    assert family_for("mystery", base_url="https://api.deepseek.com/v1") == "deepseek"
    assert family_for("mystery", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1") == "qwen"
    assert family_for("mystery", base_url="http://127.0.0.1:11434/v1") == "generic"
    assert family_for("claude-sonnet-4", provider="openai", base_url="https://api.openai.com/v1") == "claude"


@pytest.mark.parametrize("family", FAMILIES)
def test_estimates_stay_in_the_plausible_band(family):
    assert 800 <= count_text(repeat_to(PROSE, 4000), family) <= 1400
    assert 900 <= count_text(repeat_to(SOURCE, 4000), family) <= 1600
    assert 1200 <= count_text(repeat_to(CJK, 2000), family) <= 3200


def test_character_classes_are_priced_separately():
    ascii_text = repeat_to("The quick brown fox jumps over the lazy dog. ", 400)
    cjk_text = repeat_to(CJK, 400)
    assert count_text(cjk_text) > 2 * count_text(ascii_text)
    assert count_text(repeat_to(SOURCE, 2000)) > count_text(repeat_to(PROSE, 2000))
    assert count_text("1111") > count_text("aaaa")
    assert count_text("\u00e9\u00e9\u00e9\u00e9") > count_text("aaaa")


def test_count_text_edges():
    assert count_text("") == 0
    assert count_text("   ") >= 1
    assert count_text("hello", "nope") == count_text("hello", "generic")


def test_count_messages_counts_envelope_overhead():
    message = {"role": "user", "content": "hello"}
    assert count_messages([]) == 0
    assert count_messages([message]) > count_text("hello")
    assert count_messages([message, message]) == 2 * count_messages([message])


def test_count_messages_handles_tool_calls_and_structured_content():
    messages = [
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "print('x')"},
    ]
    assert count_messages(messages) > count_text("print('x')")
    blocks = [{"type": "text", "text": "hello"}]
    assert count_messages([{"role": "user", "content": blocks}]) > 0
    assert count_messages([{"role": "user", "content": {"answer": "hello"}}]) > 0
    assert count_messages([{"role": "user", "content": "hello", "name": "bob"}]) > count_messages(
        [{"role": "user", "content": "hello"}])


def test_count_request_includes_tool_schemas():
    bare = count_request(MESSAGES)
    assert bare == count_messages(MESSAGES)
    assert count_request(MESSAGES, []) == bare
    assert count_request(MESSAGES, None) == bare
    with_tool = count_request(MESSAGES, [TOOL])
    assert with_tool > bare
    bigger = {"type": "function",
              "function": {"name": "list_files",
                           "description": "List files in a directory, recursively, with sizes.",
                           "parameters": TOOL["function"]["parameters"]}}
    assert count_request(MESSAGES, [bigger]) > with_tool
    assert count_request(MESSAGES, [TOOL, TOOL]) > with_tool


def test_count_request_is_monotonic_in_content_length():
    history = [{"role": "system", "content": "be brief"},
               {"role": "user", "content": "abc"}]
    previous = count_request(history)
    for size in (1, 2, 5, 10, 50, 200):
        current = count_request([history[0], {"role": "user", "content": "x" * size}])
        assert current >= previous
        previous = current
    longer_history = history + [{"role": "assistant", "content": "ok"}]
    assert count_request(longer_history) > count_request(history)
    assert count_request(history, [TOOL]) >= count_request(history)


@pytest.mark.parametrize("tokens,family,expected", [
    (0, "generic", 64),
    (1, "generic", 64),
    (63, "generic", 64),
    (64, "generic", 64),
    (127, "generic", 64),
    (128, "generic", 128),
    (1, "openai", 128),
    (100, "openai", 128),
    (300, "openai", 256),
    (1000, "openai", 896),
    (63, "deepseek", 64),
    (200, "deepseek", 192),
    (31, "qwen", 32),
    (100, "qwen", 96),
    (1023, "claude", 1024),
    (1024, "claude", 1024),
    (2047, "claude", 1024),
    (2048, "claude", 2048),
])
def test_align_cache_boundaries(tokens, family, expected):
    assert align_cache(tokens, family) == expected


def test_align_cache_is_monotonic_and_block_aligned():
    previous = 0
    for tokens in (0, 1, 31, 32, 33, 100, 127, 128, 129, 500, 4096):
        aligned = align_cache(tokens, "openai")
        assert aligned >= previous
        assert aligned % CACHE_BLOCK["openai"] == 0
        assert aligned <= max(tokens, CACHE_BLOCK["openai"])
        previous = aligned


def test_calibrator_converges_on_consistent_samples():
    cal = Calibrator()
    assert cal.scale("openai") == 1.0
    for _ in range(40):
        cal.observe("openai", 1000, 800)
    assert cal.scale("openai") == pytest.approx(0.8, abs=1e-9)
    assert cal.scale("claude") == 1.0
    assert cal.as_dict()["families"]["openai"]["estimated"] == 40_000
    assert cal.as_dict()["families"]["openai"]["actual"] == 32_000


def test_calibrator_clamps_its_factor():
    cal = Calibrator()
    cal.observe("claude", 1000, 1)
    assert cal.scale("claude") == Calibrator.MIN_SCALE == 0.5
    cal.observe("qwen", 1000, 100_000)
    assert cal.scale("qwen") == Calibrator.MAX_SCALE == 2.0


def test_calibrator_ignores_unusable_samples():
    cal = Calibrator()
    cal.observe("generic", 0, 500)
    cal.observe("generic", -5, 500)
    cal.observe("generic", 100, 0)
    cal.observe("generic", "nonsense", "nonsense")
    assert cal.scale("generic") == 1.0
    assert cal.as_dict()["families"] == {}


def test_calibrator_estimate_applies_the_factor():
    cal = Calibrator()
    message = [{"role": "user", "content": "x" * 400}]
    base = count_request(message, family="openai")
    assert cal.estimate(message, family="openai") == base
    cal.observe("openai", 1000, 500)
    assert cal.estimate(message, family="openai") == max(1, round(base * 0.5))
    assert cal.estimate([], family="openai") == 0
    assert cal.estimate(message, [TOOL], family="openai") > cal.estimate(message, family="openai")


def test_calibrator_round_trips_through_save(tmp_path):
    path = tmp_path / "tokens.json"
    cal = Calibrator(path)
    cal.observe("deepseek", 1000, 900)
    cal.save()
    assert isinstance(json.loads(path.read_text("utf-8")), dict)
    again = Calibrator(path)
    assert again.scale("deepseek") == pytest.approx(0.9, abs=1e-9)
    assert again.as_dict()["families"]["deepseek"]["actual"] == 900
    assert again.as_dict()["path"] == str(path)


def test_calibrator_without_a_path_stays_in_memory():
    cal = Calibrator()
    cal.observe("qwen", 100, 50)
    cal.save()
    assert cal.scale("qwen") == 0.5
    assert cal.as_dict()["path"] is None


@pytest.mark.parametrize("payload", [
    "",
    "{not json",
    "[]",
    '{"version": 99, "families": {"openai": {"estimated": 1, "actual": 2}}}',
    '{"version": 1, "families": "nope"}',
    '{"version": 1, "families": {"openai": "nope"}}',
    '{"version": 1, "families": {"openai": {"estimated": "x", "actual": 2}}}',
])
def test_corrupt_cache_recovers(tmp_path, payload):
    path = tmp_path / "tokens.json"
    path.write_text(payload, "utf-8")
    cal = Calibrator(path)
    assert cal.scale("openai") == 1.0
    cal.observe("openai", 100, 50)
    assert cal.scale("openai") == 0.5
    cal.save()
    assert Calibrator(path).scale("openai") == 0.5


def test_unreadable_cache_path_recovers(tmp_path):
    cal = Calibrator(tmp_path)
    assert cal.scale("openai") == 1.0
    cal.observe("openai", 100, 200)
    cal.save()
    assert cal.scale("openai") == 2.0


def test_load_tokenizer_returns_none_for_unsupported_input(tmp_path):
    assert load_tokenizer(None) is None
    assert load_tokenizer("") is None
    assert load_tokenizer(tmp_path / "missing") is None
    target = tmp_path / "tokenizer.json"
    target.write_text("{broken", "utf-8")
    assert load_tokenizer(tmp_path) is None
    target.write_text(json.dumps({"model": {"type": "WordPiece", "vocab": {"a": 0},
                                           "merges": []}}), "utf-8")
    assert load_tokenizer(tmp_path) is None
    target.write_text(json.dumps({"model": {"type": "BPE", "vocab": {}, "merges": []}}), "utf-8")
    assert load_tokenizer(tmp_path) is None
    target.write_text(json.dumps({"model": {"type": "BPE", "vocab": {"a": 0},
                                           "merges": "h e"}}), "utf-8")
    assert load_tokenizer(tmp_path) is None
    target.write_text(json.dumps([1, 2, 3]), "utf-8")
    assert load_tokenizer(tmp_path) is None


def test_load_tokenizer_reads_a_byte_level_bpe(tmp_path):
    payload = {"model": {"type": "BPE",
                         "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "he": 4, "hel": 5,
                                   "hell": 6, "hello": 7, " ": 8},
                         "merges": ["h e", "he l", "hel l", "hell o"]}}
    (tmp_path / "tokenizer.json").write_text(json.dumps(payload), "utf-8")
    tokenizer = load_tokenizer(tmp_path)
    assert tokenizer is not None
    assert tokenizer.count("") == 0
    assert tokenizer.count("hello") == 1
    assert tokenizer.count("hello hello") == 3
    assert tokenizer.count("hellohello") == 2


def test_tokenizer_info_describes_the_estimator():
    info = tokenizer_info()
    assert info["version"] == 1
    assert tuple(info["families"]) == FAMILIES
    assert info["cache_block"] == CACHE_BLOCK
    assert set(info["costs"]) == set(FAMILIES)
    assert info["costs"]["qwen"]["cjk"] < info["costs"]["claude"]["cjk"]


def test_concurrent_observe_is_thread_safe():
    cal = Calibrator()

    def worker() -> None:
        for _ in range(50):
            cal.observe("openai", 100, 80)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert cal.scale("openai") == pytest.approx(0.8, abs=1e-9)
    assert cal.as_dict()["families"]["openai"]["estimated"] == 8 * 50 * 100
    assert cal.as_dict()["families"]["openai"]["actual"] == 8 * 50 * 80


def test_import_is_cheap_and_does_not_pull_config_or_loop():
    root = Path(__file__).resolve().parents[2]
    code = (
        "import sys, time\n"
        "start = time.perf_counter()\n"
        "import ordivyr_lite.tokens\n"
        "elapsed = (time.perf_counter() - start) * 1000.0\n"
        "forbidden = sorted(m for m in sys.modules\n"
        "                   if m in {'ordivyr_lite.config', 'ordivyr_lite.loop'})\n"
        "assert not forbidden, forbidden\n"
        "assert elapsed < 20.0, elapsed\n"
        "print(f'{elapsed:.3f}')\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
