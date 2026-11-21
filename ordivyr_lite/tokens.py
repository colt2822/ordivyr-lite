"""Dependency-free token accounting with per-family calibration and cache alignment.

No tokenizer vocabulary ships with Lite: every estimate is derived from the
character classes of the text (letters, digits, punctuation, whitespace, CJK)
against a per-provider-family cost table, then corrected by the observed ratio
between earlier estimates and real provider usage.  ``align_cache`` rounds an
estimate down to the provider's prefix-cache block so a cache-aware caller is
never optimistic.  Importing this module pulls in nothing beyond the standard
library and no other Lite module.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

FAMILIES: tuple[str, ...] = ("openai", "claude", "deepseek", "qwen", "generic")

CACHE_BLOCK: dict[str, int] = {
    "openai": 128,
    "deepseek": 64,
    "claude": 1024,
    "qwen": 32,
    "generic": 64,
}

MESSAGE_OVERHEAD = 4
TOOL_OVERHEAD = 8
TOOL_CALL_OVERHEAD = 8

_MIN_SCALE = 0.5
_MAX_SCALE = 2.0

# Cost per character in thousandths of a token.  Exact integer arithmetic keeps
# the estimates reproducible and monotonic: no float summation, no rounding drift.
_BASE_COSTS: dict[str, int] = {
    "letter": 250,      # ~4 ASCII letters per token
    "digit": 340,       # digit runs split more often than letter runs
    "space": 250,       # space + word merges
    "newline": 500,     # a line break rarely shares a token with the next word
    "punct": 330,       # brackets and operators merge only in short runs
    "cjk": 900,         # most modern vocabularies hold whole ideographs
    "other": 550,       # accents, emoji and control characters fall back to bytes
}

_FAMILY_COSTS: dict[str, dict[str, int]] = {
    "generic": {},
    "openai": {"digit": 330, "cjk": 1000},
    "claude": {"letter": 280, "digit": 360, "space": 270, "punct": 350, "cjk": 1100},
    "deepseek": {"letter": 260, "digit": 340, "punct": 340, "cjk": 750},
    "qwen": {"letter": 270, "digit": 360, "space": 260, "punct": 350, "cjk": 650},
}

_COSTS: dict[str, dict[str, int]] = {
    family: {**_BASE_COSTS, **overrides} for family, overrides in _FAMILY_COSTS.items()
}

_CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x1100, 0x11FF),   # Hangul Jamo
    (0x2E80, 0x2EFF),   # CJK radicals
    (0x3000, 0x303F),   # CJK punctuation
    (0x3040, 0x30FF),   # Hiragana and Katakana
    (0x3400, 0x4DBF),   # CJK extension A
    (0x4E00, 0x9FFF),   # CJK unified ideographs
    (0xAC00, 0xD7AF),   # Hangul syllables
    (0xF900, 0xFAFF),   # CJK compatibility ideographs
    (0xFF00, 0xFFEF),   # fullwidth forms
)

_MODEL_HINTS: tuple[tuple[str, str], ...] = (
    ("claude", "claude"),
    ("deepseek", "deepseek"),
    ("qwen", "qwen"),
    ("qwq", "qwen"),
    ("gpt", "openai"),
    ("chatgpt", "openai"),
    ("davinci", "openai"),
    ("o1-", "openai"),
    ("o3-", "openai"),
    ("o4-", "openai"),
)

_PROVIDER_HINTS: dict[str, str] = {
    "openai": "openai",
    "anthropic": "claude",
    "claude": "claude",
    "deepseek": "deepseek",
    "qwen": "qwen",
    "dashscope": "qwen",
    "alibaba": "qwen",
    "aliyun": "qwen",
}

_URL_HINTS: tuple[tuple[str, str], ...] = (
    ("anthropic", "claude"),
    ("deepseek", "deepseek"),
    ("dashscope", "qwen"),
    ("qwen", "qwen"),
    ("openai", "openai"),
)


def _family(family: str) -> str:
    return family if family in _COSTS else "generic"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str,
                          ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def _classify(ch: str) -> str:
    code = ord(ch)
    if code < 128:
        if 97 <= code <= 122 or 65 <= code <= 90:
            return "letter"
        if 48 <= code <= 57:
            return "digit"
        if ch in " \t":
            return "space"
        if ch in "\n\r":
            return "newline"
        return "punct"
    for low, high in _CJK_RANGES:
        if low <= code <= high:
            return "cjk"
    return "other"


def family_for(model: str, provider: str = "", base_url: str = "") -> str:
    """Best-effort family from the model id, then the provider, then the endpoint."""
    name = (model or "").strip().lower()
    for hint, family in _MODEL_HINTS:
        if hint in name:
            return family
    key = (provider or "").strip().lower()
    if key in _PROVIDER_HINTS:
        return _PROVIDER_HINTS[key]
    url = (base_url or "").strip().lower()
    for hint, family in _URL_HINTS:
        if hint in url:
            return family
    return "generic"


def count_text(text: str, family: str = "generic") -> int:
    """Tokens in one string: every character priced by class, rounded up once."""
    if not text:
        return 0
    costs = _COSTS[_family(family)]
    total = 0
    for ch in text:
        total += costs[_classify(ch)]
    return -(-total // 1000)


def _content_tokens(content: Any, family: str) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return count_text(content, family)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict):
                total += MESSAGE_OVERHEAD + count_text(_as_text(part.get("text") or ""), family)
            else:
                total += count_text(_as_text(part), family)
        return total
    return count_text(_as_text(content), family)


def _tool_call_tokens(call: Any, family: str) -> int:
    if not isinstance(call, dict):
        return TOOL_CALL_OVERHEAD + count_text(_as_text(call), family)
    function = call.get("function")
    function = function if isinstance(function, dict) else {}
    arguments = function.get("arguments")
    return (TOOL_CALL_OVERHEAD
            + count_text(str(function.get("name") or ""), family)
            + count_text("" if arguments is None else _as_text(arguments), family))


def count_messages(messages: list[dict], family: str = "generic") -> int:
    """Tokens for one chat history, including the JSON envelope of each message."""
    fam = _family(family)
    total = 0
    for message in messages or ():
        if not isinstance(message, dict):
            total += count_text(_as_text(message), fam)
            continue
        total += MESSAGE_OVERHEAD + count_text(str(message.get("role") or ""), fam)
        total += _content_tokens(message.get("content"), fam)
        for field in ("name", "tool_call_id"):
            value = message.get(field)
            if isinstance(value, str) and value:
                total += count_text(value, fam)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            for call in calls:
                total += _tool_call_tokens(call, fam)
    return total


def count_request(messages: list[dict], tools: list[dict] | None = None,
                  family: str = "generic") -> int:
    """Tokens for a whole request: the history plus every tool schema sent with it."""
    fam = _family(family)
    total = count_messages(messages, fam)
    for tool in tools or ():
        total += TOOL_OVERHEAD + count_text(_as_text(tool), fam)
    return total


def align_cache(tokens: int, family: str = "generic") -> int:
    """Round ``tokens`` DOWN to a prefix-cache block boundary, never below one block."""
    block = CACHE_BLOCK[_family(family)]
    return max(1, _int(tokens) // block) * block


class Calibrator:
    """Per-family correction learned from the ratio of real usage to estimates.

    The factor is the aggregate ratio of observed tokens to estimated tokens, so
    consistent samples converge in one step and no single sample can swing it.
    It is clamped to ``MIN_SCALE..MAX_SCALE``.  The cache file is best effort: a
    missing, corrupt or unreadable file starts from an uncalibrated factor of 1.0.
    """

    VERSION = 1
    MIN_SCALE = _MIN_SCALE
    MAX_SCALE = _MAX_SCALE

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        self._estimated: dict[str, int] = {}
        self._actual: dict[str, int] = {}
        self._load()

    def observe(self, family: str, estimated: int, actual: int) -> None:
        """Record one real sample; a non-positive measurement is ignored."""
        fam = _family(family)
        est, act = _int(estimated), _int(actual)
        if est <= 0 or act <= 0:
            return
        with self._lock:
            self._estimated[fam] = self._estimated.get(fam, 0) + est
            self._actual[fam] = self._actual.get(fam, 0) + act

    def scale(self, family: str) -> float:
        """Clamped multiplier to apply to an estimate for ``family``."""
        fam = _family(family)
        with self._lock:
            est = self._estimated.get(fam, 0)
            act = self._actual.get(fam, 0)
        return _scale_of(est, act)

    def estimate(self, messages: list[dict], tools: list[dict] | None = None,
                 family: str = "generic") -> int:
        """Calibrated token estimate for one request."""
        fam = _family(family)
        base = count_request(messages, tools, fam)
        if base <= 0:
            return 0
        return max(1, _int(round(base * self.scale(fam))))

    def as_dict(self) -> dict:
        with self._lock:
            estimated = dict(self._estimated)
            actual = dict(self._actual)
        return {
            "version": self.VERSION,
            "path": str(self.path) if self.path else None,
            "families": {
                fam: {"estimated": estimated[fam], "actual": actual[fam],
                      "scale": _scale_of(estimated[fam], actual[fam])}
                for fam in sorted(estimated)
            },
        }

    def save(self) -> None:
        """Persist the calibration atomically; a missing or unwritable path is a no-op."""
        if self.path is None:
            return
        payload = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.path)
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(data, dict) or _int(data.get("version")) != self.VERSION:
            return
        families = data.get("families")
        if not isinstance(families, dict):
            return
        for name, entry in families.items():
            if name not in _COSTS or not isinstance(entry, dict):
                continue
            est, act = _int(entry.get("estimated")), _int(entry.get("actual"))
            if est > 0 and act > 0:
                self._estimated[name] = est
                self._actual[name] = act


def _scale_of(estimated: int, actual: int) -> float:
    if estimated <= 0:
        return 1.0
    return round(min(_MAX_SCALE, max(_MIN_SCALE, actual / estimated)), 4)


_BYTE_ENCODER: dict[int, str] | None = None


def _byte_encoder() -> dict[int, str]:
    """GPT-2 style byte -> printable-symbol map, built once on first use."""
    global _BYTE_ENCODER
    if _BYTE_ENCODER is None:
        printable = (list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAC))
                     + list(range(0xAE, 0x100)))
        symbols = printable[:]
        extra = 0
        for byte in range(256):
            if byte not in printable:
                printable.append(byte)
                symbols.append(0x100 + extra)
                extra += 1
        _BYTE_ENCODER = dict(zip(printable, map(chr, symbols)))
    return _BYTE_ENCODER


class _ByteBPE:
    """Byte-level BPE driven by the merge table of a HuggingFace ``tokenizer.json``.

    Merges are applied greedily and across the whole buffer, so the result is a
    close count of the real tokenizer rather than an exact id sequence.  Text is
    counted in fixed-size chunks to keep the quadratic merge loop bounded.
    """

    CHUNK = 1024

    def __init__(self, merges: list):
        self._ranks: dict[tuple[str, str], int] = {}
        for index, merge in enumerate(merges):
            if isinstance(merge, str):
                parts = merge.split(" ")
            elif isinstance(merge, (list, tuple)):
                parts = [str(part) for part in merge]
            else:
                continue
            if len(parts) == 2:
                self._ranks.setdefault((parts[0], parts[1]), index)

    def count(self, text: str) -> int:
        """Tokens in ``text``; empty text costs nothing."""
        if not text:
            return 0
        encoder = _byte_encoder()
        total = 0
        for start in range(0, len(text), self.CHUNK):
            chunk = text[start:start + self.CHUNK]
            total += len(self._merge([encoder[byte] for byte in chunk.encode("utf-8")]))
        return total

    def _merge(self, symbols: list[str]) -> list[str]:
        ranks = self._ranks
        while len(symbols) > 1:
            best_rank: int | None = None
            best_at = -1
            for index in range(len(symbols) - 1):
                rank = ranks.get((symbols[index], symbols[index + 1]))
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_at = index
            if best_at < 0:
                break
            symbols[best_at:best_at + 2] = [symbols[best_at] + symbols[best_at + 1]]
        return symbols


def load_tokenizer(model_dir: Any) -> object | None:
    """Load a byte-level BPE from ``<model_dir>/tokenizer.json`` when it is usable.

    Returns ``None`` for a missing directory, an unreadable or malformed file, or
    a model this dependency-free loader cannot represent (anything that is not a
    byte-level BPE vocabulary with a merge table).
    """
    if not model_dir:
        return None
    path = Path(model_dir)
    if path.is_dir():
        path = path / "tokenizer.json"
    try:
        data = json.loads(path.read_text("utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    model = data.get("model")
    if not isinstance(model, dict) or str(model.get("type") or "").lower() != "bpe":
        return None
    vocab = model.get("vocab")
    merges = model.get("merges")
    if not isinstance(vocab, dict) or not vocab or not isinstance(merges, list):
        return None
    return _ByteBPE(merges)


def tokenizer_info() -> dict:
    """Static description of the estimator, for diagnostics and CLI banners."""
    return {
        "kind": "character-class heuristic",
        "version": 1,
        "families": list(FAMILIES),
        "cache_block": dict(CACHE_BLOCK),
        "message_overhead": MESSAGE_OVERHEAD,
        "tool_overhead": TOOL_OVERHEAD,
        "costs": {family: dict(_COSTS[family]) for family in FAMILIES},
        "tokenizer": None,
    }
