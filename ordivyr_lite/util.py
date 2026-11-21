"""Small stdlib-only helpers used across Lite (kept dependency-free for fast startup)."""

from __future__ import annotations

import json
import math
import os
import re
import secrets
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def new_run_id(prefix: str = "run") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{prefix}-{stamp}-{secrets.token_hex(3)}"


def estimate_tokens(text: Any) -> int:
    """Cheap token proxy (~4 chars/token).  Used only for budgeting/estimates;
    real provider usage always wins once a response arrives."""
    if text is None:
        return 0
    if not isinstance(text, str):
        text = json.dumps(text, default=str)
    return max(1, math.ceil(len(text) / 4))


def messages_chars(messages: list[dict]) -> int:
    try:
        return len(json.dumps(messages, separators=(",", ":"), default=str))
    except (TypeError, ValueError):
        return sum(len(str(m)) for m in messages)


def clip(text: Any, limit: int = 500) -> str:
    s = str(text)
    return s if len(s) <= limit else s[: max(0, limit - 20)] + "…[truncated]"


def preview(value: Any, limit: int = 500) -> str:
    try:
        s = json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError):
        s = str(value)
    return clip(s, limit)


def redact_key(key: str | None) -> str:
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return key[:3] + "…" + key[-3:]


def ensure_private_dir(path: Path) -> Path:
    path = Path(path).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(path, 0o700)
    except OSError:
        pass
    return path


def atomic_write_text(path: Path, text: str, mode: int = 0o600,
                    fsync: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as f:
            f.write(text)
            f.flush()
            if fsync:
                os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def file_contains(path: Path, needle: str) -> bool:
    try:
        return needle in Path(path).read_text("utf-8", "replace")
    except OSError:
        return False


def is_loopback(hostname: str | None) -> bool:
    h = (hostname or "").lower().strip("[]")
    return h in {"127.0.0.1", "localhost", "::1"} or h.startswith("127.")


def human_int(n: int | float | None) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def human_usd(v: float | None) -> str:
    if v is None:
        return "n/a"
    if v == 0:
        return "$0"
    if abs(v) < 0.01:
        return f"${v:.6f}"
    return f"${v:.4f}"


def human_ms(ms: float | None) -> str:
    if ms is None:
        return "n/a"
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.2f}s"


def truncate_middle(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    tail = limit - head
    return f"{text[:head]}\n[... {len(text) - limit} chars omitted ...]\n{text[-tail:]}"


_PKG_VERSION_RE = re.compile(r"^\d+(\.\d+)+")


def monotonic() -> float:
    return time.monotonic()


def stderr(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)
