"""One-level child runs. The delegate tool is absent until the operator enables it.

A child has its own messages and its own router. It cannot delegate. Parallel
children run only when every child is given an explicit read-only tool list.
The parent receives a short report, not the child's transcript.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

_WRITE_TOOLS = frozenset({"edit_file", "write_file", "run_command", "fetch_url", "delegate"})


class DelegateError(Exception):
    pass


@dataclass
class DelegateLimits:
    max_children: int = 4
    max_parallel: int = 2
    max_child_turns: int = 4
    max_child_seconds: float = 120.0
    max_result_chars: int = 4000


@dataclass
class ChildReport:
    ok: bool
    text: str
    stop_reason: str
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    model_calls: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    files: list[str] = field(default_factory=list)
    error: str | None = None


class Delegator:
    def __init__(self, *, spawn: Callable, limits: DelegateLimits | None = None,
                 cancel_event=None):
        self.spawn = spawn
        self.limits = limits or DelegateLimits()
        self.cancel_event = cancel_event
        self.used = 0
        self._lock = threading.Lock()

    def remaining(self) -> int:
        with self._lock:
            return max(0, self.limits.max_children - self.used)

    def _cancelled(self) -> bool:
        ev = self.cancel_event
        if ev is None:
            return False
        try:
            return bool(ev.is_set()) if hasattr(ev, "is_set") else bool(ev())
        except Exception:  # noqa: BLE001
            return False

    def _reserve(self) -> bool:
        with self._lock:
            if self.used >= self.limits.max_children:
                return False
            self.used += 1
            return True

    def run_one(self, task: str, allow: list[str] | None = None) -> ChildReport:
        if not self._reserve():
            return ChildReport(False, "", "budget", error=(
                f"child budget exhausted ({self.limits.max_children})"))
        if self._cancelled():
            return ChildReport(False, "", "cancelled", error="cancelled")
        try:
            runner = self.spawn(str(task), allow)
            result = runner.run()
        except Exception as exc:  # noqa: BLE001 - a child must not kill the parent
            return ChildReport(False, "", "error", error=f"{type(exc).__name__}: {exc}"[:300])
        stats = result.stats
        text = (result.text or "")[: self.limits.max_result_chars]
        return ChildReport(
            ok=result.stop_reason == "done" and not result.error,
            text=text,
            stop_reason=str(result.stop_reason or ""),
            input_tokens=int(stats.input_tokens or 0),
            cached_tokens=int(stats.cached_input_tokens or 0),
            output_tokens=int(stats.output_tokens or 0),
            model_calls=int(stats.model_calls or 0),
            tool_calls=int(stats.tool_calls or 0),
            tool_errors=int(stats.tool_errors or 0),
            files=list(result.artifacts or []),
            error=(result.error or None),
        )

    def run_many(self, specs: list[tuple[str, list[str] | None]]) -> list[ChildReport]:
        if not specs:
            return []
        # Identical task text and tool list in one batch is one child. The
        # duplicate report carries no token counts, so the parent cannot
        # bill the same work twice.
        order: list[int] = []
        unique: list[tuple[str, list[str] | None]] = []
        index: dict[tuple, int] = {}
        for task, allow in specs:
            key = (str(task).strip(), tuple(allow) if allow else None)
            if key not in index:
                index[key] = len(unique)
                unique.append((task, allow))
            order.append(index[key])
        ran = self._run_unique(unique)
        out: list[ChildReport] = []
        emitted: set[int] = set()
        for slot in order:
            if slot in emitted:
                src = ran[slot]
                out.append(ChildReport(
                    ok=src.ok,
                    text=f"[duplicate of child {slot + 1}; not run again]\n{src.text}",
                    stop_reason="duplicate",
                    files=list(src.files),
                    error=src.error))
            else:
                emitted.add(slot)
                out.append(ran[slot])
        return out

    def _run_unique(self, specs: list[tuple[str, list[str] | None]]) -> list[ChildReport]:
        mutating = any(_mutating(allow) for _task, allow in specs)
        width = 1 if mutating else max(1, min(self.limits.max_parallel, len(specs)))
        if width == 1:
            return [self.run_one(task, allow) for task, allow in specs]
        with ThreadPoolExecutor(max_workers=width) as pool:
            return list(pool.map(lambda spec: self.run_one(*spec), specs))


def _mutating(allow: list[str] | None) -> bool:
    if allow is None:
        return True
    return any(name in _WRITE_TOOLS for name in allow)
