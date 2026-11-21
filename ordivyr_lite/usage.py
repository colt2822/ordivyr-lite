"""Transparent usage accounting, pricing and fail-closed spend caps.

Every model call appends one JSON line to ``$ORDIVYR_LITE_HOME/usage.jsonl``.
The report is derived from that file, so tokens, cache hits, latency and cost
are always the numbers actually observed - never a self-reported estimate.
"""

from __future__ import annotations

try:
    import fcntl
except ImportError:  # Windows/POSIX-only lock; the append is still atomic enough for one process
    fcntl = None  # type: ignore[assignment]
import json
import os
import secrets
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Route, config_home
from .errors import BudgetError
from .util import estimate_tokens, human_usd, utc_now

RESERVE_TTL_S = 600.0


@dataclass
class Totals:
    calls: int = 0
    input_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0

    def add(self, rec: dict) -> None:
        self.calls += 1
        self.input_tokens += int(rec.get("input_tokens") or 0)
        self.cache_read_tokens += int(rec.get("cache_read_tokens") or 0)
        self.cache_write_tokens += int(rec.get("cache_write_tokens") or 0)
        self.output_tokens += int(rec.get("output_tokens") or 0)
        self.reasoning_tokens += int(rec.get("reasoning_tokens") or 0)
        self.cost_usd += float(rec.get("cost_usd") or 0.0)
        self.latency_ms += float(rec.get("latency_ms") or 0.0)

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens

    @property
    def cache_hit_rate(self) -> float:
        ctx = self.context_tokens
        return round(self.cache_read_tokens / ctx, 4) if ctx else 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "context_tokens": self.context_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost_usd": round(self.cost_usd, 8),
            "cache_hit_rate": self.cache_hit_rate,
            "latency_ms": round(self.latency_ms, 1),
        }


def cost_of(route: Route, usage: dict) -> float:
    """USD for one call.  Local endpoints cost 0; paid models require a price."""
    if not route.paid:
        return 0.0
    if route.price is None:
        raise BudgetError(f"model {route.model!r} has no configured price")
    in_p, out_p = route.price[0], route.price[1]
    cr_p = route.price[2] if len(route.price) > 2 else in_p
    cw_p = route.price[3] if len(route.price) > 3 else in_p
    fresh = int(usage.get("input") or 0)
    cr = int(usage.get("cache_read") or 0)
    cw = int(usage.get("cache_write") or 0)
    out = int(usage.get("output") or 0)
    return round((fresh * in_p + cr * cr_p + cw * cw_p + out * out_p) / 1_000_000, 10)


def estimate_cost(route: Route, messages: list[dict], max_tokens: int, tools: list[dict] | None = None) -> float:
    if not route.paid:
        return 0.0
    if route.price is None:
        raise BudgetError(f"model {route.model!r} has no configured price; spending cannot be capped")
    try:
        chars = len(json.dumps({"messages": messages, "tools": tools or []}, separators=(",", ":"), default=str))
    except (TypeError, ValueError):
        chars = sum(len(str(m)) for m in messages)
    from . import tokens as TK
    family = TK.family_for(route.model, route.provider, route.base_url)
    counted = TK.count_request(messages, tools, family)
    in_p, out_p = route.price[0], route.price[1]
    # Two independent estimates: a cap check takes the larger one, so a sparse
    # tokenizer model can never make a paid call look cheaper than it is.
    prompt_tokens = max(int(chars / 3.5), int(counted))
    return round((prompt_tokens * max((in_p, *route.price[2:])) + max_tokens * out_p) / 1_000_000, 10)


class UsageLedger:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else config_home() / "usage.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _iter_records(self) -> Iterable[dict]:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(rec, dict):
                        yield rec
        except OSError:
            return

    def records(self, since: float | None = None) -> list[dict]:
        out = [r for r in self._iter_records() if since is None or float(r.get("ts") or 0) >= since]
        out.sort(key=lambda r: float(r.get("ts") or 0))
        return out

    def totals(self, since: float | None = None, *, provider: str | None = None,
               model: str | None = None, ok_only: bool = True) -> Totals:
        t = Totals()
        for r in self.records(since):
            if ok_only and r.get("status") not in ("ok", None):
                continue
            if provider and r.get("provider") != provider:
                continue
            if model and r.get("model") != model:
                continue
            t.add(r)
        return t

    @staticmethod
    def spent_from(records: list[dict], provider: str | None = None,
                   since: float | None = None) -> float:
        """Spend including live reservations, so a reserve can be judged atomically."""
        settled = {r.get("reserve_id") for r in records
                   if r.get("status") == "settled" and r.get("reserve_id")}
        now = time.time()
        total = 0.0
        for r in records:
            status = r.get("status")
            ts = float(r.get("ts") or 0)
            if status == "reserved":
                if r.get("reserve_id") in settled or now - ts > RESERVE_TTL_S:
                    continue
            elif status not in ("ok", None):
                continue
            if since is not None and ts < since:
                continue
            if provider and r.get("provider") != provider:
                continue
            total += float(r.get("usd") or r.get("cost_usd") or 0.0)
        return round(total, 10)

    def spent(self, provider: str | None = None, since: float | None = None) -> float:
        return self.spent_from(self.records(), provider, since)

    @contextmanager
    def _locked(self):
        """Cross-process advisory lock around one check-then-write transaction."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            if fcntl is not None:
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield fd
        finally:
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _append(self, fd: int, payload: dict) -> None:
        line = json.dumps(payload, separators=(",", ":")) + "\n"
        os.write(fd, line.encode("utf-8"))

    def reserve(self, *, run_id: str, route: Route, estimate_usd: float) -> str | None:
        """Reserve the worst-case cost of ONE paid call, atomically.

        Returns a reservation id to settle (None on a free route) and raises
        BudgetError when the caps would be exceeded.  Two concurrent processes
        can no longer both pass the cap check before either writes.
        """
        if not route.paid:
            return None
        rid = "res_" + secrets.token_hex(8)
        with self._locked() as fd:
            self._check_caps_from(self.records(), route, estimate_usd)
            self._append(fd, {
                "ts": time.time(), "time": utc_now(), "run_id": run_id,
                "provider": route.provider, "model": route.model, "protocol": route.protocol,
                "paid": True, "input_tokens": 0, "cache_read_tokens": 0,
                "cache_write_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0,
                "context_tokens": 0, "cost_usd": round(float(estimate_usd), 10),
                "usd": round(float(estimate_usd), 10), "latency_ms": 0.0,
                "status": "reserved", "attempt": 1, "reserve_id": rid,
            })
        return rid

    def release(self, reserve_id: str | None) -> None:
        """Drop a reservation without spending it (the call failed or was not made)."""
        if not reserve_id:
            return
        with self._locked() as fd:
            self._append(fd, {"ts": time.time(), "status": "settled",
                              "reserve_id": reserve_id})

    def check_caps(self, route: Route, estimate_usd: float) -> None:
        """Fail closed BEFORE a paid request.  Local calls are always allowed."""
        if not route.paid:
            return
        self._check_caps_from(self.records(), route, estimate_usd)

    def _check_caps_from(self, records: list[dict], route: Route,
                         estimate_usd: float) -> None:
        if not route.authorized:
            raise BudgetError(f"{route.provider} is paid and not authorised")
        caps = route.caps or {}
        per_request = float(caps.get("per_request_usd") or 0)
        daily = float(caps.get("daily_usd") or 0)
        total = float(caps.get("total_usd") or 0)
        if not (per_request > 0 and daily > 0 and total > 0):
            raise BudgetError(f"{route.provider} is missing positive spend caps")
        if estimate_usd > per_request:
            raise BudgetError(
                f"worst-case request cost ${estimate_usd:.6f} exceeds per-request cap ${per_request:.6f}")
        spent_today = self.spent_from(records, route.provider, since=time.time() - 86400)
        if spent_today + estimate_usd > daily:
            raise BudgetError(f"daily cap ${daily:.6f} would be exceeded "
                              f"(spent ${spent_today:.6f} + estimate ${estimate_usd:.6f})")
        spent_total = self.spent_from(records, route.provider)
        if spent_total + estimate_usd > total:
            raise BudgetError(f"total cap ${total:.6f} would be exceeded "
                              f"(spent ${spent_total:.6f} + estimate ${estimate_usd:.6f})")

    def record(self, *, run_id: str, route: Route, usage: dict, cost_usd: float,
               latency_ms: float, status: str = "ok", error: str | None = None,
               attempt: int = 1, reserve_id: str | None = None) -> dict:
        rec = {
            "ts": time.time(),
            "time": utc_now(),
            "run_id": run_id,
            "provider": route.provider,
            "model": route.model,
            "protocol": route.protocol,
            "paid": route.paid,
            "input_tokens": int(usage.get("input") or 0),
            "cache_read_tokens": int(usage.get("cache_read") or 0),
            "cache_write_tokens": int(usage.get("cache_write") or 0),
            "output_tokens": int(usage.get("output") or 0),
            "reasoning_tokens": int(usage.get("reasoning") or 0),
            "context_tokens": int(usage.get("context") or 0),
            "cost_usd": round(float(cost_usd), 10),
            "latency_ms": round(float(latency_ms), 1),
            "status": status,
            "attempt": attempt,
        }
        if error:
            rec["error"] = error[:300]
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_EX)
                self._append(fd, rec)
                if reserve_id:
                    self._append(fd, {"ts": time.time(), "status": "settled",
                                      "reserve_id": reserve_id})
                if fcntl is not None:
                    fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
        return rec

    def report(self, since: float | None = None) -> dict:
        recs = [r for r in self.records(since) if r.get("status") in ("ok", None)]
        totals = Totals()
        by_model: dict[str, Totals] = {}
        by_provider: dict[str, Totals] = {}
        days: dict[str, Totals] = {}
        for r in recs:
            totals.add(r)
            by_model.setdefault(str(r.get("model")), Totals()).add(r)
            by_provider.setdefault(str(r.get("provider")), Totals()).add(r)
            day = time.strftime("%Y-%m-%d", time.localtime(float(r.get("ts") or 0)))
            days.setdefault(day, Totals()).add(r)
        return {
            "since": since,
            "path": str(self.path),
            "totals": totals.as_dict(),
            "by_model": {k: v.as_dict() for k, v in sorted(by_model.items())},
            "by_provider": {k: v.as_dict() for k, v in sorted(by_provider.items())},
            "by_day": {k: v.as_dict() for k, v in sorted(days.items())},
        }


def format_report(report: dict) -> str:
    t = report["totals"]
    lines = [
        f"usage ledger: {report.get('path')}",
        f"calls={t['calls']}  input={t['input_tokens']:,}  cache_read={t['cache_read_tokens']:,}  "
        f"cache_write={t['cache_write_tokens']:,}  output={t['output_tokens']:,}  "
        f"cache_hit={t['cache_hit_rate']:.1%}  cost={human_usd(t['cost_usd'])}",
    ]
    if report.get("by_model"):
        lines.append("by model:")
        for name, v in report["by_model"].items():
            lines.append(f"  {name:40} calls={v['calls']:<5} tokens={v['context_tokens'] + v['output_tokens']:<10,} "
                         f"cost={human_usd(v['cost_usd'])}")
    if report.get("by_provider"):
        lines.append("by provider:")
        for name, v in report["by_provider"].items():
            lines.append(f"  {name:20} calls={v['calls']:<5} cost={human_usd(v['cost_usd'])}")
    return "\n".join(lines)


def parse_since(value: str | None) -> float | None:
    if not value:
        return None
    value = value.strip().lower()
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if value[-1:] in units and value[:-1].replace(".", "", 1).isdigit():
        return time.time() - float(value[:-1]) * units[value[-1]]
    if value in ("today",):
        lt = time.localtime()
        return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, 0, 0, 0, lt.tm_wday, lt.tm_yday, -1))
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"bad --since value {value!r}; use e.g. 24h, 7d or an epoch") from exc
