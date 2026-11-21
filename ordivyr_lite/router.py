"""Provider routing: one primary route plus an explicit, capped fallback chain.

Rules:
  * a paid fallback is never used after a free/local primary unless the caller
    explicitly enabled it;
  * every route's spend caps are checked before its request;
  * only retryable provider failures (rate limit, network, 5xx, timeout) move
    down the chain; an auth or bad-request failure stops immediately.
"""

from __future__ import annotations

import time
from urllib.parse import urlparse
from dataclasses import dataclass, field
from typing import Callable

from .config import Route, output_limit
from .errors import BudgetError, ProviderError
from .providers import HttpTransport, ModelClient, ModelReply
from .usage import UsageLedger, estimate_cost


@dataclass
class RoutedReply:
    route: Route
    reply: ModelReply
    output_limit: int
    attempts: list[dict] = field(default_factory=list)
    reserve_id: str | None = None


class ModelRouter:
    def __init__(self, routes: list[Route], *, ledger: UsageLedger | None = None,
                 transport: HttpTransport | None = None,
                 allow_paid_fallback: bool = False,
                 local_only: bool = False,
                 on_event: Callable[[dict], None] | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if not routes:
            raise ValueError("at least one route is required")
        self.local_only = bool(local_only)
        if self.local_only:
            routes = [r for r in routes if (urlparse(r.base_url).hostname or "").lower()
                      in ("localhost", "127.0.0.1", "::1")]
            if not routes:
                raise ProviderError("local-only task has no eligible loopback model route",
                                    code="local_only_route_unavailable", retryable=False)
        self.routes = routes
        self.ledger = ledger or UsageLedger()
        self.transport = transport or HttpTransport()
        self.allow_paid_fallback = allow_paid_fallback
        self.on_event = on_event
        self.clock = clock
        self._clients: dict[tuple, ModelClient] = {}

    def restrict_local_only(self) -> None:
        """Persistently narrow this router to loopback routes."""
        self.local_only = True
        self.routes = [r for r in self.routes
                       if (urlparse(r.base_url).hostname or "").lower()
                       in ("localhost", "127.0.0.1", "::1")]
        if not self.routes:
            raise ProviderError("local-only task has no eligible loopback model route",
                                code="local_only_route_unavailable", retryable=False)

    @property
    def primary(self) -> Route:
        return self.routes[0]

    def describe(self) -> list[dict]:
        return [r.redacted() for r in self.routes]

    def output_limit_for(self, route: Route, requested: int | None) -> int:
        return output_limit(route.model, requested, route.context_window)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *,
             max_tokens: int | None = None, temperature: float = 0.0,
             deadline: float | None = None,
             on_delta: Callable[[str, str], None] | None = None,
             run_id: str = "", stream: bool | None = None) -> RoutedReply:
        primary = self.primary
        attempts: list[dict] = []
        last: ProviderError | None = None
        for idx, route in enumerate(self.routes):
            # Recheck the hard restriction at dispatch time, even if a caller
            # mutates the route list after router construction.
            if self.local_only and (urlparse(route.base_url).hostname or "").lower() \
                    not in ("localhost", "127.0.0.1", "::1"):
                raise ProviderError("local-only policy blocked non-loopback provider invocation",
                                    code="local_only_route_blocked", retryable=False,
                                    endpoint=route.base_url)
            if idx > 0:
                if route.paid and not primary.paid and not self.allow_paid_fallback:
                    raise ProviderError(
                        f"refusing implicit paid fallback from {primary.provider} to {route.provider}; "
                        "pass --allow-paid-fallback to opt in", code="paid_not_authorized",
                        retryable=False, endpoint=route.base_url)
                if self.on_event:
                    self.on_event({"type": "route_fallback", "from": primary.label(),
                                   "to": route.label(), "attempt": idx})
            out_limit = self.output_limit_for(route, max_tokens)
            reserve_id: str | None = None
            try:
                est = estimate_cost(route, messages, out_limit, tools)
                reserve_id = self.ledger.reserve(run_id=run_id, route=route,
                                                 estimate_usd=est)
            except BudgetError as exc:
                attempts.append({"route": route.label(), "error": str(exc), "code": "budget"})
                last = ProviderError(str(exc), code="budget", retryable=False, endpoint=route.base_url)
                has_fallback = idx < len(self.routes) - 1
                nxt = self.routes[idx + 1] if has_fallback else None
                if has_fallback and nxt is not None and (not nxt.paid or self.allow_paid_fallback):
                    # A free (or explicitly allowed) fallback may still run.
                    continue
                raise
            # One client per route: it carries the learned capability state
            # (family profile, degraded payload), so a strict endpoint pays for
            # its rejection once per run rather than once per turn.
            key = (route.provider, route.model, route.base_url, route.api_key)
            client = self._clients.get(key)
            if client is None:
                client = ModelClient(route, self.transport, self.clock)
                self._clients[key] = client
            t0 = time.monotonic()
            try:
                reply = client.chat(messages, tools, max_tokens=out_limit,
                                    temperature=temperature, timeout=route.timeout,
                                    deadline=deadline, on_delta=on_delta, stream=stream)
                attempts.append({"route": route.label(), "ok": True,
                                 "latency_ms": round((time.monotonic() - t0) * 1000, 1),
                                 "ttft_ms": round(reply.ttft_s * 1000, 1),
                                 "streamed": bool(reply.streamed),
                                 "family": client.profile.family,
                                 "degraded": list(client.degraded)})
                return RoutedReply(route=route, reply=reply, output_limit=out_limit,
                                   attempts=attempts, reserve_id=reserve_id)
            except ProviderError as e:
                last = e
                self.ledger.release(reserve_id)
                attempts.append({"route": route.label(), "ok": False, "code": e.code,
                                 "error": str(e)[:200],
                                 "latency_ms": round((time.monotonic() - t0) * 1000, 1)})
                if not e.retryable:
                    raise
                if idx == len(self.routes) - 1:
                    raise
                if self.on_event:
                    self.on_event({"type": "route_retry", "route": route.label(),
                                   "code": e.code, "error": str(e)[:200]})
        raise last or ProviderError("all routes failed", code="provider_error")
