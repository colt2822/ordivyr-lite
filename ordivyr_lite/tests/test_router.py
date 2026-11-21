from __future__ import annotations

import pytest

from ordivyr_lite.config import Route
from ordivyr_lite.errors import BudgetError, ProviderError
from ordivyr_lite.loop import Runner
from ordivyr_lite.providers import HttpTransport
from ordivyr_lite.router import ModelRouter
from ordivyr_lite.tests.support import FakeTransport, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.usage import UsageLedger


def local(name="a", model="m"):
    return Route(provider=name, protocol="openai", base_url="http://127.0.0.1:9/v1",
                 model=model, paid=False, timeout=5, max_retries=0)


def paid(name="p", model="m", caps=None):
    return Route(provider=name, protocol="openai", base_url="https://x.test/v1",
                 model=model, api_key="k", paid=True, authorized=True,
                 price=(1.0, 2.0), caps=caps or {"per_request_usd": 10, "daily_usd": 10,
                                                  "total_usd": 10}, timeout=5, max_retries=0)


def test_retryable_failure_falls_through_to_next_route(tmp_path):
    t = FakeTransport([(500, {"error": {"message": "down"}}), openai_reply(text="ok")])
    router = ModelRouter([local("a"), local("b")],
                         transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"))
    routed = router.chat([{"role": "user", "content": "x"}])
    assert routed.reply.text == "ok"
    assert routed.route.provider == "b"
    assert len(routed.attempts) == 2


def test_paid_fallback_is_refused_by_default(tmp_path):
    t = FakeTransport([(500, {"error": {"message": "down"}})])
    router = ModelRouter([local("a"), paid("p")],
                         transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"),
                         allow_paid_fallback=False)
    with pytest.raises(ProviderError) as exc:
        router.chat([{"role": "user", "content": "x"}])
    assert exc.value.code == "paid_not_authorized"


def test_paid_fallback_allowed_explicitly(tmp_path):
    t = FakeTransport([(500, {"error": {"message": "down"}}), openai_reply(text="ok")])
    router = ModelRouter([local("a"), paid("p")],
                         transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"),
                         allow_paid_fallback=True)
    routed = router.chat([{"role": "user", "content": "x"}])
    assert routed.route.provider == "p"


def test_paid_cap_failure_can_use_free_fallback(tmp_path):
    t = FakeTransport([openai_reply(text="ok")])
    tiny = paid("p", caps={"per_request_usd": 1e-12, "daily_usd": 1e-12,
                           "total_usd": 1e-12})
    router = ModelRouter([tiny, local("a")],
                         transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"))
    routed = router.chat([{"role": "user", "content": "x"}])
    assert routed.route.provider == "a"


def test_no_fallback_raises_budget(tmp_path):
    t = FakeTransport([])
    tiny = paid("p", caps={"per_request_usd": 1e-12, "daily_usd": 1e-12,
                           "total_usd": 1e-12})
    router = ModelRouter([tiny], transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"))
    with pytest.raises(BudgetError):
        router.chat([{"role": "user", "content": "x"}])


def test_local_only_filters_cloud_routes_even_when_paid_fallback_is_enabled(tmp_path):
    t = FakeTransport([(500, {"error": {"message": "local unavailable"}}),
                       openai_reply(text="must not reach cloud")])
    router = ModelRouter([paid("cloud"), local("local")],
                         transport=HttpTransport(http=t),
                         ledger=UsageLedger(tmp_path / "u.jsonl"),
                         allow_paid_fallback=True, local_only=True)
    assert router.primary.provider == "local"
    with pytest.raises(ProviderError) as exc:
        router.chat([{"role": "user", "content": "x"}])
    assert exc.value.code == "server"
    assert len(t.requests) == 1


def test_local_only_fails_before_any_provider_call_without_local_route(tmp_path):
    t = FakeTransport([openai_reply(text="should not call")])
    with pytest.raises(ProviderError, match="local-only"):
        ModelRouter([paid("cloud")], transport=HttpTransport(http=t),
                    ledger=UsageLedger(tmp_path / "u.jsonl"), local_only=True)
    assert t.requests == []


def test_resume_restores_saved_local_only_restriction(tmp_path):
    router = ModelRouter([paid("cloud"), local("local")],
                         transport=HttpTransport(http=FakeTransport([])),
                         ledger=UsageLedger(tmp_path / "u.jsonl"))
    Runner(router, ToolBox(tmp_path, profile="workspace"), "resume",
        resume_state={"messages": [{"role": "user", "content": "resume"}],
                      "local_only": True})
    assert router.local_only is True
    assert router.primary.provider == "local"
