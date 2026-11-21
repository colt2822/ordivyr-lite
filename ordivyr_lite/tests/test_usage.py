from __future__ import annotations

import time

import pytest

from ordivyr_lite.config import Route
from ordivyr_lite.errors import BudgetError
from ordivyr_lite.usage import UsageLedger, cost_of, estimate_cost, parse_since


def paid_route(**kw):
    base = dict(provider="openai", protocol="openai", base_url="https://x.test/v1",
                model="m", paid=True, authorized=True, price=(1.0, 2.0, 0.1, 0.5),
                caps={"per_request_usd": 1.0, "daily_usd": 2.0, "total_usd": 3.0})
    base.update(kw)
    return Route(**base)


def test_cost_of_counts_fresh_cache_and_output():
    r = paid_route()
    cost = cost_of(r, {"input": 1_000_000, "cache_read": 1_000_000,
                       "cache_write": 1_000_000, "output": 1_000_000})
    assert cost == pytest.approx(1.0 + 0.1 + 0.5 + 2.0)


def test_local_cost_is_zero():
    r = paid_route(paid=False, price=None, authorized=True, caps={})
    assert cost_of(r, {"input": 10, "output": 10}) == 0.0


def test_unpriced_paid_model_fails_closed():
    r = paid_route(price=None)
    with pytest.raises(BudgetError):
        cost_of(r, {"input": 1})
    with pytest.raises(BudgetError):
        estimate_cost(r, [{"role": "user", "content": "x"}], 100)


def test_ledger_records_and_reports(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    r = paid_route()
    ledger.record(run_id="r1", route=r, usage={"input": 1000, "cache_read": 500,
                                               "cache_write": 100, "output": 200,
                                               "context": 1600},
                  cost_usd=0.01, latency_ms=12.5)
    t = ledger.totals()
    assert t.calls == 1
    assert t.input_tokens == 1000
    assert t.cache_read_tokens == 500
    assert t.cache_hit_rate == pytest.approx(500 / 1600, abs=1e-4)
    report = ledger.report()
    assert report["totals"]["cost_usd"] == 0.01
    assert report["by_model"]["m"]["calls"] == 1


def test_caps_fail_closed(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    r = paid_route()
    with pytest.raises(BudgetError, match="per-request"):
        ledger.check_caps(r, 2.0)
    ledger.record(run_id="r", route=r, usage={"input": 1}, cost_usd=1.5, latency_ms=1)
    with pytest.raises(BudgetError, match="daily"):
        ledger.check_caps(r, 0.6)
    assert ledger.spent(r.provider) == 1.5


def test_local_caps_are_never_charged(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    r = paid_route(paid=False, price=None, caps={})
    ledger.check_caps(r, 100.0)


def test_reasoning_and_cache_totals(tmp_path):
    ledger = UsageLedger(tmp_path / "usage.jsonl")
    t = ledger.totals()
    assert t.input_tokens == 0


def test_parse_since():
    assert parse_since(None) is None
    before = time.time()
    parsed = parse_since("24h")
    assert parsed is not None and before - parsed == pytest.approx(86400, abs=2)
    with pytest.raises(ValueError):
        parse_since("nonsense")
