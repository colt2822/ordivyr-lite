from __future__ import annotations

import json
import os

import pytest

from ordivyr_lite.config import (
    auth_header_name, normalize_headers,
    DEFAULT_OUTPUT_TOKENS, MIN_OUTPUT_TOKENS, load_config, output_limit, resolve_route,
)
from ordivyr_lite.errors import ConfigError


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_local_route_is_free_and_keyless(home):
    cfg = load_config()
    route = resolve_route(cfg, provider="llamacpp", model="local-7b")
    assert route.paid is False
    assert route.authorized is True
    assert route.api_key == ""
    assert route.price is None
    assert route.base_url.startswith("http://127.0.0.1")


def test_paid_route_needs_a_key(home):
    cfg = load_config()
    with pytest.raises(ConfigError, match="no credential"):
        resolve_route(cfg, provider="openai", model="gpt-4o-mini")


def test_header_credential_authorises_a_paid_route(home):
    cfg = load_config()
    cfg["providers"]["account"] = {"base_url": "https://api.example.test/v1", "paid": True,
                                   "authorized": True, "auth_header": "x-account-token",
                                   "prices": {"m": {"input": 0.3, "output": 1.2}}}
    route = resolve_route(cfg, provider="account", model="m", allow_paid=True, max_cost_usd=1.0,
                          headers={"X-Account-Token": "tok-not-a-key"})
    assert route.api_key == ""
    assert route.has_credentials is True
    assert route.header_auth == "X-Account-Token"
    assert route.key_source == "header:X-Account-Token"


def test_header_value_reads_the_environment(home, monkeypatch):
    monkeypatch.setenv("ORDIVYR_TEST_TOKEN", "from-env")
    assert normalize_headers({"x-token": "@ORDIVYR_TEST_TOKEN"}) == {"x-token": "from-env"}
    monkeypatch.delenv("ORDIVYR_TEST_TOKEN")
    with pytest.raises(ConfigError, match="empty or unset"):
        normalize_headers({"x-token": "@ORDIVYR_TEST_TOKEN"})


def test_header_rejects_injection_and_bad_names(home):
    with pytest.raises(ConfigError, match="single line"):
        normalize_headers({"x-a": "v\r\nX-Evil: 1"})
    with pytest.raises(ConfigError, match="invalid header name"):
        normalize_headers({"bad name": "v"})


def test_declared_auth_header_must_be_present(home):
    with pytest.raises(ConfigError, match="not present in headers"):
        auth_header_name({"x-other": "v"}, "x-account-token")


def test_paid_route_fails_closed_without_authorization(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    cfg = load_config()
    with pytest.raises(ConfigError, match="not authorised"):
        resolve_route(cfg, provider="openai", model="gpt-4o-mini")


def test_paid_route_needs_a_price(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    cfg = load_config()
    with pytest.raises(ConfigError, match="no price"):
        resolve_route(cfg, provider="openai", model="gpt-4o-mini",
                      allow_paid=True, max_cost_usd=1.0)


def test_paid_route_authorized_by_flag_and_price(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    cfg = load_config()
    cfg["prices"]["gpt-4o-mini"] = {"input": 0.15, "output": 0.60, "cache_read": 0.015}
    route = resolve_route(cfg, provider="openai", model="gpt-4o-mini",
                          allow_paid=True, max_cost_usd=2.5)
    assert route.paid is True
    assert route.authorized is True
    assert route.caps == {"per_request_usd": 2.5, "daily_usd": 2.5, "total_usd": 2.5}
    assert route.price[0] == 0.15 and route.price[1] == 0.60
    assert "sk-test" not in repr(route)


def test_key_file_is_supported(home, tmp_path):
    key = tmp_path / "key"
    key.write_text("sk-from-file\n")
    cfg = load_config()
    cfg["prices"]["gpt-4o-mini"] = [0.1, 0.2]
    route = resolve_route(cfg, provider="openai", model="gpt-4o-mini",
                          api_key_file=str(key), allow_paid=True, max_cost_usd=1)
    assert route.api_key == "sk-from-file"


def test_output_limit_model_aware_and_context_bounded():
    assert output_limit("gpt-4o") == 16384
    assert output_limit("gpt-3.5-turbo") == 4096
    assert output_limit("claude-3-haiku") == 4096
    assert output_limit("unknown-model") == DEFAULT_OUTPUT_TOKENS
    assert output_limit("unknown-model", 999_999) <= 131_072
    assert output_limit("unknown-model", context_window=16_000) == 8000
    assert output_limit("unknown-model", 1) == MIN_OUTPUT_TOKENS


def test_config_defaults_are_stable(home):
    cfg = load_config()
    assert cfg["limits"]["max_turns"] == 64
    assert cfg["permissions"]["profile"] == "workspace"
    assert cfg["validate"] == "auto"


def test_route_repr_redacts_key():
    from ordivyr_lite.config import Route
    r = Route(provider="x", protocol="openai", base_url="https://x.test/v1",
              model="m", api_key="sk-very-secret")
    assert "very-secret" not in repr(r)
    assert "very-secret" not in json.dumps(r.redacted())


def test_exit_codes_are_stable():
    from ordivyr_lite.errors import exit_code_for
    assert exit_code_for("done") == 0
    assert exit_code_for("paid_not_authorized") == 3
    assert exit_code_for("unpriced_model") == 5
    assert exit_code_for("auth") == 4
    assert exit_code_for("malformed") == 6
    assert exit_code_for("cancelled") == 7
