from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ordivyr_lite import onboarding
from ordivyr_lite.config import load_config, save_config, resolve_route
from ordivyr_lite.tests.support import ScriptedServer, openai_reply
from ordivyr_lite.tests.test_cli import cli


@pytest.mark.parametrize("selection", ["2", "claude", "Claude", "anthropic"])
def test_claude_alias_connects_without_url_prompt(tmp_path, monkeypatch, selection, capsys):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path))
    monkeypatch.setenv("TEST_KEY", "offline-key")
    answers = [selection, "env", "TEST_KEY", "no", "test-model", "1", "2", ".1", "1", "2", "yes"]
    with patch("builtins.input", side_effect=answers):
        assert onboarding.configure()
    route = resolve_route(load_config())
    assert route.provider == "anthropic" and route.protocol == "anthropic"
    assert route.model == "test-model" and route.api_key == "offline-key"
    assert route.caps["total_usd"] == 2
    assert "offline-key" not in (tmp_path / "config.json").read_text()
    assert "offline-key" not in capsys.readouterr().out
    assert (tmp_path / "config.json").stat().st_mode & 0o777 == 0o600


def test_metadata_lookup_selects_actual_model_without_inference(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path))
    server = ScriptedServer(lambda _: pytest.fail("inference is forbidden during setup")).start()
    try:
        with patch("builtins.input", side_effect=["custom", server.base, "openai", "none", "yes", "1", "yes"]):
            assert onboarding.configure()
        assert load_config()["default_model"]["model"] == "test-model"
        assert server.requests == []
    finally:
        server.stop()


def test_cancel_does_not_change_existing_config_or_store_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path))
    save_config(load_config())
    before = (tmp_path / "config.json").read_bytes()
    with patch("getpass.getpass", return_value="do-not-save"), patch("builtins.input", side_effect=[
        "anthropic", "paste", "no", "test", "1", "2", ".1", "1", "2", "no"
    ]):
        assert not onboarding.configure()
    assert (tmp_path / "config.json").read_bytes() == before
    assert not (tmp_path / "keys").exists()


def test_invalid_final_route_removes_new_secret_and_preserves_config(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path))
    save_config(load_config())
    before = (tmp_path / "config.json").read_bytes()
    with patch("getpass.getpass", return_value="secret"), patch("builtins.input", side_effect=[
        "anthropic", "paste", "no", "test", "1", "2", ".1", "1", "2", "yes"
    ]), patch("ordivyr_lite.config.resolve_route", side_effect=ValueError("invalid")):
        with pytest.raises(ValueError):
            onboarding.configure()
    assert (tmp_path / "config.json").read_bytes() == before
    assert not list((tmp_path / "keys").glob("*.key"))


def test_native_setup_noninteractive_is_actionable(tmp_path):
    result = cli(tmp_path / "home", "setup", "--no-chat", input_text="")
    assert result.returncode == 2
    assert "in a terminal" in result.stderr
    assert not (tmp_path / "home/config.json").exists()


def test_chat_switches_endpoint_without_inference_on_commands(tmp_path):
    a = ScriptedServer(lambda _: openai_reply(text="alpha response")).start()
    b = ScriptedServer(lambda _: openai_reply(text="beta response")).start()
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.json").write_text(json.dumps({
        "providers": {"alpha": {"base_url": a.base}, "beta": {"base_url": b.base}},
    }))
    try:
        result = cli(home, "chat", "--provider", "alpha", "--model", "first",
                     "--workdir", str(tmp_path), "--no-stream", "--validate", "off", "--no-session",
                     input_text="/typo\n/status\n/models\n/model beta:second\nsay hello\n/exit\n")
        assert result.returncode == 0, result.stderr
        assert a.requests == []
        assert len(b.requests) == 1 and b.requests[0]["model"] == "second"
        assert "beta response" in result.stdout
        assert "Unknown command" in result.stderr
        assert "succeeded (done)" in result.stderr
    finally:
        a.stop()
        b.stop()


def test_switch_does_not_carry_key_url_or_paid_authorization(tmp_path, monkeypatch):
    from ordivyr_lite import cli as module
    from ordivyr_lite.errors import ConfigError
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "new-provider-key")
    args = module.build_parser().parse_args([
        "chat", "--provider", "custom", "--model", "first", "--base-url", "http://localhost:9999/v1",
        "--api-key-file", "/old-provider-key", "--header", "X-Secret:old",
        "--allow-paid", "--max-cost", "10", "--protocol", "openai",
    ])
    class Router:
        ledger = object()
    with pytest.raises(ConfigError, match="not authorised"):
        module._switch_model(Router(), args, "anthropic:claude-test", None)
    assert args.api_key_file == "/old-provider-key" and args.allow_paid
    cfg = load_config()
    cfg["providers"]["anthropic"] = {
        "authorized": True, "caps": {"per_request_usd": 1, "daily_usd": 2, "total_usd": 3}}
    cfg["prices"]["claude-test"] = {"input": 1, "output": 2}
    save_config(cfg)
    router, changed = module._switch_model(Router(), args, "anthropic:claude-test", None)
    assert router.primary.api_key == "new-provider-key"
    assert router.primary.base_url == "https://api.anthropic.com"
    assert not changed.allow_paid and changed.max_cost == 0 and changed.header is None
    assert router.ledger is Router.ledger


def test_new_chat_session_preserves_previous_journal(tmp_path):
    server = ScriptedServer(lambda _: openai_reply(text="hello")).start()
    try:
        result = cli(tmp_path / "home", "chat", "--provider", "custom", "--model", "test-model",
                     "--base-url", server.base, "--workdir", str(tmp_path), "--no-stream", "--validate", "off",
                     input_text="first task\n/new\nsecond task\n/exit\n")
        assert result.returncode == 0, result.stderr
        assert len(list((tmp_path / "home/sessions").glob("*.jsonl"))) == 2
        assert "first task" not in json.dumps(server.requests[-1]["messages"])
    finally:
        server.stop()
