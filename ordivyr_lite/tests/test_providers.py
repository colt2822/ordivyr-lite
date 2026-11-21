from __future__ import annotations

import json

import pytest

from ordivyr_lite.config import Route
from ordivyr_lite.errors import ProviderError
from ordivyr_lite.providers import HttpTransport, ModelClient, _convert_anthropic_messages
from ordivyr_lite.tests.support import FakeTransport, openai_reply


def route(protocol="openai", base_url="https://api.example.test/v1", max_retries=0, **kw):
    return Route(provider="p", protocol=protocol, base_url=base_url,
                 model="m", api_key="k", paid=True, authorized=True,
                 price=(1.0, 2.0, 0.1, 1.0), caps={"per_request_usd": 10, "daily_usd": 10,
                                                   "total_usd": 10}, max_retries=max_retries, **kw)


def test_openai_normalises_text_calls_and_cache_usage():
    t = FakeTransport([openai_reply(text="hello", calls=[("read_file", {"path": "x"})],
                                    prompt_tokens=1000, completion_tokens=20,
                                    cached_tokens=600)])
    client = ModelClient(route(), transport=HttpTransport(http=t))
    reply = client.chat([{"role": "user", "content": "hi"}])
    assert reply.text == "hello"
    assert reply.calls[0].name == "read_file"
    assert reply.calls[0].args == {"path": "x"}
    assert reply.usage["input"] == 400
    assert reply.usage["cache_read"] == 600
    assert reply.usage["context"] == 1000
    assert reply.usage["output"] == 20
    assert reply.usage["telemetry"] is True


def test_malformed_arguments_are_recorded_never_defaulted():
    payload = {"choices": [{"message": {"role": "assistant", "content": "",
                                        "tool_calls": [{"id": "c", "type": "function",
                                                        "function": {"name": "write_file",
                                                                     "arguments": '{"path": "x", "content": "par'}}]},
                           "finish_reason": "tool_calls"}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    t = FakeTransport([payload])
    reply = ModelClient(route(), transport=HttpTransport(http=t)).chat([{"role": "user", "content": "x"}])
    assert reply.calls[0].error is not None
    assert reply.calls[0].args == {}


def test_truncated_turn_is_flagged():
    t = FakeTransport([openai_reply(text="cut", calls=[("read_file", {"path": "x"})],
                                    finish="length")])
    reply = ModelClient(route(), transport=HttpTransport(http=t)).chat([{"role": "user", "content": "x"}])
    assert reply.truncated is True


def test_anthropic_conversion_groups_tool_results_and_cache_breakpoints():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "read_file",
                                                          "arguments": json.dumps({"path": "a"})}},
            {"id": "t2", "type": "function", "function": {"name": "grep",
                                                          "arguments": json.dumps({"pattern": "x"})}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "a"},
        {"role": "tool", "tool_call_id": "t2", "content": "b"},
    ]
    out, system = _convert_anthropic_messages(msgs, cache=True)
    assert system == "sys"
    assert out[1]["role"] == "assistant"
    assert [b["type"] for b in out[1]["content"]] == ["tool_use", "tool_use"]
    assert out[2]["role"] == "user"
    assert len(out[2]["content"]) == 2
    assert out[2]["content"][-1]["cache_control"] == {"type": "ephemeral"}


def test_anthropic_client_caches_system_tools_and_messages():
    captured = {}

    def http(method, url, headers, body, timeout):
        captured["body"] = json.loads(body)
        captured["headers"] = headers
        return 200, json.dumps({"content": [{"type": "text", "text": "ok"}],
                                "stop_reason": "end_turn", "model": "claude",
                                "usage": {"input_tokens": 3, "cache_read_input_tokens": 7,
                                          "cache_creation_input_tokens": 2, "output_tokens": 1}}).encode()

    r = route("anthropic", base_url="https://api.anthropic.com", max_retries=0)
    client = ModelClient(r, transport=HttpTransport(http=http))
    reply = client.chat([{"role": "system", "content": "sys"},
                         {"role": "user", "content": "hi"}],
                        tools=[{"type": "function", "function": {"name": "read_file",
                                                                 "description": "d",
                                                                 "parameters": {"type": "object"}}}],
                        max_tokens=50)
    assert reply.usage["cache_read"] == 7
    assert reply.usage["cache_write"] == 2
    body = captured["body"]
    assert body["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert captured["headers"]["x-api-key"] == "k"
    assert "Authorization" not in captured["headers"]


def test_retryable_server_error_retries_once():
    t = FakeTransport([(500, {"error": {"message": "temporary"}}), openai_reply(text="ok")])
    r = route(max_retries=1)
    reply = ModelClient(r, transport=HttpTransport(http=t)).chat([{"role": "user", "content": "x"}])
    assert reply.text == "ok"
    assert len(t.requests) == 2


def test_token_limit_error_is_classified():
    t = FakeTransport([(400, {"error": {"message": "max_tokens is too large"}})])
    with pytest.raises(ProviderError) as exc:
        ModelClient(route(max_retries=0), transport=HttpTransport(http=t)).chat(
            [{"role": "user", "content": "x"}], max_tokens=100000)
    assert exc.value.code == "token_limit"


def test_auth_error_is_not_retryable():
    t = FakeTransport([(401, {"error": {"message": "bad key"}})])
    with pytest.raises(ProviderError) as exc:
        ModelClient(route(max_retries=2), transport=HttpTransport(http=t)).chat(
            [{"role": "user", "content": "x"}])
    assert exc.value.code == "auth"
    assert exc.value.retryable is False
