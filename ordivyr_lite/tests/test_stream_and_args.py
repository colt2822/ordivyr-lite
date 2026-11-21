"""Provider-side failure handling: repair, the error window, overflow, degrade.

Every test here pins one behaviour the loop depends on: arguments that are
almost JSON become a usable call, arguments that are not stay refused, the
model is shown the character that actually failed, a prompt that is too long is
classified as such, and a strict endpoint that refuses the usage stream field is
retried without it instead of ending the run.
"""

from __future__ import annotations

import json
import re

import pytest

from ordivyr_lite.config import Route
from ordivyr_lite.errors import ProviderError
from ordivyr_lite.providers import (
    HttpTransport,
    ModelClient,
    _assemble_anthropic_stream,
    _assemble_openai_stream,
    _error_window,
    _parse_args,
    _parse_http_error,
)
from ordivyr_lite.tests.support import FakeTransport, openai_reply


def route(protocol="openai", base_url="https://api.example.test/v1", max_retries=0, **kw):
    return Route(provider="p", protocol=protocol, base_url=base_url, model="m", api_key="k",
                 paid=True, authorized=True, price=(1.0, 2.0, 0.1, 1.0),
                 caps={"per_request_usd": 10, "daily_usd": 10, "total_usd": 10},
                 max_retries=max_retries, **kw)


def raw_call(arguments: str) -> dict:
    """One OpenAI reply whose single tool call carries exactly the text given."""
    return {"choices": [{"index": 0, "message": {"role": "assistant", "content": "",
                                                 "tool_calls": [{"id": "c", "type": "function",
                                                                 "function": {"name": "read_file",
                                                                              "arguments": arguments}}]},
                         "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def one_call(arguments: str):
    transport = FakeTransport([raw_call(arguments)])
    reply = ModelClient(route(), transport=HttpTransport(http=transport)).chat(
        [{"role": "user", "content": "x"}])
    assert len(reply.calls) == 1
    return reply.calls[0]


# --------------------------------------------------------------- argument repair

def test_a_trailing_comma_is_repaired_and_the_call_is_marked():
    call = one_call('{"path": "a.txt",}')
    assert (call.args, call.error, call.repaired) == ({"path": "a.txt"}, None, True)
    # The history replays the executed arguments, so they are valid JSON.
    assert json.loads(call.raw) == {"path": "a.txt"}


def test_an_object_that_ends_early_is_closed():
    call = one_call('{"path": "a.txt"')
    assert (call.args, call.error, call.repaired) == ({"path": "a.txt"}, None, True)
    assert json.loads(call.raw) == {"path": "a.txt"}


def test_an_early_end_with_a_nested_list_is_closed():
    call = one_call('{"path": ["a.txt", "b.txt"')
    assert (call.args, call.error, call.repaired) == ({"path": ["a.txt", "b.txt"]}, None, True)


def test_a_single_quoted_object_is_normalised():
    call = one_call("{'path': 'a.txt'}")
    assert (call.args, call.error, call.repaired) == ({"path": "a.txt"}, None, True)


def test_arguments_that_already_parsed_are_never_touched():
    raw = '{"path": "a.txt",   "n": 2}'
    call = one_call(raw)
    assert (call.args, call.error, call.repaired) == ({"path": "a.txt", "n": 2}, None, False)
    assert call.raw == raw, "a parsed argument string is passed through byte for byte"


def test_repair_never_rewrites_a_value_that_contains_json_punctuation():
    # The comma-before-closer pass must not edit the inside of a string value.
    call = one_call('{"command": "echo a,}", "x": 1')
    assert (call.args, call.error, call.repaired) == ({"command": "echo a,}", "x": 1}, None, True)


@pytest.mark.parametrize("raw", [
    '{path: "a.txt"',                       # unquoted key, also ends early
    '{"path": "x.txt", "content": "par',    # unterminated string: closing it would invent text
    '{"path": "x", ',                       # a comma with no key after it
    '{"a": 1}}',                            # a closer with nothing to close
    '[1, 2]',                               # not an object
])
def test_arguments_repair_cannot_fix_stay_malformed(raw):
    args, text, err, repaired = _parse_args(raw)
    assert args == {} and repaired is False
    assert err is not None
    assert text == raw, "a refused argument string is reported as it arrived"


def test_a_refused_call_is_never_marked_repaired():
    call = one_call('{path: "a.txt"')
    assert (call.args, call.repaired) == ({}, False)
    assert call.error is not None


def test_streamed_arguments_are_repaired_too():
    events = [
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c", "function": {
            "name": "read_file", "arguments": '{"path": "a.txt"'}}]}}]},
        {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]},
    ]
    reply = _assemble_openai_stream(iter(events), route(), None, 0.0, lambda: 1.0)
    assert (reply.calls[0].args, reply.calls[0].error, reply.calls[0].repaired) == (
        {"path": "a.txt"}, None, True)


def test_anthropic_streamed_arguments_are_repaired_too():
    events = [
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "tool_use", "id": "t1", "name": "read_file"}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "input_json_delta", "partial_json": '{"path": "a.txt"'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
    ]
    reply = _assemble_anthropic_stream(iter(events), route(protocol="anthropic"), None, 0.0,
                                      lambda: 1.0)
    assert (reply.calls[0].args, reply.calls[0].error, reply.calls[0].repaired) == (
        {"path": "a.txt"}, None, True)


# ------------------------------------------------------------- error text window

def test_a_malformed_error_shows_the_failing_window_not_the_head():
    raw = '{"path": "HEAD' + "x" * 2100 + 'TAIL" "k": 1}'
    call = one_call(raw)
    assert call.error is not None
    assert "TAIL" in call.error, "the window must reach the failure"
    assert "HEAD" not in call.error, "the first 160 characters are not the useful ones"
    assert len(call.error) <= 400
    offset, total = re.search(r"char (\d+) of (\d+)", call.error).groups()
    assert 2000 < int(offset) < len(raw), "the offset names the failing column"
    assert int(total) == len(raw)


def test_the_window_falls_back_to_the_head_without_a_position():
    window = _error_window("y" * 300, ValueError("no position"))
    assert window.startswith("char 0 of 300: ")
    assert window.count("y") == 120


def test_a_short_argument_is_reported_in_full():
    args, text, err, repaired = _parse_args('{"a": }')
    assert args == {} and repaired is False
    assert err is not None and '"a": }' in err


# ------------------------------------------------- context-overflow recognition

@pytest.mark.parametrize("message", [
    "prompt is too long: 200000 tokens > 128000 maximum",
    "too many tokens in this request",
    "context length exceeded",
    "This model's maximum context length is 8192 tokens",
    "maximum context reached",
])
def test_prompt_too_long_is_a_token_limit(message):
    err = _parse_http_error(400, json.dumps({"error": {"message": message}}).encode(), route())
    assert err.code == "token_limit", message
    assert str(err).startswith("provider rejected the request")


def test_a_400_without_a_length_phrase_is_still_a_bad_request():
    err = _parse_http_error(400, b'{"error": {"message": "invalid tool schema"}}', route())
    assert err.code == "bad_request"
    assert err.retryable is False


def test_max_tokens_rejection_is_still_a_token_limit():
    err = _parse_http_error(400, b'{"error": {"message": "max_tokens is too large"}}', route())
    assert err.code == "token_limit"


# ------------------------------------------------- stream_options degrade path

def test_a_400_naming_stream_options_degrades_instead_of_ending_the_run():
    t = FakeTransport([(400, {"error": {"message": "Unsupported parameter: 'stream_options'"}}),
                       openai_reply(text="ok")])
    client = ModelClient(route(), transport=HttpTransport(http=t))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.text == "ok"
    assert client.degraded == ["extras:400"]
    assert "stream_options" in t.requests[0]
    assert "stream_options" not in t.requests[1]


def test_a_400_naming_the_stream_field_with_other_wording_also_degrades():
    t = FakeTransport([(400, {"error": {"message":
                                        "stream_options.include_usage must be an object"}}),
                       openai_reply(text="ok")])
    client = ModelClient(route(), transport=HttpTransport(http=t))
    assert client.chat([{"role": "user", "content": "x"}]).text == "ok"
    assert client.degraded and "stream_options" not in t.requests[1]


def test_the_degrade_latch_is_one_shot():
    t = FakeTransport([(400, {"error": {"message": "unknown field stream_options"}}),
                       openai_reply(text="first"), openai_reply(text="second")])
    client = ModelClient(route(), transport=HttpTransport(http=t))
    client.chat([{"role": "user", "content": "x"}])
    client.chat([{"role": "user", "content": "x"}])
    assert len(t.requests) == 3, "the rejection must be paid once per run, not per turn"
    assert "stream_options" not in t.requests[2]
    assert "stream" not in t.requests[2], "the degraded route stops streaming entirely"


def test_a_strict_streaming_endpoint_that_rejects_stream_options_still_finishes():
    requests: list[dict] = []

    def hook(method, url, headers, body, timeout):
        payload = json.loads(body or b"{}")
        requests.append(payload)
        if "stream_options" in payload:
            return 400, json.dumps({"error": {"message":
                                              "unknown field: stream_options"}}).encode()
        if "stream" in payload:
            # Streams without the usage field: SSE, which the degraded request does not
            # parse as JSON. That costs one retry, never the run.
            return 200, (b'data: {"choices": [{"delta": {"content": "late"}}]}\n\n'
                         b"data: [DONE]\n\n")
        return 200, json.dumps(openai_reply(text="ok")).encode()

    client = ModelClient(route(max_retries=2), transport=HttpTransport(http=hook))
    reply = client.chat([{"role": "user", "content": "x"}])
    assert reply.text == "ok"
    assert client.degraded == ["extras:400"], "the rejection is learned once for the route"
    assert requests[0]["stream_options"] == {"include_usage": True}
    assert len(requests) == 3
    assert "stream" not in requests[-1] and "stream_options" not in requests[-1]


def test_a_400_that_names_no_field_is_still_terminal():
    t = FakeTransport([(400, {"error": {"message": "bad request"}})])
    client = ModelClient(route(max_retries=0), transport=HttpTransport(http=t))
    with pytest.raises(ProviderError) as exc:
        client.chat([{"role": "user", "content": "x"}])
    assert exc.value.code == "bad_request"
    assert client.degraded == []
    assert len(t.requests) == 1


def test_a_normal_streaming_payload_keeps_its_exact_shape():
    t = FakeTransport([openai_reply(text="ok")])
    ModelClient(route(), transport=HttpTransport(http=t)).chat(
        [{"role": "user", "content": "x"}], max_tokens=64)
    first = t.requests[0]
    assert sorted(first) == ["max_tokens", "messages", "model", "stream", "stream_options",
                             "temperature"]
    assert first["stream_options"] == {"include_usage": True}
    assert first["max_tokens"] == 64
