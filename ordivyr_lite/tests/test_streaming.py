"""Streaming regression tests: SSE assembly, incremental output, tool-call safety.

Every test here talks to a real socket over a scripted SSE server, so it covers
the transport, the frame parser and the assembler together.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ordivyr_lite import profiles as PF
from ordivyr_lite.config import Route
from ordivyr_lite.errors import ProviderError
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.providers import HttpTransport, ModelClient
from ordivyr_lite.router import ModelRouter
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.usage import UsageLedger


class _Forward(HttpTransport):
    """Keep the shaped payload for a loopback port but deliver it to the test server."""

    def __init__(self, base: str):
        super().__init__()
        self._base = base.rstrip("/")

    def stream(self, method, url, headers, body, timeout):
        return super().stream(method, self._base + "/chat/completions", headers, body, timeout)


class SSEServer:
    """A scripted OpenAI/Anthropic-compatible endpoint with three reply modes."""

    def __init__(self, *, chunks=None, delay=0.0, status=200, body=None,
                 fail_with_extras=(), route="/chat/completions"):
        self.chunks = chunks or []
        self.delay = delay
        self.status = status
        self.body = body
        self.fail_with_extras = tuple(fail_with_extras)
        self.requests: list[dict] = []
        self.path = route
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code: int, payload: dict) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n) or b"{}")
                server.requests.append(payload)
                for key in server.fail_with_extras:
                    if key in payload:
                        return self._json(400, {"error": {"message": "unknown field " + key}})
                if server.status >= 400:
                    return self._json(server.status, server.body or {"error": {"message": "no"}})
                if server.body is not None:
                    return self._json(200, server.body)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                def write(block: bytes) -> None:
                    self.wfile.write(b"%x\r\n" % len(block) + block + b"\r\n")
                    self.wfile.flush()

                pieces = server.chunks
                if pieces and isinstance(pieces[0], list):
                    pieces = pieces[min(len(server.requests) - 1, len(pieces) - 1)]
                try:
                    for piece in pieces:
                        if server.delay:
                            time.sleep(server.delay)
                        write(("data: " + (piece if isinstance(piece, str) else
                                           json.dumps(piece)) + "\n\n").encode())
                    write(b"data: [DONE]\n\n")
                    write(b"")
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base = "http://127.0.0.1:" + str(self.port) + "/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()


def delta_chunk(content=None, *, reasoning=None, tool_calls=None, finish=None) -> dict:
    delta: dict = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}


def route_for(base: str, model: str = "m", provider: str = "bench",
              protocol: str = "openai") -> Route:
    return Route(provider=provider, protocol=protocol, base_url=base, model=model,
                 api_key="k", paid=False, authorized=True, timeout=20, max_retries=0)


def client_for(base: str, **kw) -> ModelClient:
    return ModelClient(route_for(base, **kw), transport=HttpTransport())


def test_sse_text_arrives_incrementally():
    chunks = [delta_chunk("Hel"), delta_chunk("lo"), delta_chunk(finish="stop"),
              {"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": 2,
                                        "prompt_tokens_details": {"cached_tokens": 8}}}]
    with SSEServer(chunks=chunks, delay=0.05) as srv:
        seen: list[tuple[str, str, float]] = []
        t0 = time.monotonic()
        reply = client_for(srv.base).chat([{"role": "user", "content": "hi"}],
                                          on_delta=lambda k, p: seen.append((k, p, time.monotonic())))
        total = time.monotonic() - t0
    assert reply.text == "Hello"
    assert reply.streamed is True
    assert [p for _, p, _ in seen] == ["Hel", "lo"]
    assert reply.ttft_s > 0
    first_delta = seen[0][2] - t0
    assert first_delta < total - 0.03, "the first token must arrive before the turn finishes"
    assert reply.usage["cache_read"] == 8
    assert reply.usage["input"] == 4
    assert reply.finish == "stop"


def test_tool_call_arguments_assemble_across_fragments():
    chunks = [
        delta_chunk(tool_calls=[{"index": 0, "id": "call_a",
                                 "function": {"name": "read_file", "arguments": ""}}]),
        delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": '{"pa'}}]),
        delta_chunk(tool_calls=[{"index": 0, "function": {"arguments": 'th": "a.txt"}'}}]),
        delta_chunk(finish="tool_calls"),
    ]
    with SSEServer(chunks=chunks) as srv:
        reply = client_for(srv.base).chat([{"role": "user", "content": "go"}])
    assert len(reply.calls) == 1
    call = reply.calls[0]
    assert (call.name, call.args, call.error) == ("read_file", {"path": "a.txt"}, None)
    assert call.id == "call_a"


def test_two_tool_calls_keep_their_index_order():
    chunks = [
        delta_chunk(tool_calls=[{"index": 1, "id": "b", "function": {"name": "grep",
                                                                     "arguments": '{"pattern": "x"}'}}]),
        delta_chunk(tool_calls=[{"index": 0, "id": "a", "function": {"name": "read_file",
                                                                     "arguments": '{"path": "y"}'}}]),
        delta_chunk(finish="tool_calls"),
    ]
    with SSEServer(chunks=chunks) as srv:
        reply = client_for(srv.base).chat([{"role": "user", "content": "go"}])
    assert [c.id for c in reply.calls] == ["a", "b"]


def test_cut_stream_is_flagged_and_never_yields_runnable_arguments():
    chunks = [
        delta_chunk(tool_calls=[{"index": 0, "id": "c", "function": {"name": "write_file",
                                                                     "arguments": '{"path": "x", '}}]),
    ]
    with SSEServer(chunks=chunks) as srv:
        reply = client_for(srv.base).chat([{"role": "user", "content": "go"}])
    assert reply.truncated is True
    assert reply.calls[0].error is not None
    assert reply.calls[0].args == {}


def test_llamacpp_timings_give_the_exact_cache_split():
    chunks = [
        delta_chunk("ok"),
        delta_chunk(finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 7},
         "timings": {"cache_n": 100, "prompt_n": 20, "predicted_n": 7}},
    ]
    with SSEServer(chunks=chunks) as srv:
        reply = client_for(srv.base).chat([{"role": "user", "content": "x"}])
    assert reply.usage["cache_read"] == 100
    assert reply.usage["input"] == 20
    assert reply.usage["context"] == 120
    assert reply.usage["output"] == 7


def test_reasoning_deltas_are_separate_from_the_answer():
    chunks = [delta_chunk(reasoning="thinking"), delta_chunk("answer"), delta_chunk(finish="stop")]
    with SSEServer(chunks=chunks) as srv:
        kinds: list[str] = []
        reply = client_for(srv.base).chat([{"role": "user", "content": "x"}],
                                          on_delta=lambda k, p: kinds.append(k))
    assert reply.text == "answer"
    assert kinds == ["reasoning", "text"]


def test_anthropic_stream_assembles_text_and_tool_use():
    events = [
        {"type": "message_start", "message": {"usage": {"input_tokens": 5,
                                                        "cache_read_input_tokens": 11,
                                                        "cache_creation_input_tokens": 2}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Hi"}},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "tu1", "name": "read_file", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"path": '}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '"a.txt"}'}},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
    ]
    with SSEServer(chunks=events, route="/v1/messages") as srv:
        reply = client_for(srv.base, protocol="anthropic").chat(
            [{"role": "user", "content": "go"}], tools=[{"type": "function", "function": {
                "name": "read_file", "description": "d", "parameters": {"type": "object"}}}])
    assert reply.text == "Hi"
    assert reply.calls[0].name == "read_file"
    assert reply.calls[0].args == {"path": "a.txt"}
    assert reply.usage["cache_read"] == 11
    assert reply.usage["cache_write"] == 2
    assert reply.usage["output"] == 9


def test_a_fenced_json_answer_is_not_executed_when_tools_were_offered():
    chunks = [delta_chunk('here is the shape: ' + PF.FENCE + 'json'
                          + '{"name": "read_file", "arguments": {"path": "a.txt"}}'
                          + PF.FENCE),
              delta_chunk(finish="stop")]
    with SSEServer(chunks=chunks) as srv:
        reply = client_for(srv.base).chat(
            [{"role": "user", "content": "what shape?"}],
            tools=[{"type": "function", "function": {"name": "read_file",
                                                     "parameters": {"type": "object"}}}])
    assert reply.calls == []
    assert reply.recovered is False


def test_a_bad_request_that_is_not_about_a_field_does_not_degrade():
    chunks = [delta_chunk(finish="stop")]
    with SSEServer(status=400, body={"error": {"message": "maximum context length is 8192"}},
                   chunks=chunks) as srv:
        client = client_for("http://127.0.0.1:8181/v1", model="Qwen3-Coder")
        client.route.base_url = srv.base
        client.route.context_window = None
        with pytest.raises(ProviderError):
            client.chat([{"role": "user", "content": "x"}])
    assert client.degraded == []


def test_streaming_falls_back_to_plain_json():
    body = {"choices": [{"message": {"role": "assistant", "content": "plain"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 4, "completion_tokens": 1}}
    with SSEServer(body=body) as srv:
        reply = client_for(srv.base).chat([{"role": "user", "content": "x"}])
        assert srv.requests[0]["stream"] is True
    assert reply.text == "plain"
    assert reply.streamed is False


def test_streaming_can_be_disabled_by_the_operator(monkeypatch):
    with SSEServer(body={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}) as srv:
        monkeypatch.setenv("ORDIVYR_LITE_STREAM", "0")
        client_for(srv.base).chat([{"role": "user", "content": "x"}])
        assert "stream" not in srv.requests[0]


def test_deepseek_profile_sends_no_parallel_or_cache_fields():
    with SSEServer(body={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}) as srv:
        client_for(srv.base, model="deepseek-chat", provider="deepseek").chat(
            [{"role": "user", "content": "x"}],
            tools=[{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}])
        payload = srv.requests[0]
    assert payload["tools"] and payload["tool_choice"] == "auto"
    assert "parallel_tool_calls" not in payload
    assert "prompt_cache_key" not in payload
    assert "cache_prompt" not in payload
    assert payload["max_tokens"] > 0


def test_openai_profile_sets_a_stable_cache_key():
    with SSEServer(body={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}) as srv:
        client = client_for(srv.base, model="gpt-4.1", provider="openai")
        msgs = [{"role": "system", "content": "same"}, {"role": "user", "content": "task"}]
        tools = [{"type": "function", "function": {"name": "f",
                                                        "parameters": {"type": "object"}}}]
        client.chat(msgs, tools=tools)
        client.chat(msgs, tools=tools)
    first, second = srv.requests
    assert first["prompt_cache_key"] == second["prompt_cache_key"]
    assert first["parallel_tool_calls"] is True


def test_o_series_uses_max_completion_tokens_and_no_temperature():
    with SSEServer(body={"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}) as srv:
        client_for(srv.base, model="o3-mini", provider="openai").chat(
            [{"role": "user", "content": "x"}])
        payload = srv.requests[0]
    assert "max_completion_tokens" in payload
    assert "max_tokens" not in payload
    assert "temperature" not in payload


def test_qwen2_on_the_fleet_port_caches_without_a_thinking_flag():
    route = route_for("http://127.0.0.1:11436/v1", model="Qwen2.5-Coder-7B-Instruct-Q5_K_M",
                      provider="llamacpp")
    profile = PF.detect(route.model, route.provider, route.base_url)
    payload = PF.openai_payload(profile, route, messages=[{"role": "user", "content": "x"}],
                                tools=None, max_tokens=64, temperature=0.0, stream=False)
    assert payload["cache_prompt"] is True
    assert "chat_template_kwargs" not in payload


def test_qwen_on_llamacpp_asks_for_a_warm_prefix_and_no_thinking():
    route = route_for("http://127.0.0.1:8181/v1", model="Qwen3-Coder-30B-A3B", provider="llamacpp")
    profile = PF.detect(route.model, route.provider, route.base_url)
    payload = PF.openai_payload(profile, route, messages=[{"role": "user", "content": "x"}],
                                tools=None, max_tokens=64, temperature=0.0, stream=True)
    assert payload["cache_prompt"] is True
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert profile.cache_block == 32


def test_endpoint_that_rejects_extras_is_retried_without_them():
    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    with SSEServer(body=body, fail_with_extras=("chat_template_kwargs",)) as srv:
        route = route_for(srv.base, model="Qwen3-Coder-30B-A3B", provider="qwen")
        route.base_url = "http://127.0.0.1:8181/v1"
        route.headers = {"X-Bench-Forward": srv.base}
        client = ModelClient(route, transport=_Forward(srv.base))
        reply = client.chat([{"role": "user", "content": "x"}])
        assert len(srv.requests) == 2
        assert "chat_template_kwargs" not in srv.requests[1]
    assert reply.text == "ok"
    assert client.degraded


def test_a_strict_endpoint_is_probed_once_per_route():
    from pathlib import Path as _Path
    import tempfile as _tempfile

    from ordivyr_lite.router import ModelRouter
    from ordivyr_lite.usage import UsageLedger

    body = {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]}
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}]
    with SSEServer(body=body, fail_with_extras=("prompt_cache_key",)) as srv:
        router = ModelRouter([route_for(srv.base, model="gpt-4.1", provider="openai")],
                             ledger=UsageLedger(_Path(_tempfile.mkdtemp()) / "u.jsonl"),
                             transport=HttpTransport())
        first = router.chat([{"role": "user", "content": "a"}], tools=tools)
        second = router.chat([{"role": "user", "content": "b"}], tools=tools)
        assert len(srv.requests) == 3, "the rejection must be learned, not repaid every turn"
    assert (first.reply.text, second.reply.text) == ("ok", "ok")


def test_a_message_that_is_only_a_tool_call_runs_when_tools_were_offered(tmp_path):
    (tmp_path / "NOTE.txt").write_text("42\n", encoding="utf-8")
    body = {"choices": [{"message": {"content":
            '{"name": "read_file", "arguments": {"path": "NOTE.txt"}}'},
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12}}
    done = {"choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 1}}
    state = {"n": 0}

    def respond(_request):
        state["n"] += 1
        return body if state["n"] == 1 else done

    from ordivyr_lite.tests.support import ScriptedServer
    srv = ScriptedServer(respond).start()
    try:
        box = ToolBox(tmp_path, profile="workspace")
        router = ModelRouter([route_for(srv.base, model="Qwen2.5-Coder-7B")],
                             ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        result = Runner(router, box, "Read NOTE.txt and reply with only the integer.",
                        limits=RunLimits(max_turns=3), stream=False).run()
    finally:
        srv.stop()
    assert [ev["tool"] for ev in result.tool_events] == ["read_file"]
    assert result.text == "42"


def test_a_fenced_python_tool_call_is_executed_once(tmp_path):
    (tmp_path / "text.py").write_text('"""word utilities"""\n', encoding="utf-8")
    fence = ("```python\n"
             "write_file({\"path\": \"text.py\", \"content\": "
             "\"def count_words(text):\\n    return len(text.split())\\n\"})\n"
             "```\n\ncount_words function added to text.py.")
    replies = [
        {"choices": [{"message": {"content": fence}, "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 40, "completion_tokens": 30}},
        {"choices": [{"message": {"content": "count_words added to text.py."},
                      "finish_reason": "stop"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 8}},
    ]
    from ordivyr_lite.tests.support import ScriptedServer
    srv = ScriptedServer(lambda _request: replies.pop(0)).start()
    try:
        box = ToolBox(tmp_path, profile="workspace")
        router = ModelRouter([route_for(srv.base, model="Qwen2.5-Coder-7B")],
                             ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        result = Runner(router, box, "Add count_words(text) to text.py.",
                        limits=RunLimits(max_turns=4), stream=False).run()
    finally:
        srv.stop()
    assert [ev["tool"] for ev in result.tool_events] == ["write_file"]
    assert "def count_words" in (tmp_path / "text.py").read_text(encoding="utf-8")
    assert result.stats.recovered_calls == 1


def test_a_fenced_function_definition_is_not_a_tool_call(tmp_path):
    fence = ("```python\ndef count_words(text):\n    return len(text.split())\n```\n\n"
              "count_words function added to text.py.")
    body = {"choices": [{"message": {"content": fence}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 20}}
    from ordivyr_lite.tests.support import ScriptedServer
    srv = ScriptedServer(lambda _request: body).start()
    try:
        box = ToolBox(tmp_path, profile="workspace")
        router = ModelRouter([route_for(srv.base, model="Qwen2.5-Coder-7B")],
                             ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        result = Runner(router, box, "Add count_words(text) to text.py.",
                        limits=RunLimits(max_turns=2), stream=False).run()
    finally:
        srv.stop()
    assert result.tool_events == []


def test_a_sentence_that_mentions_a_call_is_not_executed(tmp_path):
    body = {"choices": [{"message": {"content":
            'I would call {"name": "read_file", "arguments": {"path": "NOTE.txt"}} next.'},
            "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12}}
    from ordivyr_lite.tests.support import ScriptedServer
    srv = ScriptedServer(lambda _request: body).start()
    try:
        box = ToolBox(tmp_path, profile="workspace")
        router = ModelRouter([route_for(srv.base, model="Qwen2.5-Coder-7B")],
                             ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        result = Runner(router, box, "Read NOTE.txt", limits=RunLimits(max_turns=2),
                        stream=False).run()
    finally:
        srv.stop()
    assert result.tool_events == []
    assert result.stats.tool_calls == 0


def test_tool_calls_written_into_the_text_are_recovered_once():
    chunks = [delta_chunk('Running it now. <tool_call>{"name": "read_file", '
                          '"arguments": {"path": "a.txt"}}</tool_call>'),
              delta_chunk(finish="stop")]
    with SSEServer(chunks=chunks) as srv:
        client = ClientNoTools(srv.base)
        reply = client.chat([{"role": "user", "content": "go"}],
                            tools=[{"type": "function", "function": {
                                "name": "read_file", "parameters": {"type": "object"}}}])
        payload = srv.requests[0]
    assert "tools" not in payload
    assert reply.recovered is True
    assert reply.calls[0].name == "read_file"
    assert reply.calls[0].args == {"path": "a.txt"}


def ClientNoTools(base: str) -> ModelClient:
    client = ModelClient(route_for(base, model="deepseek-reasoner", provider="deepseek"),
                         transport=HttpTransport())
    assert client.profile.supports_tools is False
    return client


def test_loop_records_ttft_and_streams_deltas(tmp_path):
    (tmp_path / "a.txt").write_text("hello\n", "utf-8")
    chunks = [
        [delta_chunk(tool_calls=[{"index": 0, "id": "c1", "function": {
            "name": "read_file", "arguments": '{"path": "a.txt"}'}}]),
         delta_chunk(finish="tool_calls"),
         {"choices": [], "usage": {"prompt_tokens": 30, "completion_tokens": 5}}],
        [delta_chunk("the file says hello"),
         delta_chunk(finish="stop"),
         {"choices": [], "usage": {"prompt_tokens": 60, "completion_tokens": 6}}],
    ]
    with SSEServer(chunks=chunks) as srv:
        box = ToolBox(tmp_path, profile="read-only")
        router = ModelRouter([route_for(srv.base)], ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        events: list[dict] = []
        runner = Runner(router, box, "read a.txt", limits=RunLimits(max_turns=5),
                        on_event=events.append)
        result = runner.run()
    assert result.stop_reason == "done"
    assert result.stats.streamed_turns == 2
    assert result.stats.first_token_ms > 0
    assert any(e.get("type") == "delta" for e in events)
    assert [e["tool"] for e in result.tool_events] == ["read_file"]


def test_loop_executes_nothing_from_a_cut_stream(tmp_path):
    chunks = [
        delta_chunk(tool_calls=[{"index": 0, "id": "c1", "function": {
            "name": "write_file", "arguments": '{"path": "b.txt", "cont'}}]),
    ]
    with SSEServer(chunks=chunks) as srv:
        box = ToolBox(tmp_path, profile="workspace")
        router = ModelRouter([route_for(srv.base)], ledger=UsageLedger(tmp_path / "usage.jsonl"),
                             transport=HttpTransport())
        result = Runner(router, box, "write b.txt", limits=RunLimits(max_turns=2)).run()
    assert not (tmp_path / "b.txt").exists()
    assert result.stats.tool_calls == 0
    assert result.stats.truncated_turns >= 1


def test_a_closed_keep_alive_connection_is_reopened_once():
    chunks = [delta_chunk("one"), delta_chunk(finish="stop")]
    with SSEServer(chunks=chunks) as srv:
        client = client_for(srv.base)
        first = client.chat([{"role": "user", "content": "a"}])
        second = client.chat([{"role": "user", "content": "b"}])
        assert len(srv.requests) == 2
    assert (first.text, second.text) == ("one", "one")


def test_stream_socket_error_becomes_a_retryable_provider_error(tmp_path):
    class Dead(SSEServer):
        def __init__(self):
            super().__init__(chunks=[delta_chunk("x")])

    srv = Dead()
    srv.__enter__()
    srv.httpd.shutdown()
    srv.httpd.server_close()
    client = ModelClient(route_for(srv.base), transport=HttpTransport())
    with pytest.raises(ProviderError) as exc:
        client.chat([{"role": "user", "content": "x"}], retries=0)
    assert exc.value.code in ("network", "timeout")
