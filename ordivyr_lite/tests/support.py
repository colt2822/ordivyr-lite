"""Test support: fake transports, an OpenAI-compatible scripted server and helpers."""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable

from ordivyr_lite.config import Route
from ordivyr_lite.providers import HttpTransport
from ordivyr_lite.router import ModelRouter
from ordivyr_lite.usage import UsageLedger


def openai_reply(*, text: str = "", calls: list[tuple[str, dict]] | None = None,
                 prompt_tokens: int = 100, completion_tokens: int = 10,
                 cached_tokens: int = 0, finish: str | None = None,
                 model: str = "test-model") -> dict:
    tool_calls = [{"id": f"call_{i}", "type": "function",
                   "function": {"name": name, "arguments": json.dumps(args)}}
                  for i, (name, args) in enumerate(calls or [])]
    msg = {"role": "assistant", "content": text or None}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    usage = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "total_tokens": prompt_tokens + completion_tokens,
             "prompt_tokens_details": {"cached_tokens": cached_tokens}}
    return {"id": "chatcmpl-test", "object": "chat.completion", "model": model,
            "choices": [{"index": 0, "message": msg,
                         "finish_reason": finish or ("tool_calls" if tool_calls else "stop")}],
            "usage": usage}


class FakeTransport:
    """Injected HTTP hook: a queue of responses or a callable(body)."""

    def __init__(self, responses):
        self.responses = list(responses) if isinstance(responses, list) else responses
        self.requests: list[dict] = []
        self.headers: list[dict] = []

    def __call__(self, method, url, headers, body, timeout):
        data = json.loads(body) if body else {}
        self.requests.append(data)
        self.headers.append(dict(headers))
        if callable(self.responses):
            result = self.responses(data)
        else:
            result = self.responses.pop(0)
        if isinstance(result, tuple):
            status, payload = result
            return status, json.dumps(payload).encode()
        return 200, json.dumps(result).encode()


def local_route(model: str = "test-model", **kw) -> Route:
    return Route(provider="local", protocol="openai", base_url="http://127.0.0.1:9/v1",
                 model=model, paid=False, timeout=5, max_retries=0, **kw)


def make_router(transport: FakeTransport, routes: list[Route] | None = None,
                tmp_path: Path | None = None, **kw) -> ModelRouter:
    ledger = UsageLedger(tmp_path / "usage.jsonl") if tmp_path else UsageLedger()
    return ModelRouter(routes or [local_route()], transport=HttpTransport(http=transport),
                       ledger=ledger, **kw)


class ScriptedServer:
    """Tiny OpenAI-compatible server for CLI/job end-to-end tests."""

    def __init__(self, responder: Callable[[dict], dict], *, model: str = "test-model",
                 delay: float = 0.0):
        self.responder = responder
        self.model = model
        self.delay = delay
        self.requests: list[dict] = []
        server = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def _send(self, code: int, payload: dict):
                raw = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def do_GET(self):
                if self.path.rstrip("/").endswith("/models"):
                    return self._send(200, {"object": "list",
                                            "data": [{"id": server.model, "object": "model"}]})
                self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    return self._send(404, {"error": "not found"})
                server.requests.append(body)
                if server.delay:
                    import time as _time
                    _time.sleep(server.delay)
                try:
                    reply = server.responder(body)
                except Exception as exc:  # noqa: BLE001
                    return self._send(500, {"error": {"message": str(exc)}})
                self._send(200, reply)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def file_tool_reply(body: dict) -> dict:
    """A deterministic two-step policy: read a file, then write b.txt, then finish."""
    messages = body.get("messages") or []
    tool_messages = [m for m in messages if m.get("role") == "tool"]
    if not tool_messages:
        return openai_reply(calls=[("read_file", {"path": "a.txt"})])
    if len(tool_messages) == 1:
        return openai_reply(calls=[("write_file", {"path": "b.txt", "content": "world\n"})])
    return openai_reply(text="done")
