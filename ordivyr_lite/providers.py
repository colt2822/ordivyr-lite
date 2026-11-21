"""One small stdlib HTTP client for the two wire protocols Lite speaks.

  openai     POST {base}/chat/completions   OpenAI, OpenRouter, Groq, DeepSeek,
                                            Mistral, xAI, Together, Ollama,
                                            LM Studio, llama.cpp, vLLM, ...
  anthropic  POST {base}/v1/messages        Anthropic Messages

The client normalises both protocols into one provider-neutral history (OpenAI
message shape) and one usage record (fresh input / cache read / cache write /
output / context).  Anthropic cache breakpoints are applied to the request view
only; the stored history is never mutated.
"""

from __future__ import annotations

import http.client
import json
import re
import secrets
import socket
import ssl
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator
from urllib.parse import urlparse

from . import profiles as PF
from .config import Route
from .errors import ProviderError

MAX_TOOL_CALLS_PER_TURN = 64
_TOKEN_ERR = re.compile(
    r"max_tokens|max_completion_tokens|maximum context|context.?length|context window|"
    r"prompt is too long|prompt too long|input is too long|too many tokens|too long: |"
    r"too large|exceeds? |output token", re.I)
_CACHE_BREAKPOINT = {"type": "ephemeral"}


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict
    raw: str = "{}"
    error: str | None = None
    repaired: bool = False


@dataclass
class ModelReply:
    text: str
    calls: list[ToolCall]
    usage: dict
    finish: str
    model: str
    protocol: str
    latency_s: float
    raw_message: dict = field(default_factory=dict)
    ttft_s: float = 0.0
    streamed: bool = False
    recovered: bool = False

    @property
    def truncated(self) -> bool:
        return self.finish in ("length", "max_tokens", "max_output_tokens")


def _read_lines(resp) -> Iterator[bytes]:
    while True:
        line = resp.readline()
        if not line:
            return
        yield line


@dataclass
class StreamResponse:
    """One HTTP response whose body is consumed incrementally.

    An injected transport hook has no socket to stream from, so it delivers the
    whole body up front and the caller detects a JSON reply instead of SSE.
    """

    status: int
    content_type: str = ""
    raw_lines: Iterator[bytes] | None = None
    closer: Callable[[], None] | None = None
    early: bytes = b""

    def is_sse(self, want: bool) -> bool:
        if not want:
            return False
        ct = (self.content_type or "").lower()
        if "event-stream" in ct:
            return True
        if self.raw_lines is not None:
            return "json" not in ct
        return self.early.lstrip()[:5] == b"data:"

    def lines(self) -> Iterator[bytes]:
        if self.raw_lines is None:
            for line in self.early.splitlines(keepends=True):
                yield line
            return
        yield from self.raw_lines

    def read_all(self) -> bytes:
        if self.raw_lines is None:
            return self.early
        return self.early + b"".join(self.lines())

    def close(self) -> None:
        if self.closer is not None:
            closer, self.closer = self.closer, None
            try:
                closer()
            except OSError:
                pass


class HttpTransport:
    """Keep-alive HTTP(S) with one connection per (thread, scheme, host, port).

    Proxies and redirects fall back to urllib.  Every call takes an injectable
    `http` hook so tests never touch a network.
    """

    def __init__(self, http: Callable | None = None):
        self._tl = threading.local()
        self.http = http

    def __call__(self, method: str, url: str, headers: dict, body: bytes | None,
                 timeout: float) -> tuple[int, bytes]:
        if self.http is not None:
            return self.http(method, url, headers, body, timeout)
        return self._urlopen(method, url, headers, body, timeout)

    def stream(self, method: str, url: str, headers: dict, body: bytes | None,
               timeout: float) -> StreamResponse:
        """Open a response without buffering the whole body (SSE or JSON)."""
        if self.http is not None:
            status, raw = self.http(method, url, headers, body, timeout)
            return StreamResponse(status=status, early=raw or b"")
        return self._open_stream(method, url, headers, body, timeout)

    def _pool(self) -> dict:
        pool = getattr(self._tl, "pool", None)
        if pool is None:
            pool = self._tl.pool = {}
        return pool

    def _open_stream(self, method, url, headers, body, timeout) -> StreamResponse:
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return self._open_stream_slow(method, url, headers, body, timeout)
        try:
            if urllib.request.getproxies().get(u.scheme) and not urllib.request.proxy_bypass(u.hostname):
                return self._open_stream_slow(method, url, headers, body, timeout)
        except Exception:  # noqa: BLE001 - proxy detection is best effort
            return self._open_stream_slow(method, url, headers, body, timeout)
        key = (u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80))
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        hdr = dict(headers)
        hdr.setdefault("Accept-Encoding", "identity")
        hdr.setdefault("Accept", "text/event-stream")
        pool = self._pool()
        for attempt in (0, 1):
            conn = pool.get(key)
            fresh = conn is None
            if fresh:
                cls = (http.client.HTTPSConnection if u.scheme == "https"
                       else http.client.HTTPConnection)
                conn = cls(key[1], key[2], timeout=timeout)
                pool[key] = conn
            try:
                conn.timeout = timeout
                if conn.sock:
                    conn.sock.settimeout(timeout)
                if fresh:
                    conn.connect()
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.request(method, path, body=body, headers=hdr)
                resp = conn.getresponse()
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest,
                    http.client.NotConnected, ConnectionResetError, BrokenPipeError,
                    http.client.BadStatusLine) as e:
                conn.close()
                pool.pop(key, None)
                if fresh or attempt:
                    raise ProviderError(f"cannot reach the provider: {e}", code="network",
                                        retryable=True, endpoint=url) from e
                continue
            except (TimeoutError, socket.timeout) as e:
                conn.close()
                pool.pop(key, None)
                raise ProviderError(f"provider timed out: {e}", code="timeout",
                                    retryable=True, endpoint=url) from e
            except OSError as e:
                conn.close()
                pool.pop(key, None)
                raise ProviderError(f"cannot reach the provider: {e}", code="network",
                                    retryable=True, endpoint=url) from e
            break
        if 300 <= resp.status < 400:
            try:
                resp.read()
            except OSError:
                pass
            conn.close()
            pool.pop(key, None)
            return self._open_stream_slow(method, url, headers, body, timeout)

        def _close() -> None:
            try:
                resp.close()
            except OSError:
                pass
            if resp.will_close:
                try:
                    conn.close()
                finally:
                    pool.pop(key, None)

        return StreamResponse(status=resp.status,
                              content_type=resp.getheader("content-type") or "",
                              raw_lines=_read_lines(resp), closer=_close)

    def _open_stream_slow(self, method, url, headers, body, timeout) -> StreamResponse:
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            return StreamResponse(status=e.code, early=e.read() or b"")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"cannot reach the provider: {getattr(e, 'reason', e)}",
                                code="network", retryable=True, endpoint=url) from e
        return StreamResponse(status=resp.status, content_type=resp.headers.get("content-type") or "",
                              raw_lines=_read_lines(resp), closer=resp.close)


    def _slow_path(self, method, url, headers, body, timeout):
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise ProviderError(f"cannot reach the provider: {getattr(e, 'reason', e)}",
                                code="network", retryable=True, endpoint=url) from e

    def _urlopen(self, method, url, headers, body, timeout):
        u = urlparse(url)
        if u.scheme not in ("http", "https") or not u.hostname:
            return self._slow_path(method, url, headers, body, timeout)
        try:
            if urllib.request.getproxies().get(u.scheme) and not urllib.request.proxy_bypass(u.hostname):
                return self._slow_path(method, url, headers, body, timeout)
        except Exception:
            return self._slow_path(method, url, headers, body, timeout)
        pool = getattr(self._tl, "pool", None)
        if pool is None:
            pool = self._tl.pool = {}
        key = (u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80))
        path = (u.path or "/") + (("?" + u.query) if u.query else "")
        hdr = dict(headers)
        hdr.setdefault("Accept-Encoding", "identity")
        for attempt in (0, 1):
            conn = pool.get(key)
            fresh = conn is None
            if fresh:
                cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                conn = cls(key[1], key[2], timeout=timeout)
                pool[key] = conn
            else:
                try:
                    conn.timeout = timeout
                    if conn.sock:
                        conn.sock.settimeout(timeout)
                except OSError:
                    conn.close()
                    pool.pop(key, None)
                    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
                    conn = cls(key[1], key[2], timeout=timeout)
                    pool[key] = conn
                    fresh = True
            try:
                if fresh:
                    conn.connect()
                    conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                conn.request(method, path, body=body, headers=hdr)
                r = conn.getresponse()
                data = r.read()
                status = r.status
            except (http.client.RemoteDisconnected, http.client.CannotSendRequest,
                    http.client.NotConnected, ConnectionResetError, BrokenPipeError) as e:
                conn.close()
                pool.pop(key, None)
                if fresh or attempt:
                    raise ProviderError(f"cannot reach the provider: {e}", code="network",
                                        retryable=True, endpoint=url) from e
                continue
            except (TimeoutError, socket.timeout) as e:
                conn.close()
                pool.pop(key, None)
                raise ProviderError(f"provider timed out: {e}", code="timeout",
                                    retryable=True, endpoint=url) from e
            except OSError as e:
                conn.close()
                pool.pop(key, None)
                raise ProviderError(f"cannot reach the provider: {e}", code="network",
                                    retryable=True, endpoint=url) from e
            if r.will_close:
                conn.close()
                pool.pop(key, None)
            if 300 <= status < 400:
                return self._slow_path(method, url, headers, body, timeout)
            return status, data
        raise ProviderError("cannot reach the provider", code="network", retryable=True, endpoint=url)


def _headers(route: Route) -> dict:
    h = {"Content-Type": "application/json", "User-Agent": "ordivyr-lite/0.1"}
    if route.protocol == "anthropic":
        h["anthropic-version"] = "2023-06-01"
        if route.api_key:
            h["x-api-key"] = route.api_key
    elif route.api_key:
        h["Authorization"] = "Bearer " + route.api_key
    h.update(route.headers or {})
    return h


def _cache_ok(base: str) -> bool:
    h = (urlparse(base).hostname or "").lower()
    return h == "anthropic.com" or h.endswith(".anthropic.com")


def _parse_http_error(status: int, raw: bytes, route: Route) -> ProviderError:
    try:
        body = json.loads(raw or b"{}")
    except ValueError:
        body = {}
    err = body.get("error", body) if isinstance(body, dict) else body
    message = json.dumps(err)[:300] if not isinstance(err, str) else err[:300]
    low = message.lower()
    endpoint = route.base_url
    if status in (401, 403):
        return ProviderError("the provider rejected the API key", code="auth", status=status, endpoint=endpoint)
    if status == 429 or re.search(r"rate.?limit|quota|insufficient_quota|overloaded", low):
        return ProviderError("provider rate limit or quota reached", code="rate_limit",
                             status=status, retryable=True, endpoint=endpoint)
    if status in (400, 404, 413, 422):
        code = "token_limit" if _TOKEN_ERR.search(message) else "bad_request"
        return ProviderError(f"provider rejected the request (HTTP {status}): {message}",
                             code=code, status=status, endpoint=endpoint)
    return ProviderError(f"provider error (HTTP {status}): {message}", code="server",
                         status=status, retryable=status >= 500, endpoint=endpoint)


def _call(transport: HttpTransport, route: Route, method: str, path: str,
          payload: dict | None, timeout: float) -> dict:
    url = route.base_url + path
    try:
        status, raw = transport(method, url, _headers(route),
                                None if payload is None else json.dumps(payload).encode(), timeout)
    except ProviderError:
        raise
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise ProviderError(f"cannot reach the provider: {e}", code="network", retryable=True,
                            endpoint=route.base_url) from e
    if status >= 400:
        raise _parse_http_error(status, raw, route)
    try:
        return json.loads(raw or b"{}")
    except ValueError as e:
        raise ProviderError("provider returned invalid JSON", code="bad_response",
                            retryable=True, endpoint=route.base_url) from e


def _usage_openai(j: dict) -> dict:
    u = j.get("usage") or {}
    prompt = int(u.get("prompt_tokens") or 0)
    completion = int(u.get("completion_tokens") or 0)
    if not prompt and not completion:
        prompt = int(u.get("total_tokens") or 0)
    details = u.get("prompt_tokens_details") or {}
    cached = min(prompt, int(details.get("cached_tokens") or u.get("prompt_cache_hit_tokens") or 0))
    reasoning = int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    return {
        "input": prompt - cached,
        "cache_read": cached,
        "cache_write": 0,
        "output": completion,
        "context": prompt,
        "telemetry": bool(details or "prompt_cache_hit_tokens" in u),
        "reasoning": reasoning,
    }


def _usage_anthropic(j: dict) -> dict:
    u = j.get("usage") or {}
    fresh = int(u.get("input_tokens") or 0)
    cache_read = int(u.get("cache_read_input_tokens") or 0)
    cache_write = int(u.get("cache_creation_input_tokens") or 0)
    return {
        "input": fresh,
        "cache_read": cache_read,
        "cache_write": cache_write,
        "output": int(u.get("output_tokens") or 0),
        "context": fresh + cache_read + cache_write,
        "telemetry": True,
        "reasoning": 0,
    }


_ARG_WINDOW = 120
_ARG_ERROR_CAP = 400


def _strip_trailing_commas(text: str) -> str:
    """Repair (a): drop a comma sitting right before a closer, outside strings."""
    out: list[str] = []
    in_str = escaped = False
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if in_str:
            out.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            i += 1
            continue
        if ch == '"':
            in_str = True
            out.append(ch)
            i += 1
            continue
        if ch == ",":
            j = i + 1
            while j < n and text[j] in " \t\r\n":
                j += 1
            if j < n and text[j] in "}]":
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _close_open_structures(text: str) -> str:
    """Repair (b): append the closers of every brace/bracket left open outside strings."""
    closers: list[str] = []
    in_str = escaped = False
    for ch in text:
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            closers.append("}")
        elif ch == "[":
            closers.append("]")
        elif ch in "}]":
            if not closers or closers.pop() != ch:
                return text  # a mismatched closer is not a shape repair can fix
    if in_str or not closers:
        # An unterminated string would need invented text; leave it alone.
        return text
    return text + "".join(reversed(closers))


def _repair_args(raw: str) -> str | None:
    """Bounded repair of arguments that failed to parse, in a fixed order.

    Trailing comma, then an object that ends early, then single-quoted strings.
    The result is accepted only when it parses to a JSON object.  Values are
    never invented, missing keys are never filled in, and text that already
    parsed is never passed in here.
    """
    fixed = _close_open_structures(_strip_trailing_commas(raw))
    if '"' not in fixed and "'" in fixed:
        fixed = fixed.replace("'", '"')
    try:
        parsed = json.loads(fixed)
    except (ValueError, TypeError):
        return None
    return fixed if isinstance(parsed, dict) else None


def _error_window(raw: str, exc: ValueError) -> str:
    """The failing column plus the ~120 characters around it, never the head."""
    n = len(raw)
    pos = getattr(exc, "pos", None)
    if not isinstance(pos, int) or pos < 0:
        return f"char 0 of {n}: {raw[:_ARG_WINDOW]!r}"
    pos = min(pos, n)
    start = max(0, min(pos - _ARG_WINDOW // 2, n - _ARG_WINDOW))
    return f"char {pos} of {n}: {raw[start:start + _ARG_WINDOW]!r}"


def _parse_args(raw: Any) -> tuple[dict, str, str | None, bool]:
    """-> (args, raw_text, error, repaired).  A malformed call NEVER gets defaulted args."""
    if isinstance(raw, dict):
        return raw, json.dumps(raw, sort_keys=True), None, False
    if raw in (None, ""):
        return {}, "{}", None, False
    if not isinstance(raw, str):
        return {}, json.dumps(raw, default=str), "arguments must be a JSON object", False
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError) as e:
        fixed = _repair_args(raw)
        if fixed is not None:
            return json.loads(fixed), fixed, None, True
        why = str(getattr(e, "msg", "") or e)[:60]
        msg = f"malformed JSON arguments: {why} at {_error_window(raw, e)}"
        return {}, raw, msg[:_ARG_ERROR_CAP], False
    if not isinstance(parsed, dict):
        return {}, raw, "arguments must be a JSON object", False
    return parsed, raw, None, False


def _norm_openai(j: dict, route: Route) -> ModelReply:
    ch = (j.get("choices") or [{}])[0]
    message = ch.get("message") or {}
    calls: list[ToolCall] = []
    for c in (message.get("tool_calls") or [])[:MAX_TOOL_CALLS_PER_TURN]:
        fn = c.get("function") or {}
        args, raw, err, repaired = _parse_args(fn.get("arguments"))
        calls.append(ToolCall(id=c.get("id") or "call_" + secrets.token_hex(5),
                              name=fn.get("name") or "", args=args, raw=raw, error=err,
                              repaired=repaired))
    text = message.get("content") or ""
    if isinstance(text, list):
        text = "".join(b.get("text", "") for b in text if isinstance(b, dict))
    return ModelReply(text=str(text), calls=calls, usage=_usage_openai(j),
                      finish=ch.get("finish_reason") or "", model=j.get("model") or route.model,
                      protocol="openai", latency_s=0.0, raw_message=message)


def _norm_anthropic(j: dict, route: Route) -> ModelReply:
    calls: list[ToolCall] = []
    text_parts: list[str] = []
    for b in (j.get("content") or []):
        if not isinstance(b, dict):
            continue
        if b.get("type") == "text":
            text_parts.append(b.get("text", ""))
        elif b.get("type") == "tool_use":
            args = b.get("input")
            if isinstance(args, dict):
                calls.append(ToolCall(id=b.get("id") or "call_" + secrets.token_hex(5),
                                      name=b.get("name") or "", args=args,
                                      raw=json.dumps(args, sort_keys=True), error=None))
            else:
                calls.append(ToolCall(id=b.get("id") or "call_" + secrets.token_hex(5),
                                      name=b.get("name") or "", args={},
                                      raw=json.dumps(args, default=str),
                                      error="arguments must be a JSON object"))
    return ModelReply(text="".join(text_parts), calls=calls[:MAX_TOOL_CALLS_PER_TURN],
                      usage=_usage_anthropic(j), finish=j.get("stop_reason") or "",
                      model=j.get("model") or route.model, protocol="anthropic",
                      latency_s=0.0, raw_message={"content": j.get("content") or []})


def _convert_anthropic_messages(messages: list[dict], cache: bool) -> tuple[list[dict], str]:
    """Canonical OpenAI-shaped history -> Anthropic messages + system text.

    Consecutive tool results are grouped into one user message, as the API
    requires.  Cache breakpoints are applied to the request view only.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            if isinstance(content, str):
                system_parts.append(content)
            continue
        if role == "assistant" and m.get("tool_calls"):
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": str(content)})
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                raw = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw) if isinstance(raw, str) else raw
                except (ValueError, TypeError):
                    args = {}
                if not isinstance(args, dict):
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id") or "call",
                               "name": fn.get("name") or "", "input": args})
            out.append({"role": "assistant", "content": blocks})
            continue
        if role == "tool":
            blk = {"type": "tool_result", "tool_use_id": m.get("tool_call_id") or "",
                   "content": str(content) if content is not None else ""}
            if out and out[-1].get("role") == "user" and isinstance(out[-1].get("content"), list) \
                    and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                out[-1]["content"].append(blk)
            else:
                out.append({"role": "user", "content": [blk]})
            continue
        if role in ("user", "assistant"):
            if isinstance(content, list):
                out.append({"role": role, "content": content})
            else:
                out.append({"role": role, "content": [{"type": "text", "text": str(content or "")}]})
    if cache and out:
        # Mark the end of this request and the end of the previous request.
        for idx in (len(out) - 1, len(out) - 3):
            if 0 <= idx < len(out):
                blocks = out[idx].get("content")
                if isinstance(blocks, list) and blocks and isinstance(blocks[-1], dict) \
                        and blocks[-1].get("type") not in ("thinking", "redacted_thinking"):
                    blocks[-1] = {**blocks[-1], "cache_control": dict(_CACHE_BREAKPOINT)}
    return out, "\n".join(system_parts)


def _iter_events(lines: Iterator[bytes]) -> Iterator[dict]:
    """SSE frames -> JSON payloads, with the socket errors translated to provider errors."""
    data: list[str] = []
    try:
        for rawblob in lines:
            try:
                line = rawblob.decode("utf-8", "replace")
            except AttributeError:
                line = str(rawblob)
            line = line.rstrip("\r\n")
            if not line:
                if not data:
                    continue
                payload = "\n".join(data)
                data = []
                if payload.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(payload)
                except ValueError as e:
                    raise ProviderError("provider sent a malformed stream frame",
                                        code="bad_response", retryable=True) from e
                continue
            if line.startswith(":"):
                continue
            if line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            payload = "\n".join(data)
            if payload.strip() not in ("", "[DONE]"):
                try:
                    yield json.loads(payload)
                except ValueError:
                    return
    except (OSError, TimeoutError, socket.timeout, http.client.HTTPException) as e:
        raise ProviderError(f"provider stream broke off: {e}", code="network",
                            retryable=True) from e


@dataclass
class _CallBuilder:
    index: int
    id: str = ""
    name: str = ""
    args: list[str] = field(default_factory=list)


def _usage_from_timings(usage: dict, timings: Any) -> dict:
    """llama.cpp reports the exact prompt/cache split; prefer it over arithmetic."""
    if not isinstance(timings, dict):
        return usage
    cached = int(timings.get("cache_n") or 0)
    prompt_n = int(timings.get("prompt_n") or 0)
    predicted = int(timings.get("predicted_n") or 0)
    if cached or prompt_n:
        usage = {**usage, "cache_read": cached, "input": prompt_n,
                 "context": cached + prompt_n, "telemetry": True}
    if predicted and not int(usage.get("output") or 0):
        usage = {**usage, "output": predicted}
    return usage


def _assemble_openai_stream(events: Iterator[dict], route: Route, on_delta, t0: float,
                            clock) -> ModelReply:
    text: list[str] = []
    reasoning: list[str] = []
    builders: dict[int, _CallBuilder] = {}
    finish = ""
    usage_raw: dict = {}
    timings: Any = None
    ttft = [0.0]

    def note_first() -> None:
        if not ttft[0]:
            ttft[0] = max(0.0, clock() - t0)

    for ev in events:
        if not isinstance(ev, dict):
            continue
        if ev.get("usage"):
            usage_raw = {**usage_raw, **(ev.get("usage") or {})}
        if ev.get("timings") is not None:
            timings = ev.get("timings")
        for ch in (ev.get("choices") or []):
            if not isinstance(ch, dict):
                continue
            delta = ch.get("delta") or {}
            if ch.get("finish_reason"):
                finish = str(ch["finish_reason"])
            piece = delta.get("content")
            if isinstance(piece, list):
                piece = "".join(b.get("text", "") for b in piece if isinstance(b, dict))
            if isinstance(piece, str) and piece:
                note_first()
                text.append(piece)
                if on_delta:
                    on_delta("text", piece)
            thought = delta.get(PF.REASONING_FIELD) or delta.get("reasoning") or delta.get("thinking")
            if isinstance(thought, str) and thought:
                note_first()
                reasoning.append(thought)
                if on_delta:
                    on_delta("reasoning", thought)
            for tc in (delta.get("tool_calls") or []):
                if not isinstance(tc, dict):
                    continue
                idx = int(tc.get("index") or 0)
                b = builders.setdefault(idx, _CallBuilder(index=idx))
                if tc.get("id"):
                    b.id = str(tc["id"])
                fn = tc.get("function") or {}
                if isinstance(fn.get("name"), str) and fn["name"]:
                    b.name += fn["name"]
                frag = fn.get("arguments")
                if isinstance(frag, str) and frag:
                    b.args.append(frag)
                elif isinstance(frag, dict):
                    b.args.append(json.dumps(frag))
                note_first()
                if on_delta:
                    on_delta("tool_call", "")

    calls: list[ToolCall] = []
    for idx in sorted(builders):
        b = builders[idx]
        raw_args = "".join(b.args).strip()
        if raw_args:
            args, raw, err, repaired = _parse_args(raw_args)
        else:
            args, raw, err, repaired = {}, "{}", None, False
        calls.append(ToolCall(id=b.id or "call_" + secrets.token_hex(5), name=b.name,
                              args=args, raw=raw, error=err, repaired=repaired))
    usage = _usage_from_timings(_usage_openai({"usage": usage_raw}), timings)
    truncated = finish in ("length", "max_tokens", "max_output_tokens") or (not finish and bool(calls))
    return ModelReply(text="".join(text), calls=calls[:MAX_TOOL_CALLS_PER_TURN], usage=usage,
                      finish=finish or ("length" if truncated else "stop"),
                      model=route.model, protocol="openai",
                      latency_s=0.0, ttft_s=round(ttft[0], 4), streamed=True,
                      raw_message={"content": "".join(text), "reasoning": "".join(reasoning)})


def _assemble_anthropic_stream(events: Iterator[dict], route: Route, on_delta, t0: float,
                               clock) -> ModelReply:
    blocks: dict[int, dict] = {}
    finish = ""
    usage = {"input": 0, "cache_read": 0, "cache_write": 0, "output": 0, "context": 0,
             "telemetry": True, "reasoning": 0}
    ttft = [0.0]

    def note_first() -> None:
        if not ttft[0]:
            ttft[0] = max(0.0, clock() - t0)

    for ev in events:
        if not isinstance(ev, dict):
            continue
        kind = ev.get("type")
        idx = int(ev.get("index") or 0)
        if kind == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            usage["input"] = int(u.get("input_tokens") or 0)
            usage["cache_read"] = int(u.get("cache_read_input_tokens") or 0)
            usage["cache_write"] = int(u.get("cache_creation_input_tokens") or 0)
        elif kind == "content_block_start":
            blk = ev.get("content_block") or {}
            if blk.get("type") == "tool_use":
                blocks[idx] = {"kind": "tool", "id": str(blk.get("id") or ""),
                               "name": str(blk.get("name") or ""), "json": []}
            else:
                blocks[idx] = {"kind": "text", "text": []}
        elif kind == "content_block_delta":
            d = ev.get("delta") or {}
            b = blocks.setdefault(idx, {"kind": "text", "text": []})
            if d.get("type") == "text_delta":
                piece = d.get("text") or ""
                if piece:
                    note_first()
                    b["text"].append(piece)
                    if on_delta:
                        on_delta("text", piece)
            elif d.get("type") == "thinking_delta":
                piece = d.get("thinking") or ""
                if piece:
                    note_first()
                    if on_delta:
                        on_delta("reasoning", piece)
            elif d.get("type") == "input_json_delta":
                b.setdefault("json", []).append(d.get("partial_json") or "")
                note_first()
                if on_delta:
                    on_delta("tool_call", "")
        elif kind == "message_delta":
            finish = str((ev.get("delta") or {}).get("stop_reason") or finish)
            u = ev.get("usage") or {}
            if u.get("output_tokens"):
                usage["output"] = int(u["output_tokens"])
        elif kind == "error":
            msg = (ev.get("error") or {}).get("message") or "stream error"
            raise ProviderError(f"provider stream error: {msg}", code="server", retryable=True,
                                endpoint=route.base_url)

    text = "".join("".join(b.get("text") or []) for b in blocks.values() if b["kind"] == "text")
    calls: list[ToolCall] = []
    for idx in sorted(blocks):
        b = blocks[idx]
        if b["kind"] != "tool":
            continue
        raw_args = "".join(b.get("json") or []).strip()
        if raw_args:
            args, raw, err, repaired = _parse_args(raw_args)
        else:
            args, raw, err, repaired = {}, "{}", None, False
        calls.append(ToolCall(id=b["id"] or "call_" + secrets.token_hex(5), name=b["name"],
                              args=args, raw=raw, error=err, repaired=repaired))
    usage["context"] = usage["input"] + usage["cache_read"] + usage["cache_write"]
    truncated = finish in ("max_tokens", "length") or (not finish and bool(calls))
    return ModelReply(text=text, calls=calls[:MAX_TOOL_CALLS_PER_TURN], usage=usage,
                      finish=finish or ("max_tokens" if truncated else "end_turn"),
                      model=route.model, protocol="anthropic", latency_s=0.0,
                      ttft_s=round(ttft[0], 4), streamed=True)


def _recover_text_calls(reply: ModelReply, tool_names: set[str], profile: PF.Profile,
                       tools_offered: bool = False) -> ModelReply:
    """A model that wrote a tool call into the text got the call, once, if unambiguous.

    When the endpoint accepts a tools array, only an explicit tool_call block is
    trusted; a fenced JSON object that merely looks like one is left as text.
    """
    if reply.calls or not profile.text_tool_calls or not tool_names:
        return reply
    if tools_offered and "<tool_call>" not in (reply.text or ""):
        # Native tool_calls were requested and not used. Recover only a message
        # that is the call itself. Prose that mentions a call stays prose.
        bare = PF.bare_tool_call(reply.text)
        if bare is None or bare["name"] not in tool_names:
            bare = PF.fenced_tool_call(reply.text, tool_names)
        if bare is None:
            return reply
        reply.calls = [ToolCall(id="call_" + secrets.token_hex(5), name=bare["name"],
                                args=bare["arguments"],
                                raw=json.dumps(bare["arguments"], sort_keys=True), error=None)]
        reply.recovered = True
        return reply
    calls = []
    for item in PF.text_tool_calls(reply.text):
        if item["name"] not in tool_names:
            continue
        calls.append(ToolCall(id="call_" + secrets.token_hex(5), name=item["name"],
                              args=item["arguments"],
                              raw=json.dumps(item["arguments"], sort_keys=True), error=None))
    if calls:
        reply.calls = calls[:MAX_TOOL_CALLS_PER_TURN]
        reply.recovered = True
    return reply


class ModelClient:
    def __init__(self, route: Route, transport: HttpTransport | None = None,
                 clock: Callable[[], float] = time.monotonic):
        self.route = route
        self.transport = transport or HttpTransport()
        self.clock = clock
        self.profile = PF.detect(route.model, route.provider, route.base_url, route.protocol)
        self._no_extras = False
        self.degraded: list[str] = []

    def cache_hint(self, messages: list[dict]) -> str:
        system = ""
        task = ""
        for m in messages:
            role = m.get("role")
            if role == "system" and not system:
                system = str(m.get("content") or "")
            elif role == "user" and not task:
                task = str(m.get("content") or "")
        return self.profile.cache_hint(system, task)

    @staticmethod
    def _tool_names(tools: list[dict] | None) -> set[str]:
        names = set()
        for t in tools or []:
            fn = (t or {}).get("function") or {}
            if fn.get("name"):
                names.add(str(fn["name"]))
        return names

    def _anthropic_payload(self, messages: list[dict], tools: list[dict] | None,
                           max_tokens: int, temperature: float, stream: bool) -> dict:
        route = self.route
        cache = _cache_ok(route.base_url)
        msgs, system = _convert_anthropic_messages(messages, cache)
        payload: dict[str, Any] = {"model": route.model, "max_tokens": int(max_tokens),
                                   "messages": msgs, "temperature": float(temperature)}
        if stream:
            payload["stream"] = True
        if system:
            payload["system"] = ([{"type": "text", "text": system,
                                   "cache_control": dict(_CACHE_BREAKPOINT)}]
                                 if cache else system)
        if tools and self.profile.supports_tools:
            payload["tools"] = [{"name": t["function"]["name"],
                                 "description": t["function"].get("description", ""),
                                 "input_schema": t["function"].get("parameters", {"type": "object"})}
                                for t in tools]
            if cache:
                payload["tools"][-1] = {**payload["tools"][-1],
                                        "cache_control": dict(_CACHE_BREAKPOINT)}
        return payload

    def _send(self, payload: dict, path: str, stream: bool, on_delta, timeout: float, *,
              assembler, normaliser) -> ModelReply:
        route = self.route
        t0 = self.clock()
        resp = self.transport.stream("POST", route.base_url + path, _headers(route),
                                     json.dumps(payload).encode(), timeout)
        try:
            if resp.status >= 400:
                raw = resp.read_all()
                if not self._no_extras and PF.field_rejection(payload, resp.status, raw):
                    self._no_extras = True
                    self.degraded.append("extras:" + str(resp.status))
                    resp.close()
                    return self._send(PF.strip_extras(payload), path, False, on_delta,
                                      timeout, assembler=assembler, normaliser=normaliser)
                raise _parse_http_error(resp.status, raw, route)
            if resp.is_sse(stream):
                reply = assembler(_iter_events(resp.lines()), route, on_delta, t0, self.clock)
            else:
                raw = resp.read_all()
                try:
                    body = json.loads(raw or b"{}")
                except ValueError as e:
                    raise ProviderError("provider returned invalid JSON",
                                        code="bad_response", retryable=True,
                                        endpoint=route.base_url) from e
                reply = normaliser(body, route)
        finally:
            resp.close()
        reply.latency_s = round(self.clock() - t0, 4)
        return reply

    def _request(self, messages: list[dict], tools: list[dict] | None,
                 max_tokens: int, temperature: float, timeout: float,
                 on_delta=None, stream: bool | None = None) -> ModelReply:
        # --no-stream forces a single JSON body. ORDIVYR_LITE_STREAM=0 still
        # wins when the caller left streaming on.
        if stream is False:
            stream_on = False
        else:
            stream_on = PF.streaming_enabled(self.profile)
        stream = stream_on
        if self._no_extras:
            # A previous turn in this run proved the endpoint rejects the
            # family extras, so do not pay for the same rejection again.
            stream = False
        if self.route.protocol == "anthropic":
            payload = self._anthropic_payload(messages, tools, max_tokens, temperature, stream)
            if self._no_extras:
                payload = PF.strip_extras(payload)
            reply = self._send(payload, "/v1/messages", stream, on_delta, timeout,
                               assembler=_assemble_anthropic_stream, normaliser=_norm_anthropic)
        else:
            payload = PF.openai_payload(self.profile, self.route, messages=messages, tools=tools,
                                        max_tokens=max_tokens, temperature=temperature,
                                        stream=stream, cache_hint=self.cache_hint(messages))
            if self._no_extras:
                payload = PF.strip_extras(payload)
            reply = self._send(payload, "/chat/completions", stream, on_delta, timeout,
                               assembler=_assemble_openai_stream, normaliser=_norm_openai)
        return _recover_text_calls(reply, self._tool_names(tools), self.profile,
                                   tools_offered=bool(tools) and self.profile.supports_tools)

    def chat(self, messages: list[dict], tools: list[dict] | None = None, *,
             max_tokens: int = 4096, temperature: float = 0.0,
             timeout: float | None = None, retries: int | None = None,
             deadline: float | None = None, on_delta: Callable[[str, str], None] | None = None,
             stream: bool | None = None) -> ModelReply:
        timeout = float(timeout or self.route.timeout)
        retries = self.route.max_retries if retries is None else retries
        last: ProviderError | None = None
        for attempt in range(retries + 1):
            if deadline is not None and time.monotonic() >= deadline:
                raise ProviderError("time limit reached before the model call", code="timeout",
                                    retryable=False, endpoint=self.route.base_url)
            attempt_timeout = timeout if deadline is None else min(timeout, max(0.001, deadline - time.monotonic()))
            try:
                return self._request(messages, tools, max_tokens, temperature, attempt_timeout,
                                     on_delta=on_delta, stream=stream)
            except ProviderError as e:
                last = e
                if not e.retryable or attempt >= retries:
                    raise
                wait = min(4.0, 0.5 * (2 ** attempt))
                if deadline is not None:
                    wait = max(0.0, min(wait, deadline - time.monotonic()))
                    if wait <= 0:
                        raise
                time.sleep(wait)
        raise last or ProviderError("provider call failed", code="provider_error")

    def list_models(self, timeout: float = 20.0) -> list[str]:
        route = self.route
        path = "/v1/models" if route.protocol == "anthropic" else "/models"
        j = _call(self.transport, route, "GET", path, None, timeout)
        items = j.get("data") or j.get("models") or []
        names = []
        for m in items if isinstance(items, list) else []:
            if isinstance(m, dict):
                name = m.get("id") or m.get("name") or ""
                names.append(str(name).removeprefix("models/"))
        return sorted(n for n in names if n)


def probe_local(base_url: str, timeout: float = 3.0) -> dict:
    """Cheap reachability probe for doctor; never sends a model request."""
    import urllib.request

    base = base_url.rstrip("/")
    out: dict[str, Any] = {"base_url": base_url, "reachable": False}
    for path in ("/models", "/v1/models"):
        try:
            with urllib.request.urlopen(base + path, timeout=timeout) as r:
                out["reachable"] = True
                out["status"] = r.status
                try:
                    body = json.loads(r.read() or b"{}")
                    data = body.get("data") or body.get("models") or []
                    out["models"] = [m.get("id") or m.get("name") for m in data if isinstance(m, dict)][:20]
                except ValueError:
                    out["models"] = []
                return out
        except Exception as exc:  # noqa: BLE001 - probe is best effort
            out["error"] = str(exc)[:120]
    return out
