"""Minimal MCP client (stdio, JSON-RPC, Content-Length or newline JSON).

Servers come from an operator config file. They are not started for a normal
run. Tool results are untrusted data. This client does not sandbox the server
process: the operator chose the command. A read-only profile receives only
tools the server marks readOnlyHint.
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

PROTOCOL = "2024-11-05"
MAX_TOOLS = 20
MAX_DESC = 300
MAX_SCHEMA = 2000
MAX_OUTPUT = 8000


class McpError(Exception):
    pass


@dataclass
class McpTool:
    name: str
    description: str
    schema: dict
    read_only: bool = False


@dataclass
class McpClient:
    name: str
    command: list[str]
    env: dict[str, str] = field(default_factory=dict)
    cwd: str | None = None
    timeout: float = 20.0

    def __post_init__(self) -> None:
        self.proc: subprocess.Popen | None = None
        self._inbox: queue.Queue = queue.Queue()
        self._id = 0
        self._reader: threading.Thread | None = None
        self._write_lock = threading.Lock()

    def start(self) -> None:
        if self.proc is not None:
            return
        if not self.command or not self.command[0]:
            raise McpError(f"{self.name}: command is empty")
        child_env = os.environ.copy()
        # Config env is passed to the server and never copied into errors.
        for key, value in self.env.items():
            child_env[str(key)] = str(value)
        try:
            self.proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env=child_env, cwd=self.cwd, bufsize=0)
        except OSError as exc:
            raise McpError(f"{self.name}: cannot start server ({exc.errno})") from exc
        self._reader = threading.Thread(target=self._read_loop, name=f"mcp-{self.name}", daemon=True)
        self._reader.start()
        self.request("initialize", {
            "protocolVersion": PROTOCOL,
            "capabilities": {},
            "clientInfo": {"name": "ordivyr-lite", "version": "0.2.0"},
        })
        self.notify("notifications/initialized")

    def list_tools(self) -> list[McpTool]:
        result = self.request("tools/list", {}) or {}
        raw = result.get("tools") if isinstance(result, dict) else None
        tools: list[McpTool] = []
        for item in raw or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            if len(tools) >= MAX_TOOLS:
                break
            schema = item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {"type": "object"}
            encoded = json.dumps(schema, separators=(",", ":"))
            if len(encoded) > MAX_SCHEMA:
                raise McpError(f"{self.name}: tool schema exceeds {MAX_SCHEMA} characters")
            annotations = item.get("annotations") if isinstance(item.get("annotations"), dict) else {}
            tools.append(McpTool(
                name=str(item["name"]),
                description=str(item.get("description") or "")[:MAX_DESC],
                schema=schema,
                read_only=annotations.get("readOnlyHint") is True,
            ))
        return tools

    def call(self, name: str, arguments: dict) -> str:
        result = self.request("tools/call", {"name": name, "arguments": arguments or {}}) or {}
        if not isinstance(result, dict):
            return "ERROR: mcp returned a non-object result"
        parts = []
        for block in result.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "\n".join(parts)[:MAX_OUTPUT]
        if result.get("isError"):
            return "ERROR: " + (text or "mcp tool failed")
        return text

    def request(self, method: str, params: dict) -> dict:
        if self.proc is None:
            self.start()
        self._id += 1
        msg_id = self._id
        self._write({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            try:
                msg = self._inbox.get(timeout=0.1)
            except queue.Empty:
                if self.proc is not None and self.proc.poll() is not None:
                    raise McpError(f"{self.name}: server exited")
                continue
            if msg is None:
                raise McpError(f"{self.name}: server closed the pipe")
            if not isinstance(msg, dict):
                continue
            if msg.get("id") == msg_id and "method" not in msg:
                if "error" in msg:
                    err = msg["error"]
                    message = err.get("message") if isinstance(err, dict) else "rpc error"
                    raise McpError(f"{self.name}: {str(message)[:200]}")
                result = msg.get("result")
                return result if isinstance(result, dict) else {}
            if "method" in msg and "id" in msg:
                self._write({"jsonrpc": "2.0", "id": msg["id"],
                             "error": {"code": -32601, "message": "unsupported"}})
        raise McpError(f"{self.name}: timed out waiting for {method}")

    def notify(self, method: str) -> None:
        self._write({"jsonrpc": "2.0", "method": method})

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        if proc.stdin:
            try:
                proc.stdin.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                proc.kill()

    def _write(self, payload: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise McpError(f"{self.name}: server is not running")
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        frame = f"Content-Length: {len(raw)}\r\n\r\n".encode("ascii") + raw
        with self._write_lock:
            try:
                proc.stdin.write(frame)
                proc.stdin.flush()
            except OSError as exc:
                raise McpError(f"{self.name}: write failed") from exc

    def _read_loop(self) -> None:
        proc = self.proc
        if proc is None or proc.stdout is None:
            self._inbox.put(None)
            return
        buf = b""
        try:
            while True:
                chunk = proc.stdout.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    msg, nxt = _take_message(buf)
                    if msg is None and nxt == buf:
                        break
                    buf = nxt
                    if msg is not None:
                        self._inbox.put(msg)
        except Exception:  # noqa: BLE001 - the reader must not kill the process
            pass
        self._inbox.put(None)


def _take_message(buf: bytes):
    data = buf.lstrip(b"\r\n")
    if not data:
        return None, b""
    if data.lower().startswith(b"content-length:"):
        sep = data.find(b"\r\n\r\n")
        if sep < 0:
            return None, data
        header = data[:sep].decode("ascii", "replace")
        length = None
        for line in header.split("\r\n"):
            if line.lower().startswith("content-length:"):
                try:
                    length = int(line.split(":", 1)[1].strip())
                except ValueError:
                    return None, b""
        if length is None:
            return None, data[sep + 4:]
        start = sep + 4
        if len(data) < start + length:
            return None, data
        raw = data[start:start + length]
        try:
            return json.loads(raw.decode("utf-8")), data[start + length:]
        except ValueError:
            return None, data[start + length:]
    if data[:1] == b"{":
        try:
            text = data.decode("utf-8")
            obj, idx = json.JSONDecoder().raw_decode(text)
        except (UnicodeDecodeError, ValueError):
            return None, data
        return obj, text[idx:].encode("utf-8")
    # Drop a junk line so a banner cannot stall the reader.
    nl = data.find(b"\n")
    if nl < 0:
        return None, data
    return None, data[nl + 1:]


def exposed_name(server: str, tool: str) -> str:
    raw = f"mcp_{_slug(server)}_{_slug(tool)}"
    # Preserve compact canonical names; append identity when sanitizing loses
    # information. The digest includes full, untruncated server/tool names.
    if raw == f"mcp_{server}_{tool}" and len(raw) <= 64 and "_" not in server and "_" not in tool:
        return raw
    identity = hashlib.sha256(json.dumps([server, tool]).encode()).hexdigest()[:16]
    return raw[:47] + "_" + identity


def _slug(text: str) -> str:
    out = []
    for ch in str(text).lower():
        out.append(ch if ch.isalnum() else "_")
    slug = "".join(out).strip("_") or "tool"
    return slug[:24]


def attach_client(box, client: McpClient) -> list[str]:
    """Register the server's tools on a toolbox. Read-only profiles skip writers."""
    client.start()
    names: list[str] = []
    try:
        tools = client.list_tools()
    except Exception:
        client.close()
        raise
    box._closers.append(client.close)
    for tool in tools:
        if box.profile == "read-only" and not tool.read_only:
            continue
        name = exposed_name(client.name, tool.name)
        description = tool.description or f"MCP tool {tool.name} from {client.name}"

        def handler(args, c=client, tool_name=tool.name):
            if not isinstance(args, dict):
                return "ERROR: arguments must be an object"
            try:
                return c.call(tool_name, args)
            except McpError as exc:
                return "ERROR: " + str(exc)[:200]

        box.add_tool(name, description, tool.schema, handler,
                     kind="read" if tool.read_only else "write")
        names.append(name)
    return names


def attach_config(box, path: Path) -> list[str]:
    try:
        raw = json.loads(Path(path).read_text("utf-8"))
    except (OSError, ValueError) as exc:
        raise McpError(f"cannot read mcp config: {exc}") from exc
    servers = raw.get("servers") if isinstance(raw, dict) else None
    if not isinstance(servers, dict) or not servers:
        raise McpError("mcp config needs a servers object")
    attached: list[str] = []
    started: list[McpClient] = []
    try:
        for name, spec in servers.items():
            if not isinstance(spec, dict):
                raise McpError(f"{name}: server spec must be an object")
            command = spec.get("command")
            args = spec.get("args") or []
            if isinstance(command, list):
                argv = [str(x) for x in command]
            elif isinstance(command, str) and command:
                argv = [command, *[str(x) for x in args]]
            else:
                raise McpError(f"{name}: command is required")
            env = spec.get("env") or {}
            if not isinstance(env, dict):
                raise McpError(f"{name}: env must be an object")
            client = McpClient(name=str(name), command=argv,
                               env={str(k): str(v) for k, v in env.items()},
                               cwd=str(spec["cwd"]) if spec.get("cwd") else None,
                               timeout=float(spec.get("timeout") or 20))
            started.append(client)
            attached.extend(attach_client(box, client))
    except Exception:
        for client in started:
            client.close()
        raise
    return attached
