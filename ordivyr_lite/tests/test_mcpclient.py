"""Stdio MCP against a tiny in-process protocol server. No network, no npx."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from ordivyr_lite.mcpclient import McpClient, McpError, attach_config, exposed_name
from ordivyr_lite.tools import ToolBox

SERVER = textwrap.dedent(r'''
    import json, sys

    def read_msg():
        headers = {}
        while True:
            line = sys.stdin.buffer.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            if b":" in line:
                k, v = line.decode("ascii", "replace").split(":", 1)
                headers[k.strip().lower()] = v.strip()
        n = int(headers.get("content-length") or 0)
        if n <= 0:
            return None
        return json.loads(sys.stdin.buffer.read(n).decode())

    def write_msg(obj):
        raw = json.dumps(obj).encode()
        sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode() + raw)
        sys.stdout.buffer.flush()

    while True:
        msg = read_msg()
        if not msg:
            break
        if "id" not in msg:
            continue
        method = msg.get("method")
        if method == "initialize":
            write_msg({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "protocolVersion": "2024-11-05", "capabilities": {"tools": {}},
                "serverInfo": {"name": "echo", "version": "0"}}})
        elif method == "tools/list":
            write_msg({"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": [
                {"name": "echo", "description": "Echo text",
                 "inputSchema": {"type": "object",
                                 "properties": {"text": {"type": "string"}},
                                 "required": ["text"]},
                 "annotations": {"readOnlyHint": True}},
                {"name": "write_secret", "description": "A writer.",
                 "inputSchema": {"type": "object", "properties": {}},
                 "annotations": {"readOnlyHint": False}},
            ]}})
        elif method == "tools/call":
            args = (msg.get("params") or {}).get("arguments") or {}
            write_msg({"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": "echo:" + str(args.get("text", ""))}],
                "isError": False}})
        elif method == "ping":
            write_msg({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "nope"}})
        else:
            write_msg({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "nope"}})
''')


def _config(tmp_path: Path, script: Path) -> Path:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({
        "servers": {
            "echo": {
                "command": "python3",
                "args": [str(script)],
                "env": {"MCP_SECRET": "super-secret-value"},
            }
        }
    }), "utf-8")
    return path


def test_stdio_server_round_trip_and_read_only_filter(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    box = ToolBox(tmp_path, profile="read-only")
    try:
        names = attach_config(box, _config(tmp_path, script))
        assert names == [exposed_name("echo", "echo")]
        assert exposed_name("echo", "write_secret") not in box.names()
        out = box.call(names[0], {"text": "hi"})
        assert out == "echo:hi"
    finally:
        box.close()


def test_workspace_receives_the_writer(tmp_path):
    script = tmp_path / "server.py"
    script.write_text(SERVER)
    box = ToolBox(tmp_path, profile="workspace")
    try:
        names = attach_config(box, _config(tmp_path, script))
        assert exposed_name("echo", "write_secret") in names
    finally:
        box.close()


def test_error_text_does_not_include_the_server_env(tmp_path):
    client = McpClient(name="missing", command=["/no/such/mcp-server-binary"],
                       env={"MCP_SECRET": "super-secret-value"})
    with pytest.raises(McpError) as exc:
        client.start()
    assert "super-secret-value" not in str(exc.value)


def test_empty_config_is_refused(tmp_path):
    path = tmp_path / "mcp.json"
    path.write_text('{"servers": {}}')
    box = ToolBox(tmp_path, profile="workspace")
    with pytest.raises(McpError):
        attach_config(box, path)
