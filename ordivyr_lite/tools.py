"""The tool surface offered to the model, with permission profiles and a
confined working directory.

Profiles:
  read-only   list/read/grep/find only
  workspace   read-only + write_file/edit_file inside the workdir (default)
  trusted     workspace + sandboxed run_command + fetch_url
  full        trusted (run_command still sandboxed unless the operator opts out)

A tool is never offered just because the model asks for it: the profile and an
optional allow-list decide, and every call is checked again at execution time.
"""

from __future__ import annotations

from copy import deepcopy

import http.client
import ipaddress
import os
import re
import socket
import ssl
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import filetools as FT
from . import sandbox as SB

MAX_RESULT = 8_000
MEASURE = time.monotonic
READ_ONLY_NAMES = frozenset({"list_files", "read_file", "grep", "find_files", "fetch_url"})

PROFILES: dict[str, dict[str, bool]] = {
    "read-only": {"write": False, "shell": False, "web": False},
    "workspace": {"write": True, "shell": False, "web": False},
    "trusted": {"write": True, "shell": True, "web": True},
    "full": {"write": True, "shell": True, "web": True},
}
PERMISSION_NAMES = tuple(PROFILES)


class ToolError(Exception):
    pass


@dataclass
class ToolDef:
    name: str
    description: str
    schema: dict
    kind: str = "read"


@dataclass
class ToolStats:
    calls: int = 0
    errors: int = 0
    duration_ms: float = 0.0
    by_name: dict[str, dict] = field(default_factory=dict)

    def add(self, name: str, ok: bool, duration_ms: float) -> None:
        self.calls += 1
        self.duration_ms += duration_ms
        self.by_name.setdefault(name, {"calls": 0, "errors": 0, "duration_ms": 0.0})
        bucket = self.by_name[name]
        bucket["calls"] += 1
        bucket["duration_ms"] += duration_ms
        if not ok:
            self.errors += 1
            bucket["errors"] += 1

    @property
    def success_rate(self) -> float:
        return round(1 - (self.errors / self.calls), 4) if self.calls else 0.0

    def as_dict(self) -> dict:
        return {"calls": self.calls, "errors": self.errors,
                "success_rate": self.success_rate,
                "duration_ms": round(self.duration_ms, 1),
                "by_name": {k: {**v, "duration_ms": round(v["duration_ms"], 1)}
                            for k, v in self.by_name.items()}}


def _schema(props: dict[str, tuple[str, str]], required: list[str]) -> dict:
    return {"type": "object",
            "properties": {k: {"type": t, "description": d} for k, (t, d) in props.items()},
            "required": required}


ALL_TOOL_NAMES = frozenset({"list_files", "read_file", "grep", "find_files", "edit_file",
                            "write_file", "run_command", "fetch_url"})
_ACCEPTS = {
    "integer": "whole number such as 50",
    "number": "number such as 0.5",
    "boolean": "true or false",
    "string": "text",
}
_INT_TEXT = re.compile(r"[+-]?\d+\Z")


def _short(value: Any, limit: int = 40) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[:limit] + "..."


def _coerce(kind: str, value: Any) -> tuple[bool, Any]:
    """-> (accepted, coerced). Only an unambiguous conversion is accepted."""
    if kind not in _ACCEPTS or value is None:
        return True, value
    try:
        if kind == "integer":
            if isinstance(value, bool):
                return False, value
            if isinstance(value, int):
                return True, value
            if isinstance(value, float):
                return (True, int(value)) if value.is_integer() else (False, value)
            if isinstance(value, str) and _INT_TEXT.match(value.strip()):
                return True, int(value.strip())
            return False, value
        if kind == "number":
            if isinstance(value, bool):
                return False, value
            if isinstance(value, (int, float)):
                return True, value
            if isinstance(value, str):
                number = float(value.strip())
                finite = number == number and abs(number) != float("inf")
                return (True, number) if finite else (False, value)
            return False, value
        if kind == "boolean":
            if isinstance(value, bool):
                return True, value
            if isinstance(value, int) and value in (0, 1):
                return True, bool(value)
            if isinstance(value, str) and value in ("true", "false"):
                return True, value == "true"
            return False, value
        if isinstance(value, str):
            return True, value
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return True, str(value)
        return False, value
    except (TypeError, ValueError):
        return False, value


def _parameter_list(d: ToolDef) -> str:
    required = set(d.schema.get("required") or [])
    rows = [f"{k} ({v.get('type')}, {'required' if k in required else 'optional'})"
            for k, v in (d.schema.get("properties") or {}).items()]
    return ", ".join(rows) or "none"


def _command_text(res: Any, timeout: int) -> str:
    """Model-facing text for one sandbox result.

    A command that ran and exited non-zero is a FAILED tool call: loop.py reads
    ok from the "ERROR:" prefix, so a failing build or test must not look like
    a success.  SandboxResult itself is untouched, so every other reader of
    rc/timed_out keeps its view.
    """
    code = getattr(res, "exit_code", 0)
    body = (getattr(res, "stdout", "") or "") + (getattr(res, "stderr", "") or "")
    if getattr(res, "timed_out", False):
        # The sandbox and the timed_out flag stay visible for the operator,
        # after the salvaged output rather than in front of it.
        trailer = f"[timed_out=True sandbox={getattr(res, 'sandbox', 'unknown')}]"
        if body and not body.endswith("\n"):
            body += "\n"
        return f"ERROR: command timed out after {timeout}s (exit=-9)\n{body}{trailer}"
    if code != 0:
        shown = "unknown" if code is None else code
        return f"ERROR: command failed (exit={shown})\n{body}"
    return res.text()


class ToolBox:
    def __init__(self, root: str | os.PathLike, *, profile: str = "workspace",
                 allow_tools: list[str] | None = None, sandbox_mode: str = "auto",
                 network: bool = False, timeout: int = 60,
                 allow_shell: bool | None = None, allow_web: bool | None = None,
                 plan: bool = False):
        self.plan = bool(plan)
        self.plan_hook = None
        self._handlers: dict[str, Any] = {}
        self._closers: list[Any] = []
        self.root = Path(os.path.realpath(root))
        if not self.root.is_dir():
            raise ToolError(f"working directory does not exist: {root}")
        if profile not in PROFILES:
            raise ToolError(f"unknown permission profile {profile!r}; choose {', '.join(PERMISSION_NAMES)}")
        base = PROFILES[profile]
        self.profile = profile
        self.write = base["write"]
        if profile == "read-only" and (allow_shell is True or allow_web is True):
            raise ToolError("read-only profile cannot enable shell or web; select workspace for a scoped grant")
        # None inherits the profile. Explicit booleans are deliberate operator
        # grants or denials; shell and web stay independent capabilities.
        self.shell = base["shell"] if allow_shell is None else bool(allow_shell)
        self.web = base["web"] if allow_web is None else bool(allow_web)
        self._allow_shell_request = allow_shell
        self._allow_web_request = allow_web
        self.sandbox_mode = sandbox_mode
        self.network = bool(network)
        self.timeout = int(timeout or 60)
        self.allow = set(allow_tools) if allow_tools is not None else None
        if self.allow is not None:
            unknown_tools = self.allow - ALL_TOOL_NAMES
            if unknown_tools:
                raise ToolError("unknown tool name(s) in allow-list: " + ", ".join(sorted(unknown_tools)))
            if "write_file" in self.allow:
                self.allow.add("edit_file")
            if "read_file" in self.allow:
                self.allow |= {"grep", "find_files"}
        self._spill: Path | None = None
        self._sandbox = None
        self._sandbox_lock = threading.Lock()
        self._nspill = 0
        self._spill_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._observed_files: dict[str, str] = {}
        self.task_constraints = None
        self.stats = ToolStats()
        self.defs = self._build_defs()

    # ------------------------------------------------------------------ definitions
    def _d(self, name: str, desc: str, props: dict[str, tuple[str, str]],
           required: list[str], kind: str) -> ToolDef:
        return ToolDef(name, desc, _schema(props, required), kind)

    def _build_defs(self) -> list[ToolDef]:
        defs = [
            self._d("list_files", "List names in one folder.",
                    {"path": ("string", "folder, default '.'")}, [], "read"),
            self._d("read_file",
                    f"Read a text file with line numbers, up to {FT.READ_DEFAULT_LINES} from offset. "
                    "A truncation notice gives the next offset. Path 'spill:N' reads a saved long output.",
                    {"path": ("string", "workspace path"),
                     "offset": ("integer", "first line, 1-based"),
                     "limit": ("integer", f"max lines, default {FT.READ_DEFAULT_LINES}, max {FT.READ_MAX_LINES}")},
                    ["path"], "read"),
            self._d("grep",
                    "Search file text. Literal unless regex=true. Skips .git, dependencies, and binaries. "
                    "Returns path:line: text.",
                    {"pattern": ("string", "text or regex"),
                     "path": ("string", "folder or file, default '.'"),
                     "glob": ("string", "name filter such as *.py"),
                     "regex": ("boolean", "pattern is a regex"),
                     "ignore_case": ("boolean", "case-insensitive")}, ["pattern"], "read"),
            self._d("find_files", "Find files by name glob, for example '*.py'.",
                    {"glob": ("string", "name or path glob"),
                     "path": ("string", "folder, default '.'")}, [], "read"),
        ]
        if self.write:
            defs.append(self._d(
                "edit_file",
                "Replace exact text. old_string must match once unless replace_all=true. "
                "Use this instead of rewriting a whole file.",
                {"path": ("string", "workspace path"),
                 "old_string": ("string", "exact text to replace"),
                 "new_string": ("string", "replacement"),
                 "replace_all": ("boolean", "replace every match"),
                 "expected_sha256": ("string", "optional SHA256 of exact file bytes; stale files are refused")},
                ["path", "old_string", "new_string"], "write"))
            defs.append(self._d(
                "write_file",
                "Create a file or overwrite it completely. Use edit_file to change part of a file.",
                {"path": ("string", "workspace path"),
                 "content": ("string", "full file content"),
                 "expected_sha256": ("string", "optional SHA256 of existing file bytes; stale content is refused")}, ["path", "content"], "write"))
        if self.shell:
            defs.append(self._d(
                "run_command",
                "Run one sandboxed shell command. The workspace is writable. Network is off unless enabled. "
                "Do not use this to read files.",
                {"command": ("string", "one command, max 8000 chars"),
                 "timeout": ("integer", f"seconds, default {self.timeout}, max {SB.MAX_TIMEOUT}")},
                ["command"], "shell"))
        if self.web:
            defs.append(self._d("fetch_url", "GET a public http(s) URL as text.",
                                {"url": ("string", "http(s) URL")}, ["url"], "web"))
        if self.allow is not None:
            defs = [d for d in defs if d.name in self.allow]
        if self.plan:
            defs.append(self._d(
                "update_plan",
                "Track the task plan. add: one milestone per line. done / doing: comma-separated milestone "
                "ids. Keep milestones few and checkable; mark one done only when it is verified.",
                {"add": ("string", "new milestones, one per line"),
                 "done": ("string", "ids now complete, e.g. '1,2'"),
                 "doing": ("string", "ids in progress")}, [], "read"))
        return defs

    def names(self) -> list[str]:
        return [d.name for d in self.defs]

    def permission_manifest(self) -> dict:
        return {"profile": self.profile, "shell": self.shell, "web": self.web,
                "network": self.network, "sandbox": self.sandbox_mode,
                "tools": sorted(self.allow) if self.allow is not None else None}

    def unavailable_reason(self, name: str) -> str | None:
        if name not in ALL_TOOL_NAMES:
            return f"unknown tool; registered tools: {', '.join(self.names()) or 'none'}"
        if name in self.names():
            return None
        if name in ("write_file", "edit_file") and not self.write:
            return "disabled by the read-only permission profile"
        if name == "run_command" and not self.shell:
            if self._allow_shell_request is False:
                return "explicitly disabled by --no-shell or permissions.allow_shell=false"
            return ("shell is disabled by the effective policy; use --permission trusted or "
                    "explicitly authorize a scoped shell with --permission workspace --allow-shell")
        if name == "fetch_url" and not self.web:
            if self._allow_web_request is False:
                return "explicitly disabled by --no-web or permissions.allow_web=false"
            return ("web access is disabled by the effective policy; use --permission trusted or "
                    "explicitly authorize it with --permission workspace --allow-web")
        if self.allow is not None and name not in self.allow:
            return "excluded by the explicit --tools allow-list"
        return f"not registered under the {self.profile} profile"

    def permission_report(self) -> dict:
        unavailable = {name: self.unavailable_reason(name)
                       for name in sorted(ALL_TOOL_NAMES) if not self.has(name)}
        return {"profile": self.profile, "shell": self.shell, "web": self.web,
                "network": self.network, "sandbox": self.sandbox_mode,
                "available": self.names(), "unavailable": unavailable}

    def permission_summary(self) -> str:
        return (f"Effective tools ({self.profile}; shell={'enabled' if self.shell else 'disabled'}, "
                f"web={'enabled' if self.web else 'disabled'}, network={'enabled' if self.network else 'disabled'}, "
                f"sandbox={self.sandbox_mode}): {', '.join(self.names()) or 'none'}")

    def permission_refresh_message(self) -> str:
        guidance = self.permission_guidance()
        if guidance:
            return "Session resumed. " + guidance + " The current invocation policy is enforced."
        return "Session resumed. The current invocation policy is enforced."

    def available_names(self) -> list[str]:
        """Advertise only tools that can satisfy the current task constraints."""
        names = self.names()
        policy = getattr(self, 'task_constraints', None)
        if policy is not None and policy.files:
            from .constraints import READS, WRITES
            names = [name for name in names if name in READS or name in WRITES
                     or name == 'run_command' and self.readonly_commands_available()]
        return names

    def readonly_commands_available(self) -> bool:
        return self.shell and self.has('run_command') and SB.bwrap_path() is not None

    def permission_guidance(self, names: list[str] | None = None) -> str:
        """One line for the model. Refusal reasons stay on the tool result, not the prompt."""
        available = self.available_names()
        names = available if names is None else [name for name in names if name in available]
        if not names:
            return ""
        return "Tools enabled: " + ", ".join(names) + ". Any other tool call is refused."

    def add_tool(self, name: str, description: str, schema: dict, handler, *, kind: str = "write") -> None:
        """Register an extra tool. Replacing the handler keeps chat turns on the current run."""
        if not name or not str(name).replace("_", "").isalnum():
            raise ToolError(f"invalid tool name {name!r}")
        if name in ALL_TOOL_NAMES:
            raise ToolError(f"cannot replace built-in tool {name!r}")
        definition = ToolDef(str(name), str(description), deepcopy(schema or {"type": "object"}), kind)
        for i, d in enumerate(self.defs):
            if d.name == name:
                if d.kind != kind:
                    raise ToolError(f"cannot change tool kind for {name!r}")
                self.defs[i] = definition
                break
        else:
            self.defs.append(definition)
        self._handlers[str(name)] = handler

    def has(self, name: str) -> bool:
        return any(d.name == name for d in self.defs)

    def spec(self) -> list[dict]:
        return [{"type": "function", "function": {"name": d.name, "description": d.description,
                                                  "parameters": d.schema}} for d in self.defs]

    def request_spec(self, names: list[str] | None = None) -> list[dict]:
        """The payload sent to the model.

        Function descriptions and required-parameter names stay. Per-property
        descriptions repeat those names, so they are not sent. spec() keeps
        them for callers that inspect the full contract.
        """
        wanted = None if names is None else set(names)
        out = []
        for d in self.defs:
            if wanted is not None and d.name not in wanted:
                continue
            if d.name in self._handlers:
                out.append({"type": "function", "function": {
                    "name": d.name, "description": d.description,
                    "parameters": deepcopy(d.schema)}})
                continue
            props = {}
            for key, prop in (d.schema.get("properties") or {}).items():
                if isinstance(prop, dict) and prop.get("type"):
                    props[key] = {"type": prop["type"]}
                else:
                    props[key] = {"type": "string"}
            params: dict = {"type": "object", "properties": props}
            required = list(d.schema.get("required") or [])
            if required:
                params["required"] = required
            description = d.description
            if d.name == 'run_command' and self.task_constraints is not None and self.task_constraints.files:
                description = ('Run diagnostics or tests with the ENTIRE workspace READ-ONLY and network disabled. '
                               'Use file tools for edits; shell writes cannot change the workspace.')
            out.append({"type": "function", "function": {
                "name": d.name, "description": description, "parameters": params}})
        return out

    def describe(self) -> list[dict]:
        return [{"name": d.name, "description": d.description, "kind": d.kind,
                 "schema": d.schema} for d in self.defs]

    # ------------------------------------------------------------------ validation
    def validate(self, name: str, args: Any) -> str | None:
        """-> None if this call may run, else a short refusal reason.

        A scalar parameter of a convertible kind is coerced IN PLACE before the
        type check, so the very dict the caller executes carries the converted
        value ("50" -> 50 for an integer).  Pass the dict you intend to run.
        Every refusal names what the tool accepts.
        """
        if not isinstance(args, dict):
            return "arguments must be a JSON object"
        known = {d.name: d for d in self.defs}
        offered = ", ".join(d.name for d in self.defs) or "none"
        d = known.get(name)
        if d is None:
            if name in ALL_TOOL_NAMES:
                return (f"tool {name!r} is not available in the {self.profile} permission profile; "
                        f"reason: {self.unavailable_reason(name)}; this profile offers: {offered}")
            return f"unknown tool {name!r}; available tools: {offered}"
        if name in self._handlers:
            from .schema import check
            try:
                return check(d.schema, args)
            except (ValueError, TypeError, RecursionError, OverflowError):
                return "tool schema cannot be validated locally"
        required = d.schema.get("required") or []
        missing = [k for k in required if args.get(k) is None]
        if missing:
            return ("missing required parameter(s): " + ", ".join(missing)
                    + "; this tool accepts: " + _parameter_list(d))
        for key, value in args.items():
            prop = d.schema.get("properties", {}).get(key)
            if prop is None or value is None:
                continue
            expected = prop.get("type")
            accepted, coerced = _coerce(expected, value)
            if not accepted:
                return (f"parameter {key!r} must be {expected} "
                        f"({_ACCEPTS.get(expected, expected)}); got {_short(value)}")
            args[key] = coerced
        if name == "run_command" and not str(args.get("command") or "").strip():
            return "command must not be empty"
        if name in ("write_file", "edit_file", "list_files", "read_file", "grep", "find_files"):
            path = args.get("path")
            if path is not None and (not isinstance(path, str) or "\0" in path):
                return "path must be a valid workspace string"
        return None

    # ------------------------------------------------------------------ execution
    def call(self, name: str, args: dict) -> str:
        t0 = MEASURE()
        ok = False
        try:
            error = self.validate(name, args)
            if error:
                raise ToolError(error)
            if self.task_constraints is not None:
                refusal = self.task_constraints.refusal(self, name, args)
                if refusal:
                    raise ToolError(refusal)
            if name in ('write_file', 'edit_file'):
                path = self._path(args.get('path'), must_exist=False)
                expected = args.get('expected_sha256') or self._observed_files.get(str(path))
                if expected is not None:
                    import hashlib
                    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        raise ToolError('stale file identity: file changed since it was read; read it again before editing')
                args = {**args, **({'expected_sha256': expected} if expected else {})}
            out = self._call(name, args)
            ok = not (isinstance(out, str) and out.startswith("ERROR:"))
        except (FT.FileToolError, ToolError) as exc:
            out, ok = f"ERROR: {exc}", False
        except Exception as exc:  # noqa: BLE001 - a tool must never crash the run
            out, ok = f"ERROR: {type(exc).__name__}: {str(exc)[:200]}", False
        finally:
            with self._stats_lock:
                self.stats.add(name, ok=ok, duration_ms=(MEASURE() - t0) * 1000)
        if name != "read_file":
            out = FT.retain(self, out, MAX_RESULT)
        return out

    def _call(self, name: str, args: dict) -> str:
        handler = self._handlers.get(name)
        if handler is not None:
            out = handler(args)
            return out if isinstance(out, str) else str(out)
        if name in ("read_file", "grep", "find_files"):
            fn = {"read_file": FT.read_file, "grep": FT.grep, "find_files": FT.find_files}[name]
            return fn(self, args)
        if name == "edit_file":
            if not self.write:
                raise ToolError("not allowed: this agent has no file-write permission")
            return FT.edit_file(self, args)
        if name == "update_plan":
            if self.plan_hook is None:
                raise ToolError("no plan is being tracked in this session")
            return self.plan_hook(str(args.get("add") or ""), str(args.get("done") or ""),
                                  str(args.get("doing") or ""))
        if name == "list_files":
            p = self._path(args.get("path") or ".")
            names = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir()
                           if x.name != ".git")[:300]
            return "\n".join(names) or "(empty)"
        if name == "write_file":
            if not self.write:
                raise ToolError("not allowed: this agent has no file-write permission")
            p = self._path(args.get("path"), must_exist=False)
            content = str(args.get("content", ""))
            if len(content) > 400_000:
                raise ToolError("file too large (max 400,000 chars)")
            rel = p.relative_to(self.root)
            if p.name == ".git" or ".git" in rel.parts:
                raise ToolError("not allowed")
            p.parent.mkdir(parents=True, exist_ok=True)
            FT.atomic_write(p, content, expected_sha256=args.get('expected_sha256'))
            import hashlib
            self._observed_files[str(p)] = hashlib.sha256(content.encode('utf-8')).hexdigest()
            return f"wrote {len(content)} characters to {rel}"
        if name == "run_command":
            if not self.shell:
                raise ToolError("not allowed: this agent has no shell permission")
            timeout = max(1, min(int(args.get("timeout") or self.timeout), SB.MAX_TIMEOUT))
            if self.task_constraints is not None and self.task_constraints.files:
                res = self.sandbox_run(str(args.get("command") or ""), timeout, readonly_workspace=True)
            else:
                res = self.sandbox_run(str(args.get("command") or ""), timeout)
            return _command_text(res, timeout)
        if name == "fetch_url":
            if not self.web:
                raise ToolError("not allowed: this agent has no web permission")
            return fetch_url(str(args.get("url", "")))
        raise ToolError("unknown tool")

    def sandbox_run(self, command: str, timeout: int, *, readonly_workspace: bool = False):
        """Run one command, reusing a warm sandbox session when the operator allows it.

        The session keeps the exact one-shot confinement; it only removes the
        bwrap start-up cost per command.  Any failure falls back to one-shot.
        """
        if readonly_workspace:
            # Never reuse a previously writable session, even when the operator
            # otherwise selected process/none. Protection requires this stronger boundary.
            return SB.run(command, self.root, timeout=timeout, network=False,
                          mode='bwrap', readonly_workspace=True)
        if self.sandbox_mode not in ("auto", "bwrap"):
            return SB.run(command, self.root, timeout=timeout, network=self.network,
                          mode=self.sandbox_mode)
        with self._sandbox_lock:
            if self._sandbox is None:
                try:
                    self._sandbox = SB.session(self.root, network=self.network, mode="auto")
                except Exception:  # noqa: BLE001 - fall back to one-shot execution
                    self._sandbox = False  # type: ignore[assignment]
            sess = self._sandbox
        if not sess:
            return SB.run(command, self.root, timeout=timeout, network=self.network,
                          mode=self.sandbox_mode)
        return sess.run(command, timeout=timeout)

    # ------------------------------------------------------------------ paths / spill
    def _path(self, rel, must_exist: bool = True) -> Path:
        rel = str(rel or ".")
        if "\0" in rel:
            raise ToolError("path must not contain NUL")
        candidate = Path(rel)
        if ".." in candidate.parts:
            raise ToolError("path escapes the working directory")
        if candidate.is_absolute():
            try:
                rel = str(candidate.relative_to(self.root))
            except ValueError:
                raise ToolError("absolute path is outside the working directory; use a relative path or an absolute path inside it") from None
        cur = self.root
        for part in Path(rel).parts:
            if part == "..":
                raise ToolError("path escapes the working directory")
            cur = cur / part
            if cur.is_symlink():
                raise ToolError("symlinks are not followed")
        p = self.root / rel
        real = Path(os.path.realpath(p))
        if self.root not in real.parents and real != self.root:
            raise ToolError("path escapes the working directory")
        if must_exist and not p.exists():
            raise ToolError("no such file or folder")
        return p

    def spill_dir(self) -> Path:
        if self._spill is None:
            self._spill = Path(tempfile.mkdtemp(prefix="ordivyr-lite-spill-"))
            try:
                os.chmod(self._spill, 0o700)
            except OSError:
                pass
        return self._spill

    def spill(self, text: str) -> int:
        with self._spill_lock:
            self._nspill += 1
            n = self._nspill
        (self.spill_dir() / str(n)).write_text(text, "utf-8")
        return n

    def close(self) -> None:
        closers, self._closers = self._closers, []
        for closer in closers:
            try:
                closer()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass
        session, self._sandbox = self._sandbox, None
        if session not in (None, False):
            try:
                session.close()
            except Exception:  # noqa: BLE001 - cleanup must never raise
                pass
        if self._spill is not None:
            import shutil
            shutil.rmtree(self._spill, ignore_errors=True)
            self._spill = None


# --------------------------------------------------------------------------- web fetch with SSRF guard
def _public_ip(host: str) -> str:
    infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    ips = [ipaddress.ip_address(i[4][0]) for i in infos]
    if not ips or not all(ip.is_global for ip in ips):
        raise ToolError("that address is private or local; only public web pages can be fetched")
    return str(ips[0])


class _Pinned(http.client.HTTPSConnection):
    """Connect to the IP we vetted (no second DNS lookup), still verifying the hostname."""

    def __init__(self, host: str, ip: str, **kw):
        super().__init__(host, **kw)
        self._ip = ip

    def connect(self) -> None:
        sock = socket.create_connection((self._ip, self.port or 443), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def fetch_url(url: str, _resolve=_public_ip) -> str:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ToolError("only plain http(s) URLs are allowed")
    if u.username or u.password:
        raise ToolError("do not put credentials in the URL")
    ip = _resolve(u.hostname)
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    if u.scheme == "https":
        conn = _Pinned(u.hostname, ip, port=u.port or 443, timeout=15,
                       context=ssl.create_default_context())
    else:
        conn = http.client.HTTPConnection(ip, u.port or 80, timeout=15)
    try:
        conn.request("GET", path, headers={"Host": u.netloc.split("@")[-1],
                                           "User-Agent": "ordivyr-lite",
                                           "Accept": "text/*,application/json"})
        r = conn.getresponse()
        if 300 <= r.status < 400:
            raise ToolError(f"redirected ({r.status}); redirects are not followed - fetch the target URL directly")
        body = r.read(300_000)
        if r.status >= 400:
            raise ToolError(f"HTTP {r.status}")
        text = body.decode("utf-8", "replace")
        if "html" in (r.getheader("content-type") or ""):
            text = re.sub(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", " ", text, flags=re.S)
        return text[:MAX_RESULT]
    finally:
        conn.close()
