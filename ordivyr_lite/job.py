"""Versioned JSON job/result protocol for Board -> Lite execution.

Lite is the execution authority: it validates the job, resolves the model,
enforces permissions and limits, runs the loop and returns one versioned
result.  Board owns orchestration/teams/scheduling/memory/hosting/billing; it
only needs to speak this contract.

Wire examples live in docs/ordivyr-lite/COMPATIBILITY.md.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from . import JOB_PROTOCOL, PROTOCOL_VERSION, RESULT_PROTOCOL, __version__
from .config import config_home, load_config, output_limit, resolve_route
from .errors import JobError
from .loop import RunLimits, Runner
from .router import ModelRouter
from .session import Session
from .tools import ALL_TOOL_NAMES, PROFILES, ToolBox
from .usage import UsageLedger
from .util import clip, new_run_id, utc_now
from .validators import Validator

SECRET_FIELDS = {"api_key", "key", "token", "secret", "password", "authorization"}
SUPPORTED_LIMIT_KEYS = {"max_turns", "max_tool_calls", "max_seconds", "max_cost_usd",
                        "max_tokens", "max_repairs", "output_tokens", "compact_at"}
SUPPORTED_MODEL_KEYS = {"provider", "model", "base_url", "protocol", "api_key_env",
                        "api_key_file", "context_window", "timeout_s", "max_retries",
                        "allow_paid_fallback", "prices"}
SUPPORTED_PERMISSION_KEYS = {"profile", "tools", "sandbox", "network", "allow_shell",
                             "allow_web", "timeout_s"}
SUPPORTED_VALIDATION_KEYS = {"mode", "command", "timeout_s"}
SUPPORTED_JOB_KEYS = {"protocol", "version", "job_id", "task", "model", "limits",
                      "permissions", "validation", "session", "spend", "metadata",
                      "created_at", "callback_url"}


def _check_no_secrets(obj: Any, path: str = "") -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in SECRET_FIELDS:
                raise JobError(f"inline secret field {path + str(k)!r} is not allowed; "
                               "use api_key_env or api_key_file")
            _check_no_secrets(v, path + str(k) + ".")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _check_no_secrets(v, f"{path}{i}.")


def validate_job(data: Any) -> dict:
    """Validate and normalise a job.  Raises JobError with a precise message."""
    if not isinstance(data, dict):
        raise JobError("job must be a JSON object")
    if data.get("protocol") != JOB_PROTOCOL:
        raise JobError(f"protocol must be {JOB_PROTOCOL!r}, got {data.get('protocol')!r}")
    try:
        version = int(data.get("version"))
    except (TypeError, ValueError):
        raise JobError("version must be an integer") from None
    if version != PROTOCOL_VERSION:
        raise JobError(f"unsupported protocol version {version}; this Lite speaks "
                       f"{PROTOCOL_VERSION}")
    unknown = set(data) - SUPPORTED_JOB_KEYS
    if unknown:
        raise JobError(f"unknown job field(s): {', '.join(sorted(unknown))}")
    _check_no_secrets(data)

    task = data.get("task")
    if not isinstance(task, dict):
        raise JobError("task must be an object")
    prompt = task.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise JobError("task.prompt must be a non-empty string")
    workdir = Path(str(task.get("workdir") or os.getcwd())).expanduser()
    if not workdir.is_absolute():
        workdir = (Path.cwd() / workdir).resolve()
    if not workdir.is_dir():
        raise JobError(f"task.workdir is not a directory: {workdir}")
    system = task.get("system")
    if system is not None and not isinstance(system, str):
        raise JobError("task.system must be a string or null")

    model = data.get("model")
    if isinstance(model, str):
        model = {"model": model}
    if not isinstance(model, dict):
        raise JobError("model must be an object")
    unknown = set(model) - SUPPORTED_MODEL_KEYS
    if unknown:
        raise JobError(f"unknown model field(s): {', '.join(sorted(unknown))}")
    if not model.get("model"):
        raise JobError("model.model is required")
    # `prices` is accepted for forward-compatible Board metadata but never
    # overrides the operator's local price table.
    model.pop("prices", None)

    limits = data.get("limits") or {}
    if not isinstance(limits, dict):
        raise JobError("limits must be an object")
    unknown = set(limits) - SUPPORTED_LIMIT_KEYS
    if unknown:
        raise JobError(f"unknown limit field(s): {', '.join(sorted(unknown))}")
    for key, value in limits.items():
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0:
            raise JobError(f"limits.{key} must be a non-negative number")

    perms = data.get("permissions") or {}
    if not isinstance(perms, dict):
        raise JobError("permissions must be an object")
    unknown = set(perms) - SUPPORTED_PERMISSION_KEYS
    if unknown:
        raise JobError(f"unknown permission field(s): {', '.join(sorted(unknown))}")
    profile = perms.get("profile") or "workspace"
    if profile not in PROFILES:
        raise JobError(f"permissions.profile must be one of {', '.join(PROFILES)}")
    for key in ("network", "allow_shell", "allow_web"):
        if key in perms and perms[key] is not None and type(perms[key]) is not bool:
            raise JobError(f"permissions.{key} must be true or false")
    if profile == "read-only" and (perms.get("allow_shell") is True or perms.get("allow_web") is True):
        raise JobError("read-only profile cannot enable shell or web; select workspace for a scoped grant")
    if perms.get("tools") is not None:
        if not isinstance(perms["tools"], list) or any(not isinstance(x, str) for x in perms["tools"]):
            raise JobError("permissions.tools must be a list of tool names")
        unknown_tools = set(perms["tools"]) - ALL_TOOL_NAMES
        if unknown_tools:
            raise JobError("permissions.tools contains unknown name(s): " + ", ".join(sorted(unknown_tools)))
    sandbox = perms.get("sandbox") or "auto"
    if sandbox not in ("auto", "bwrap", "process", "none"):
        raise JobError("permissions.sandbox must be auto|bwrap|process|none")
    if sandbox == "none":
        raise JobError("permissions.sandbox=none is not allowed through the job interface; "
                       "use bwrap or process")

    validation = data.get("validation") or {}
    if isinstance(validation, str):
        validation = {"mode": validation}
    if not isinstance(validation, dict):
        raise JobError("validation must be an object or a mode string")
    unknown = set(validation) - SUPPORTED_VALIDATION_KEYS
    if unknown:
        raise JobError(f"unknown validation field(s): {', '.join(sorted(unknown))}")
    mode = validation.get("mode") or "auto"
    if mode not in ("off", "syntax", "auto", "tests", "command"):
        raise JobError("validation.mode must be off|syntax|auto|tests|command")
    if mode == "command" and not validation.get("command"):
        raise JobError("validation.command is required when validation.mode is command")

    session = data.get("session") or {}
    if not isinstance(session, dict):
        raise JobError("session must be an object")
    unknown = set(session) - {"path", "resume", "enabled"}
    if unknown:
        raise JobError(f"unknown session field(s): {', '.join(sorted(unknown))}")
    if session.get("path") is not None and not isinstance(session.get("path"), str):
        raise JobError("session.path must be a string")

    spend = data.get("spend") or {}
    if not isinstance(spend, dict):
        raise JobError("spend must be an object")
    unknown = set(spend) - {"allow_paid", "max_cost_usd"}
    if unknown:
        raise JobError(f"unknown spend field(s): {', '.join(sorted(unknown))}")
    if spend.get("allow_paid") is not None and not isinstance(spend.get("allow_paid"), bool):
        raise JobError("spend.allow_paid must be a boolean")

    return {
        "protocol": JOB_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "job_id": str(data.get("job_id") or ""),
        "created_at": data.get("created_at") or utc_now(),
        "task": {"prompt": prompt, "workdir": str(workdir), "system": system},
        "model": model,
        "limits": limits,
        "permissions": {"profile": profile,
                        "tools": perms.get("tools"),
                        "sandbox": sandbox,
                        "network": bool(perms.get("network", False)),
                        "allow_shell": perms.get("allow_shell"),
                        "allow_web": perms.get("allow_web"),
                        "timeout_s": perms.get("timeout_s")},
        "validation": validation,
        "session": session,
        "spend": spend,
        "metadata": data.get("metadata") or {},
    }


def run_job(job: dict, *, cfg: dict | None = None, on_event=None,
            allow_paid: bool = False, cancel_event=None, exact_tools: bool = False) -> dict:
    job = validate_job(job)
    cfg = cfg or load_config()
    job_id = job["job_id"] or new_run_id("job")
    model = job["model"]
    spend = job["spend"]
    max_cost = float(spend.get("max_cost_usd") or job["limits"].get("max_cost_usd") or 0)
    route = resolve_route(
        cfg,
        provider=model.get("provider"),
        model=model.get("model"),
        base_url=model.get("base_url"),
        protocol=model.get("protocol"),
        api_key_env=model.get("api_key_env"),
        api_key_file=model.get("api_key_file"),
        context_window=model.get("context_window"),
        timeout=model.get("timeout_s"),
        # A job cannot authorise paid spending by itself; the invoking process
        # must pass --allow-paid (plus a positive max cost).
        allow_paid=bool(allow_paid),
        max_cost_usd=max_cost,
    )
    limits = RunLimits.from_config(cfg, **{k: v for k, v in job["limits"].items() if v is not None})
    perms = job["permissions"]
    box = ToolBox(job["task"]["workdir"], profile=perms["profile"],
                  allow_tools=perms.get("tools"), sandbox_mode=perms["sandbox"],
                  network=bool(perms.get("network")), timeout=int(perms.get("timeout_s") or 60),
                  allow_shell=perms.get("allow_shell"), allow_web=perms.get("allow_web"))
    val = job["validation"]
    if exact_tools and perms.get('tools') is not None:
        # Board permissions are exact grants; CLI convenience aliases must not
        # broaden an external runtime contract's capabilities.
        box.allow = set(perms['tools'])
        box.defs = box._build_defs()
    validator = Validator(mode=val.get("mode") or "auto", command=val.get("command"),
                          timeout=int(val.get("timeout_s") or 120),
                          sandbox_mode=perms["sandbox"])
    session = None
    try:
        sess_cfg = job["session"]
        if sess_cfg.get("enabled", True) is not False:
            if sess_cfg.get("path"):
                p = Path(sess_cfg["path"]).expanduser()
                if p.exists():
                    if not sess_cfg.get("resume"):
                        raise JobError(f"session {p} already exists; set session.resume=true "
                                       "or use a new path")
                    session = Session.open(p)
                else:
                    session = Session.create(p, run_id=job_id)
            else:
                if sess_cfg.get("resume"):
                    raise JobError("session.path is required to resume a job")
                session = Session.create()
        resume_state = session.latest_snapshot() if (session and sess_cfg.get("resume")) else None
        router = ModelRouter([route], ledger=UsageLedger(), on_event=on_event,
                             allow_paid_fallback=bool(allow_paid and model.get("allow_paid_fallback")))
        runner = Runner(router, box, job["task"]["prompt"], system=job["task"]["system"],
                        limits=limits, validator=validator, session=session,
                        on_event=on_event, run_id=job_id, cancel_event=cancel_event,
                        resume_state=resume_state)
        result = runner.run()
        return result_to_dict(result, job, job_id)
    finally:
        box.close()
        if session is not None:
            session.close()


def result_to_dict(result, job: dict, job_id: str) -> dict:
    return {
        "protocol": RESULT_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "lite_version": __version__,
        "job_id": job_id,
        "created_at": utc_now(),
        "status": result.status,
        "stop_reason": result.stop_reason,
        "error": result.error,
        "text": result.text,
        "provider": result.provider,
        "model": result.model,
        "route": result.route_label,
        "cost_usd": round(result.stats.cost_usd, 10),
        "usage": result.usage(),
        "stats": result.stats.as_dict(),
        "completion": result.completion,
        "tool_events": result.tool_events[-200:],
        "artifacts": result.artifacts,
        "session": {"path": result.session_path, "run_id": result.run_id,
                    "resume": bool((job.get("session") or {}).get("resume"))},
        "limits": job.get("limits") or {},
        "duration_ms": round(result.duration_ms, 1),
        "started_at": result.started_at,
        "ended_at": result.ended_at,
    }


def job_schema() -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "ORDIVYR Lite job",
        "type": "object",
        "required": ["protocol", "version", "task", "model"],
        "properties": {
            "protocol": {"const": JOB_PROTOCOL},
            "version": {"const": PROTOCOL_VERSION},
            "job_id": {"type": "string"},
            "task": {"type": "object", "required": ["prompt"], "properties": {
                "prompt": {"type": "string"}, "workdir": {"type": "string"},
                "system": {"type": ["string", "null"]}}},
            "model": {"type": "object", "required": ["model"], "properties": {
                "provider": {"type": "string"}, "model": {"type": "string"},
                "base_url": {"type": "string"}, "protocol": {"enum": ["openai", "anthropic"]},
                "api_key_env": {"type": "string"}, "api_key_file": {"type": "string"},
                "context_window": {"type": "integer"}, "timeout_s": {"type": "number"},
                "max_retries": {"type": "integer"}}},
            "limits": {"type": "object", "additionalProperties": {"type": "number"}},
            "permissions": {"type": "object", "properties": {
                "profile": {"enum": list(PROFILES)}, "tools": {"type": "array",
                                                               "items": {"type": "string"}},
                "sandbox": {"enum": ["auto", "bwrap", "process", "none"]},
                "network": {"type": "boolean"},
                "allow_shell": {"type": "boolean"}, "allow_web": {"type": "boolean"}}},
            "validation": {"type": "object", "properties": {
                "mode": {"enum": ["off", "syntax", "auto", "tests", "command"]},
                "command": {"type": "string"}, "timeout_s": {"type": "number"}}},
            "session": {"type": "object", "properties": {
                "path": {"type": "string"}, "resume": {"type": "boolean"},
                "enabled": {"type": "boolean"}}},
            "spend": {"type": "object", "properties": {
                "allow_paid": {"type": "boolean"}, "max_cost_usd": {"type": "number"}}},
            "metadata": {"type": "object"},
        },
        "additionalProperties": False,
    }


def example_job(workdir: str = ".") -> dict:
    return {
        "protocol": JOB_PROTOCOL,
        "version": PROTOCOL_VERSION,
        "job_id": "job-example-1",
        "task": {
            "prompt": "Read README.md and summarize the project in three sentences.",
            "workdir": workdir,
            "system": None,
        },
        "model": {
            "provider": "ollama",
            "model": "qwen2.5-coder:7b",
            "base_url": "http://127.0.0.1:11434/v1",
        },
        "limits": {"max_turns": 20, "max_tool_calls": 60, "max_seconds": 300,
                   "max_cost_usd": 0, "max_tokens": 0},
        "permissions": {"profile": "read-only", "sandbox": "auto", "network": False},
        "validation": {"mode": "syntax"},
        "session": {"resume": False, "enabled": True},
        "spend": {"allow_paid": False, "max_cost_usd": 0},
        "metadata": {"source": "example"},
    }
