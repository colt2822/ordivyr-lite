"""ORDIVYR runtime contract v1 - the integration boundary between Board (orchestration) and ANY execution runtime (the built-in harness, ORDIVYR Lite, a customer worker).

STDLIB ONLY and imports nothing else from ORDIVYR: a runtime can vendor this single file. Normative text: docs/architecture/ORDIVYR_RUNTIME_CONTRACT.md.

Rules the code below makes mechanical:
  * a job carries model/key REFERENCES, never secrets (`validate_job` rejects anything that looks like a key);
  * tool permission is decided by `tool_allowed`, which a runtime must call before EVERY tool execution (a prompt is not a permission);
  * a result always carries the same usage keys, so Board can meter any runtime the same way;
  * unknown MAJOR version -> `unsupported_version`, never a best-effort guess.
"""
from __future__ import annotations
import re

CONTRACT_VERSION = "1.1"        # 1.1 (additive): memory tools, child_results, capabilities.endpoints. 1.0 jobs/results remain valid.
MAJOR = 1

# --- tools. The built-in set is the harness's own; a runtime may offer more but Board only grants names it has seen in that runtime's capabilities.
READ_TOOLS = frozenset({"read_file", "list_files", "grep", "find_files", "ask_user"})
WRITE_TOOLS = frozenset({"write_file", "edit_file"})          # need permissions.write
WEB_TOOLS = frozenset({"fetch_url"})            # need permissions.web
SHELL_TOOLS = frozenset({"terminal_run", "terminal_read"})        # need permissions.terminal != "none"
DELEGATE_TOOLS = frozenset({"delegate", "child_results"})    # need permissions.delegate (true | list of agent ids); Board enforces org/project/depth/fan-out/budget server-side
MEMORY_READ_TOOLS = frozenset({"memory_get", "memory_list"})  # need permissions.memory in ("read","write")
MEMORY_WRITE_TOOLS = frozenset({"memory_put"})                # need permissions.memory == "write"

STOP_REASONS = ("done", "steps", "tool_calls", "time", "budget", "malformed", "repeat_failure", "unverified", "cancelled", "error")
STATUSES = ("completed", "failed", "cancelled", "held")       # held = needs a human (owner approval, unpaid quota, ambiguous side effect)
COMPLETION_LEVELS = ("model_said_done", "code_written", "syntax_ok", "tests_passed", "independent_review")   # tri-state: True | False | None (None = nothing checked it)
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "cache_read_tokens", "cache_write_tokens", "turns", "tool_calls", "cost_usd", "wall_seconds")

# error code -> retryable by Board with the same job. Anything not listed is treated as non-retryable.
ERROR_CODES = {
    "unsupported_version": False, "invalid_job": False, "permission_denied": False, "tool_not_granted": False,
    "model_unavailable": True, "provider_error": True, "rate_limited": True, "runtime_unavailable": True, "timeout": True,
    "budget_exceeded": False, "cap_exceeded": False, "needs_owner": False, "validation_failed": True, "cancelled": False, "internal": True,
}
HELD_CODES = frozenset({"needs_owner", "cap_exceeded"})        # -> status "held", not "failed"

_SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|sk-ant-[A-Za-z0-9_-]{10,}|AIza[0-9A-Za-z_-]{20,}|xox[abp]-[0-9A-Za-z-]{10,}|ghp_[0-9A-Za-z]{20,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")
_SECRET_KEYS = frozenset({"api_key", "apikey", "key", "secret", "token", "password", "authorization", "bearer"})
_LIMITS = {"max_turns": (1, 1000), "max_tool_calls": (0, 10000), "max_seconds": (1, 86400), "budget_usd": (0, 100000), "tenant_budget_usd": (0, 100000), "max_output_tokens": (1, 1_000_000), "token_budget": (1, 10**9)}


class ContractError(ValueError):
    def __init__(self, code, msg): super().__init__(msg); self.code = code


def version_ok(v):
    try: return int(str(v).split(".")[0]) == MAJOR
    except (ValueError, TypeError): return False


def _walk_secrets(x, path="", out=None):
    out = [] if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            if str(k).lower() in _SECRET_KEYS and isinstance(v, str) and v: out.append(f"{path}/{k}")
            _walk_secrets(v, f"{path}/{k}", out)
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x): _walk_secrets(v, f"{path}/{i}", out)
    elif isinstance(x, str) and _SECRET.search(x): out.append(path or "/")
    return out


def validate_job(job):
    """-> list of problems (empty = acceptable). Pure; no I/O."""
    p = []
    if not isinstance(job, dict): return ["job must be an object"]
    if not version_ok(job.get("contract_version")): p.append(f"contract_version {job.get('contract_version')!r} unsupported (this is {CONTRACT_VERSION})")
    for k in ("org", "project", "job_id"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", str(job.get(k) or "")): p.append(f"{k} missing or invalid")
    if not str(job.get("prompt") or "").strip(): p.append("prompt required")
    ag = job.get("agent")
    if not isinstance(ag, dict) or not ag.get("id"): p.append("agent.id required")
    else:
        perms = ag.get("permissions") or {}
        if not isinstance(perms, dict): p.append("agent.permissions must be an object")
        tools = ag.get("tools")
        if tools is not None and not (isinstance(tools, list) and all(isinstance(t, str) for t in tools)): p.append("agent.tools must be null or a list of names")
    m = job.get("model_ref")
    if m is not None and not (isinstance(m, dict) and m.get("endpoint") and m.get("model")): p.append("model_ref needs endpoint and model (logical names; never URLs with credentials)")
    for k, v in (job.get("limits") or {}).items():
        if k not in _LIMITS: p.append(f"unknown limit {k}")
        elif not isinstance(v, (int, float)) or isinstance(v, bool) or not (_LIMITS[k][0] <= v <= _LIMITS[k][1]): p.append(f"limit {k} out of range {_LIMITS[k]}")
    par = job.get("parent")
    if par is not None and not (isinstance(par, dict) and par.get("job_id") and isinstance(par.get("depth"), int) and par["depth"] >= 0): p.append("parent needs job_id and integer depth")
    for where in _walk_secrets({k: v for k, v in job.items() if k != "prompt"}): p.append(f"credential-like value at {where}: send a reference, never a secret")
    return p


def tool_allowed(job, tool):
    """THE permission decision. A runtime MUST call this before executing any tool and MUST refuse (error `tool_not_granted`) when it returns False.
    Grant = tool listed in agent.tools (None = every tool the permissions allow) AND the capability the tool needs is switched on in agent.permissions."""
    ag = job.get("agent") or {}; perms = ag.get("permissions") or {}; grants = ag.get("tools")
    if grants is not None and tool not in grants: return False
    if tool in WRITE_TOOLS: return bool(perms.get("write"))
    if tool in WEB_TOOLS: return bool(perms.get("web"))
    if tool in SHELL_TOOLS: return perms.get("terminal", "none") not in ("none", None, False, "")
    if tool in DELEGATE_TOOLS: return bool(perms.get("delegate"))
    if tool in MEMORY_READ_TOOLS: return perms.get("memory") in ("read", "write")
    if tool in MEMORY_WRITE_TOOLS: return perms.get("memory") == "write"
    return tool in READ_TOOLS or grants is not None and tool in grants      # unknown tools only if explicitly granted by name


def empty_usage(): return {"prompt_tokens": 0, "completion_tokens": 0, "cache_read_tokens": 0, "cache_write_tokens": 0, "turns": 0, "tool_calls": 0, "cost_usd": 0.0, "wall_seconds": 0.0}


def completion(**levels):
    c = {k: None for k in COMPLETION_LEVELS}
    for k, v in levels.items():
        if k not in c: raise ValueError(f"unknown completion level {k}")
        c[k] = v
    return c


def make_result(job_id, status, stop, text="", usage=None, error=None, artifacts=(), evidence=(), comp=None, runtime=None):
    """Build a contract-conformant result. `error` = (code, message) for failed/held."""
    if status not in STATUSES: raise ValueError(f"status {status!r}")
    if stop not in STOP_REASONS: raise ValueError(f"stop {stop!r}")
    u = {**empty_usage(), **(usage or {})}
    e = None
    if error:
        code, msg = error
        if code not in ERROR_CODES: raise ValueError(f"error code {code!r}")
        e = {"code": code, "message": str(msg)[:2000], "retryable": ERROR_CODES[code]}
    return {"contract_version": CONTRACT_VERSION, "job_id": job_id, "status": status, "stop": stop, "text": text, "usage": u, "error": e, "artifacts": list(artifacts), "evidence": list(evidence),
            "completion": comp or completion(), "runtime": runtime or {}}


def validate_result(r):
    p = []
    if not isinstance(r, dict): return ["result must be an object"]
    if not version_ok(r.get("contract_version")): p.append("contract_version unsupported")
    if r.get("status") not in STATUSES: p.append(f"status {r.get('status')!r}")
    if r.get("stop") not in STOP_REASONS: p.append(f"stop {r.get('stop')!r}")
    u = r.get("usage")
    if not isinstance(u, dict) or any(k not in u for k in USAGE_KEYS): p.append("usage must carry every key in USAGE_KEYS")
    elif any(not isinstance(u[k], (int, float)) or isinstance(u[k], bool) or not (0 <= u[k] < float("inf")) for k in USAGE_KEYS): p.append("usage values must be finite non-negative numbers")
    if r.get("status") in ("failed", "held") and not (isinstance(r.get("error"), dict) and r["error"].get("code") in ERROR_CODES): p.append("failed/held result needs error.code from ERROR_CODES")
    if r.get("status") == "held" and (r.get("error") or {}).get("code") not in HELD_CODES: p.append("held requires a HELD_CODES error")
    c = r.get("completion")
    if not isinstance(c, dict) or set(c) != set(COMPLETION_LEVELS) or any(v not in (True, False, None) for v in c.values()): p.append("completion must be tri-state for every level")
    elif r.get("status") == "completed" and r.get("stop") == "done" and c["tests_passed"] is False: p.append("completed/done contradicts tests_passed=False")
    return p


def capabilities(engine, tools, max_concurrency=1, features=(), models_via="gateway", version=None, endpoints=(), isolation="process"):
    """What a runtime publishes at registration. Board only dispatches jobs whose requirements are a subset of this."""
    return {"contract_version": version or CONTRACT_VERSION, "engine": str(engine), "tools": sorted(set(tools)), "max_concurrency": int(max_concurrency),
            "features": sorted(set(features)), "models_via": models_via, "endpoints": sorted(set(endpoints)), "isolation": isolation}


ISOLATION_LEVELS = ("process", "namespace", "container", "microvm")      # what separates one job from the host and from other jobs; Board only places SHARED-pool work on non-"process" workers
FEATURES = frozenset({"cancel", "checkpoint", "resume", "validators", "delegate", "streaming", "vision"})


def validate_capabilities(c):
    p = []
    if not isinstance(c, dict): return ["capabilities must be an object"]
    if not version_ok(c.get("contract_version")): p.append("contract_version unsupported")
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,31}", str(c.get("engine") or "")): p.append("engine id invalid")
    if not isinstance(c.get("tools"), list): p.append("tools must be a list")
    if not isinstance(c.get("max_concurrency"), int) or not 1 <= c["max_concurrency"] <= 4096: p.append("max_concurrency 1..4096")
    if not set(c.get("features") or ()) <= FEATURES: p.append("unknown feature")
    if c.get("isolation", "process") not in ISOLATION_LEVELS: p.append(f"isolation must be one of {ISOLATION_LEVELS}")
    if not isinstance(c.get("endpoints", []), list): p.append("endpoints must be a list of logical endpoint ids")
    return p


def requirements_met(job, caps):
    """Board-side dispatch check: the runtime must offer every tool the job could be granted, and `cancel` (mandatory for any hosted runtime)."""
    need = set()
    ag = job.get("agent") or {}
    for t in (ag.get("tools") or []): need.add(t)
    feats = set(caps.get("features") or ())
    if job.get("validators") and "validators" not in feats: return False          # validators execute commands on the worker host: only runtimes that explicitly advertise it
    return need <= set(caps.get("tools") or ()) and "cancel" in feats
