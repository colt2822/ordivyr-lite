"""The ``ordivyr-lite`` command line.

Commands: run, chat, job, config, doctor, models, usage, sessions, resume.
Everything is stdlib-only and imports in a few tens of milliseconds so a
one-shot task does not pay for a web stack it never uses.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_LIMITS, PRESETS, config_home, config_summary, doctor_report, load_config,
    output_limit, resolve_route, save_config, store_provider_key,
)
from .errors import ConfigError, LiteError, exit_code_for
from .loop import RunLimits, Runner
from .providers import ModelClient, probe_local
from .router import ModelRouter
from .session import Session, list_sessions, prune_sessions
from .tools import PERMISSION_NAMES, ToolBox, ToolError
from .usage import UsageLedger, format_report, parse_since
from .util import clip, human_int, human_ms, human_usd
from .validators import Validator


# --------------------------------------------------------------------------- helpers
def _set_home(path: str | None) -> None:
    if path:
        os.environ["ORDIVYR_LITE_HOME"] = str(Path(path).expanduser().resolve())


def _common_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", help="provider preset or a name in config.providers")
    p.add_argument("--model", help="model id (or provider/model)")
    p.add_argument("--base-url", help="override the provider base URL")
    p.add_argument("--protocol", choices=("openai", "anthropic"), help="wire protocol override")
    p.add_argument("--api-key-env", help="environment variable holding the API key")
    p.add_argument("--api-key-file", help="file holding the API key (first line)")
    p.add_argument("--header", action="append", default=[], metavar="NAME:VALUE",
                   help="extra request header (repeatable); VALUE @ENV reads an environment variable")
    p.add_argument("--context-window", type=int, help="model context window, for output limits")
    p.add_argument("--timeout", type=float, help="per-request timeout seconds")
    p.add_argument("--allow-paid", action="store_true",
                   help="explicitly authorise a paid endpoint (requires --max-cost)")
    p.add_argument("--max-cost", type=float, default=0.0,
                   help="paid cap in USD (also the run cost cap)")
    p.add_argument("--allow-paid-fallback", action="store_true",
                   help="allow falling back from a free route to a paid one")
    p.add_argument("--local-only", action="store_true",
                   help="hard-restrict every provider attempt to a loopback model endpoint")
    p.add_argument("--fallback", action="append", default=[],
                   metavar="PROVIDER:MODEL", help="fallback route, repeatable")
    p.add_argument("--output-tokens", type=int, help="override the requested output-token ceiling")


def _common_limit_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-turns", type=int, help="model turns (default 64)")
    p.add_argument("--max-tool-calls", type=int, help="tool executions (default 400)")
    p.add_argument("--max-seconds", type=float, help="wall-clock limit (default 1800)")
    p.add_argument("--max-tokens", type=int, help="fresh-token budget (0 = unlimited)")
    p.add_argument("--max-repairs", type=int, help="validation repair rounds (default 2)")
    p.add_argument("--compact-at", type=int,
                   help="request context compaction at this token count (capped by effective window)")
    p.add_argument('--fresh-review', action=argparse.BooleanOptionalAction, default=None,
                   help='one same-model text-only check after repeated failures, inside the existing budget')


def _common_permission_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--permission", choices=PERMISSION_NAMES, default=None,
                   help="read-only | workspace | trusted | full")
    p.add_argument("--tools", help="comma-separated tool allow-list (narrows the profile)")
    p.add_argument("--delegate", action="store_true",
                   help="enable one-level child agents (off by default)")
    p.add_argument("--mcp", metavar="FILE",
                   help="MCP server config (stdio). Off unless this file is passed")
    p.add_argument("--sandbox", choices=("auto", "bwrap", "process", "none"), default=None,
                   help="command sandbox (default auto = bwrap)")
    network = p.add_mutually_exclusive_group()
    network.add_argument("--network", dest="network", action="store_true", default=None,
                         help="allow network in run_command")
    network.add_argument("--no-network", dest="network", action="store_false",
                         help="explicitly disable network in run_command")
    shell = p.add_mutually_exclusive_group()
    shell.add_argument("--allow-shell", dest="allow_shell", action="store_true", default=None,
                       help="explicitly enable sandboxed shell (workspace or trusted profile)")
    shell.add_argument("--no-shell", dest="allow_shell", action="store_false",
                       help="explicitly disable shell even when the profile allows it")
    web = p.add_mutually_exclusive_group()
    web.add_argument("--allow-web", dest="allow_web", action="store_true", default=None,
                     help="explicitly enable public web fetch (workspace or trusted profile)")
    web.add_argument("--no-web", dest="allow_web", action="store_false",
                     help="explicitly disable web even when the profile allows it")
    p.add_argument("--unsafe-sandbox", action="store_true",
                   help="allow --sandbox none (unsafe; only for already-isolated hosts)")


def _common_output_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--json", action="store_true", help="machine-readable result on stdout")
    p.add_argument("--quiet", action="store_true", help="no progress on stderr")
    p.add_argument("--verbose", action="store_true", help="more progress detail")
    p.add_argument("--events", metavar="FILE", help="write JSONL progress events to FILE")
    p.add_argument("--dry-run", action="store_true", help="resolve config and print plan; no model call")
    p.add_argument("--no-stream", action="store_true",
                   help="wait for each complete model turn instead of streaming tokens")


def _read_prompt(args, *, allow_empty: bool = False) -> str:
    prompt = getattr(args, "prompt", None)
    source = getattr(args, "prompt_file", None)
    if prompt and source:
        raise ConfigError("give either a prompt argument or --prompt-file, not both")
    if source:
        if source == "-":
            prompt = sys.stdin.read()
        else:
            try:
                prompt = Path(source).expanduser().read_text("utf-8")
            except OSError as exc:
                raise ConfigError(f"cannot read prompt file: {exc}") from exc
    if prompt is None and not sys.stdin.isatty():
        prompt = sys.stdin.read()
    if not prompt or not prompt.strip():
        if allow_empty:
            return ""
        raise ConfigError("no prompt given")
    return prompt


def _parse_fallback(spec: str) -> tuple[str, str]:
    text = spec.strip()
    if ":" in text:
        provider, model = text.split(":", 1)
    elif "/" in text:
        provider, model = text.split("/", 1)
    else:
        raise ConfigError(f"fallback must be PROVIDER:MODEL, got {spec!r}")
    if not provider or not model:
        raise ConfigError(f"fallback must be PROVIDER:MODEL, got {spec!r}")
    return provider, model


_UNSET = object()


def _route_headers(args) -> dict[str, str]:
    """Parse repeated --header NAME:VALUE pairs into one mapping."""
    pairs: dict[str, str] = {}
    for raw in getattr(args, "header", None) or []:
        name, sep, value = str(raw).partition(":")
        if not sep or not name.strip():
            raise ConfigError(f"--header expects NAME:VALUE, got {raw!r}")
        pairs[name.strip()] = value.strip()
    return pairs


def _route(cfg, args, *, provider=None, model=None, base_url=_UNSET):
    return resolve_route(
        cfg, provider=provider if provider is not None else args.provider,
        model=model if model is not None else args.model,
        base_url=args.base_url if base_url is _UNSET else base_url,
        protocol=args.protocol, api_key_env=args.api_key_env,
        api_key_file=args.api_key_file, context_window=args.context_window,
        timeout=args.timeout, allow_paid=args.allow_paid,
        max_cost_usd=args.max_cost, headers=_route_headers(args))


def _build_router(cfg, args, on_event) -> ModelRouter:
    primary = _route(cfg, args)
    routes = [primary]
    for spec in args.fallback:
        provider, model = _parse_fallback(spec)
        routes.append(_route(cfg, args, provider=provider, model=model, base_url=None))
    return ModelRouter(routes, ledger=UsageLedger(), allow_paid_fallback=args.allow_paid_fallback,
                       local_only=getattr(args, "local_only", False), on_event=on_event)


def _attach_mcp(box, args) -> None:
    path = getattr(args, "mcp", None)
    if not path:
        return
    from .mcpclient import McpError, attach_config
    try:
        attach_config(box, Path(path).expanduser())
    except (OSError, ValueError, McpError) as exc:
        box.close()
        raise ConfigError(str(exc)) from exc


def _limits(cfg, args) -> RunLimits:
    return RunLimits.from_config(
        cfg, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
        max_seconds=args.max_seconds, max_cost_usd=args.max_cost or None,
        max_tokens=args.max_tokens, max_repairs=args.max_repairs,
        output_tokens=args.output_tokens, compact_at=args.compact_at,
        delegate=bool(getattr(args, "delegate", False)),
        fresh_failure_review=getattr(args, 'fresh_review', None))


def _box(args, cfg, *, resume_permissions: dict | None = None,
        prompt: str | None = None) -> ToolBox:
    perms = cfg.get("permissions") or {}
    try:
        raw_cfg = json.loads((config_home() / "config.json").read_text("utf-8"))
    except (OSError, ValueError):
        raw_cfg = {}
    configured = raw_cfg.get("permissions") if isinstance(raw_cfg, dict) else None
    configured = configured if isinstance(configured, dict) else {}
    configured = {k: v for k, v in configured.items()
                  if k in {"profile", "sandbox", "network", "allow_shell", "allow_web"}}
    has_saved = bool(resume_permissions)
    profile_was_set = args.permission is not None or "profile" in configured
    profile = (args.permission or configured.get("profile")
               or (resume_permissions.get("profile") if has_saved else None)
               or perms.get("profile") or "workspace")
    sandbox = (args.sandbox or configured.get("sandbox")
               or (resume_permissions.get("sandbox") if has_saved else None)
               or perms.get("sandbox") or "auto")
    saved_tools = resume_permissions.get("tools") if has_saved else None
    requested_tools = ([x.strip() for x in args.tools.split(",") if x.strip()]
                       if args.tools is not None else None)
    if requested_tools is not None and saved_tools is not None:
        tools = sorted(set(requested_tools) & set(saved_tools))
    else:
        tools = requested_tools if requested_tools is not None else saved_tools
    # A one-shot reply does not inherit the workspace tool surface. An explicit
    # profile, tool list, grant, delegate, MCP, plan, or saved resume does.
    # Interactive chat never passes prompt, so it keeps the profile tools.
    operator_chose_tools = (
        requested_tools is not None or args.permission is not None
        or bool(configured.get("profile")) or "allow_shell" in configured
        or "allow_web" in configured or bool(getattr(args, "delegate", False))
        or bool(getattr(args, "mcp", None)) or bool(getattr(args, "plan", False))
        or args.allow_shell is not None or args.allow_web is not None
        or resume_permissions is not None
    )
    if prompt is not None and tools is None and not operator_chose_tools:
        from .context import task_needs_tools
        if not task_needs_tools(prompt):
            tools = []
    network = (args.network if args.network is not None else configured.get("network",
               resume_permissions.get("network", perms.get("network", False)) if has_saved
               else perms.get("network", False)))
    if args.allow_shell is not None:
        allow_shell = args.allow_shell
    elif "allow_shell" in configured:
        allow_shell = configured["allow_shell"]
    elif has_saved and not profile_was_set:
        allow_shell = resume_permissions.get("shell")
    else:
        allow_shell = None
    if args.allow_web is not None:
        allow_web = args.allow_web
    elif "allow_web" in configured:
        allow_web = configured["allow_web"]
    elif has_saved and not profile_was_set:
        allow_web = resume_permissions.get("web")
    else:
        allow_web = None
    for key, value in (("allow_shell", allow_shell), ("allow_web", allow_web)):
        if value is not None and type(value) is not bool:
            raise ConfigError(f"permissions.{key} must be true or false")
    if not isinstance(network, bool):
        raise ConfigError("permissions.network must be true or false")
    if profile == "read-only" and (allow_shell is True or allow_web is True):
        raise ConfigError("read-only profile cannot enable shell or web; select workspace for a scoped grant")
    if sandbox == "none" and not getattr(args, "unsafe_sandbox", False):
        raise ConfigError("--sandbox none is unsafe; pass --unsafe-sandbox only on an already-isolated host")
    workdir = Path(args.workdir).expanduser().resolve()
    try:
        return ToolBox(workdir, profile=profile, allow_tools=tools, sandbox_mode=sandbox,
                       network=network,
                       timeout=int(perms.get("timeout_s") or 60),
                       allow_shell=allow_shell, allow_web=allow_web,
                       plan=bool(getattr(args, "plan", False)))
    except ToolError as exc:
        raise ConfigError(str(exc)) from exc


def _validator(args, cfg, box: ToolBox, resume_state=None) -> Validator:
    state = resume_state or {}
    saved = (state.get('verification') or {}).get('policy') or {}
    timeout = args.validate_timeout or saved.get('timeout') or 120
    if getattr(args, "check", None):
        return Validator("command", args.check, timeout=timeout,
                         sandbox_mode=box.sandbox_mode)
    if args.validate is None and saved:
        mode = saved.get('mode')
        if mode not in ('off', 'syntax', 'auto', 'tests', 'command'):
            raise ConfigError('Saved validation policy is invalid; select --validate or --check explicitly')
        if mode == 'command' and not saved.get('command'):
            raise ConfigError('Saved custom check is missing; inspect the session before continuing')
        return Validator(mode, saved.get('command'), timeout=timeout, sandbox_mode=box.sandbox_mode)
    if args.validate is None and not saved and state.get('pending_final_text') is not None:
        raise ConfigError('Legacy session has no saved validation policy; inspect it and select --validate or --check explicitly')
    mode = args.validate or cfg.get("validate") or "auto"
    return Validator(mode, timeout=timeout, sandbox_mode=box.sandbox_mode)


def _session_for_run(args, run_id: str | None = None) -> Session | None:
    if args.no_session:
        return None
    if args.session:
        p = Path(args.session).expanduser()
        if p.is_dir():
            p = p / f"{run_id or 'run'}.jsonl"
        if p.exists():
            if args.resume:
                return Session.open(p)
            raise ConfigError(f"session {p} already exists; use --resume or a new path")
        return Session.create(p, run_id=run_id)
    if args.resume:
        sessions = list_sessions()
        if not sessions:
            raise ConfigError("no session to resume")
        return Session.open(sessions[0]["path"])
    return Session.create(run_id=run_id)


def _stream_enabled(args) -> bool:
    """`--json` and `--quiet` hide token deltas. Only `--no-stream` closes the SSE wire."""
    return not getattr(args, "no_stream", False)


def _event_printer(args):
    if args.json or args.quiet:
        def quiet(_ev):
            return None
        return quiet

    state = {"open": False}

    def show(ev: dict) -> None:
        t = ev.get("type")
        if t == "delta":
            piece = ev.get("text") or ""
            if piece and ev.get("kind") != "reasoning":
                sys.stdout.write(piece)
                sys.stdout.flush()
                state["open"] = True
                state["text"] = state.get("text", "") + piece
            return
        if state["open"]:
            sys.stdout.write("\n")
            sys.stdout.flush()
            state["open"] = False
        if t == "run_start":
            print(f"lite: {ev.get('model')} · {ev.get('workdir')} · {ev.get('permission')}",
                  file=sys.stderr)
        elif t == "model_turn":
            u = ev.get("usage") or {}
            parts = [f"turn {ev.get('turn')}", f"{u.get('input', 0) + u.get('cache_read', 0)} ctx tok"]
            if u.get("cache_read"):
                parts.append(f"cache {u['cache_read']}")
            parts += [human_ms(ev.get("latency_ms")), human_usd(ev.get("cost_usd"))]
            if ev.get("tool_calls"):
                parts.append(f"{ev['tool_calls']} call(s)")
            print("  " + " · ".join(str(x) for x in parts), file=sys.stderr)
        elif t == "tool_call":
            if args.verbose:
                print(f"    → {ev.get('tool')} {ev.get('args')}", file=sys.stderr)
            else:
                print(f"    → {ev.get('tool')}", file=sys.stderr)
        elif t == "tool_result":
            if args.verbose:
                print(f"    ← {ev.get('tool')} {'ok' if ev.get('ok') else 'ERROR'} "
                      f"({human_ms(ev.get('duration_ms'))}, {ev.get('output_chars')} chars)",
                      file=sys.stderr)
        elif t == "repair":
            print(f"  repair {ev.get('attempt')}: validation failed ({ev.get('kind')})",
                  file=sys.stderr)
        elif t == "compaction":
            print(f"  compacted history: removed {ev.get('removed_chars')} chars", file=sys.stderr)
        elif t == "route_fallback":
            print(f"  route fallback: {ev.get('from')} -> {ev.get('to')}", file=sys.stderr)
        elif t == "output_limit_halved":
            print(f"  output limit halved to {ev.get('limit')}", file=sys.stderr)
        elif t == "error":
            print(f"  error: {ev.get('message')}", file=sys.stderr)

    show.streamed = state  # type: ignore[attr-defined]
    return show


def _summary(result) -> str:
    s = result.stats
    return (f"{result.status} ({result.stop_reason}) · {s.model_calls} turns · {s.tool_calls} tools "
            f"({s.tool_success_rate:.0%} ok) · "
            f"tokens in {human_int(s.input_tokens + s.cached_input_tokens + s.cache_write_tokens)} "
            f"(cache {s.cache_hit_rate:.0%}) out {human_int(s.output_tokens)} · "
            f"{human_usd(s.cost_usd)} · {human_ms(s.wall_ms)}")


def _emit_events_file(path: str | None):
    if not path:
        return None
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    f = open(p, "a", encoding="utf-8", buffering=1)

    def write(ev: dict) -> None:
        f.write(json.dumps(ev, default=str) + "\n")

    write.close = f.close  # type: ignore[attr-defined]
    return write


def _print_plan(args, cfg, route, box, limits, validator) -> int:
    plan = {
        "lite_version": __version__,
        "workdir": str(box.root),
        "route": route.redacted(),
        "limits": limits.as_dict(),
        "permission": box.profile,
        "tools": box.names(),
        "tool_permissions": box.permission_report(),
        "sandbox": box.sandbox_mode,
        "network": box.network,
        "validator": {"mode": validator.mode, "command": validator.command},
        "session": "disabled" if args.no_session else (args.session or "auto"),
    }
    print(json.dumps(plan, indent=2))
    return 0


# --------------------------------------------------------------------------- run
def cmd_run(args) -> int:
    cfg = load_config()
    args.workdir = str(Path(args.workdir).expanduser().resolve())
    if not Path(args.workdir).is_dir():
        raise ConfigError(f"working directory does not exist: {args.workdir}")
    prompt = _read_prompt(args, allow_empty=bool(getattr(args, "resume", False)))
    system = _read_system(args)
    events_file = None
    user_printer = _event_printer(args)

    def on_event(ev):
        user_printer(ev)
        if events_file:
            events_file(ev)

    router = _build_router(cfg, args, on_event)
    session = _session_for_run(args) if args.resume else None
    resume_state = session.latest_snapshot() if session else None
    saved_permissions = session.metadata().get("permissions") if session else None
    box = _box(args, cfg, resume_permissions=saved_permissions, prompt=prompt)
    _attach_mcp(box, args)
    validator = _validator(args, cfg, box, resume_state)
    limits = _limits(cfg, args)
    if args.dry_run:
        if session:
            session.close()
        return _print_plan(args, cfg, router.primary, box, limits, validator)
    events_file = _emit_events_file(args.events)
    if session is None:
        session = _session_for_run(args)
    resume_state = session.latest_snapshot() if (session and args.resume) else None
    if resume_state is not None and not resume_state.get("effective_permissions") and saved_permissions:
        resume_state["effective_permissions"] = saved_permissions
    if session:
        session.set_metadata(permissions=box.permission_manifest())
    print(box.permission_summary(), file=sys.stderr)
    cancel_event = None
    try:
        runner = Runner(router, box, prompt, system=system, limits=limits,
                        validator=validator, session=session, on_event=on_event,
                        run_id=(session.run_id if session else None),
                        resume_state=resume_state, cancel_event=cancel_event,
                        stream=_stream_enabled(args), scope=getattr(args, "scope", None))
        result = runner.run()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    finally:
        box.close()
    if events_file and hasattr(events_file, "close"):
        events_file.close()
    if session:
        session.close()
    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        streamed = getattr(user_printer, "streamed", {}).get("text") or ""
        if not args.quiet and not streamed.strip():
            print(result.text)
        elif streamed.strip() and streamed.strip() != (result.text or "").strip():
            print(result.text)
        print(_summary(result), file=sys.stderr)
    return exit_code_for(result.stop_reason)


def _read_system(args) -> str | None:
    if getattr(args, "system_file", None):
        try:
            return Path(args.system_file).expanduser().read_text("utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read system prompt file: {exc}") from exc
    return getattr(args, "system", None)


# --------------------------------------------------------------------------- chat
def cmd_start(args) -> int:
    """Native onboarding, preserving noninteractive configuration errors."""
    cfg = load_config()
    force = args.command in ("setup", "connect")
    explicit = bool(args.provider or args.model or args.base_url)
    if force or (not explicit and not cfg.get("default_model")):
        if not sys.stdin.isatty():
            raise ConfigError("Connect a model first: run `ordivyr-lite setup` in a terminal.")
        from .onboarding import configure
        try:
            if not configure():
                return 0
        except EOFError:
            print("\nSetup cancelled; unsaved choices were discarded.", file=sys.stderr)
            return 0
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
    if args.no_chat:
        return 0
    return cmd_chat(args)


def _switch_model(router, args, spec, on_event):
    """Never carry one provider's explicit credential or spend grant to another."""
    provider, model = _parse_fallback(spec)
    candidate = copy.copy(args)
    candidate.provider, candidate.model = provider, model
    for name in ("base_url", "protocol", "api_key_env", "api_key_file", "header", "context_window"):
        setattr(candidate, name, None)
    candidate.fallback = []
    candidate.allow_paid = False
    candidate.max_cost = 0.0
    candidate.allow_paid_fallback = False
    replacement = _build_router(load_config(), candidate, on_event)
    replacement.ledger = router.ledger
    return replacement, candidate


def cmd_chat(args) -> int:
    cfg = load_config()
    args.workdir = str(Path(args.workdir).expanduser().resolve())
    if not Path(args.workdir).is_dir():
        raise ConfigError(f"working directory does not exist: {args.workdir}")
    events_file = _emit_events_file(args.events)
    user_printer = _event_printer(args)

    def on_event(ev):
        user_printer(ev)
        if events_file:
            events_file(ev)

    router = _build_router(cfg, args, on_event)
    session = _session_for_run(args) if args.resume else None
    resume_state = session.latest_snapshot() if session else None
    saved_permissions = session.metadata().get("permissions") if session else None
    box = _box(args, cfg, resume_permissions=saved_permissions)
    _attach_mcp(box, args)
    validator = _validator(args, cfg, box, resume_state)
    limits = _limits(cfg, args)
    if session is None:
        session = _session_for_run(args)
    prior_permissions = session.metadata().get("permissions") if session else None
    if session:
        session.set_metadata(permissions=box.permission_manifest())
    conversation: list[dict] = []
    if session and args.resume:
        state = session.latest_snapshot()
        if state:
            conversation = [dict(m) for m in (state.get("messages") or [])]
            print(f"resumed {len(conversation)} messages from {session.path}", file=sys.stderr)
    print(f"ORDIVYR Lite {__version__} · {box.profile} · {box.root}", file=sys.stderr)
    print(f"Model: {router.primary.label()} · validation: {validator.mode}", file=sys.stderr)
    print(box.permission_summary(), file=sys.stderr)
    if args.resume and prior_permissions and prior_permissions != box.permission_manifest():
        conversation.append({"role": "user", "content": box.permission_refresh_message()})
    print("Type /help for commands, /exit to leave.", file=sys.stderr)
    while True:
        try:
            line = input("\nlite> ").strip()
        except EOFError:
            print(file=sys.stderr)
            break
        except KeyboardInterrupt:
            print(file=sys.stderr)
            break
        if not line:
            continue
        if line in ("/exit", "/quit", "exit", "quit"):
            break
        if line == "/help":
            print("  /exit  leave   /cost  session totals   /clear  drop conversation\n"
                  "  /tools show tools   /session  show session path\n"
                  "  /status connection and permissions   /models fetch model IDs\n"
                  "  /model PROVIDER:MODEL switch model (keeps conversation)\n"
                  "  /connect configure a provider   /new start a fresh session", file=sys.stderr)
            continue
        if line in ("/status", "/model"):
            print(f"Model: {router.primary.label()}\nEndpoint: {router.primary.base_url}\n"
                  f"Validation: {validator.mode}\n{box.permission_summary()}", file=sys.stderr)
            continue
        if line == "/models":
            try:
                names = ModelClient(router.primary).list_models(timeout=10)
                print("\n".join(names[:100]) or "No models listed by this endpoint.", file=sys.stderr)
            except LiteError:
                print("Model listing failed. Check endpoint, credentials and access with /connect.", file=sys.stderr)
            continue
        if line.startswith("/model "):
            if resume_state is not None:
                print("Resume or clear the interrupted task before switching models.", file=sys.stderr)
                continue
            try:
                router, args = _switch_model(router, args, line[7:].strip(), on_event)
                print(f"Model: {router.primary.label()}. Existing conversation will be sent to this provider.", file=sys.stderr)
            except LiteError as exc:
                print(f"Model unchanged: {exc}", file=sys.stderr)
            continue
        if line == "/connect":
            from .onboarding import configure
            try:
                if configure():
                    cfg = load_config()
                    default = cfg["default_model"]
                    provider, model = ((default["provider"], default["model"]) if isinstance(default, dict)
                                       else default.split("/", 1))
                    if resume_state is None:
                        router, args = _switch_model(router, args, f"{provider}:{model}", on_event)
                        print(f"Connected: {router.primary.label()}. Existing conversation is retained.", file=sys.stderr)
                    else:
                        print("Saved for next session; interrupted task keeps its existing connection.", file=sys.stderr)
            except (LiteError, ValueError, EOFError) as exc:
                print(f"Connection setup stopped; current model unchanged: {exc}", file=sys.stderr)
            except KeyboardInterrupt:
                print("\nConnection setup cancelled; current model unchanged.", file=sys.stderr)
            continue
        if line == "/cost":
            t = router.ledger.totals()
            print(f"calls={t.calls} tokens={t.input_tokens + t.cache_read_tokens + t.cache_write_tokens + t.output_tokens} "
                  f"cost={human_usd(t.cost_usd)} cache={t.cache_hit_rate:.0%}", file=sys.stderr)
            continue
        if line == "/clear":
            conversation = []
            resume_state = None
            print("conversation cleared", file=sys.stderr)
            continue
        if line == "/new":
            conversation, resume_state = [], None
            if session:
                session.close()
                session = Session.create()
                session.set_metadata(permissions=box.permission_manifest())
            print("New conversation and session. Previous journal preserved.", file=sys.stderr)
            continue
        if line == "/tools":
            print(", ".join(box.names()), file=sys.stderr)
            continue
        if line == "/session":
            print(str(session.path) if session else "session disabled", file=sys.stderr)
            continue
        if line.startswith("/"):
            print("Unknown command. Type /help. No model request was made.", file=sys.stderr)
            continue
        try:
            streamed_state = getattr(user_printer, "streamed", {})
            streamed_state["text"] = ""
            runner = Runner(router, box, line, system=_read_system(args), limits=limits,
                            validator=validator, session=session, on_event=on_event,
                            stream=_stream_enabled(args),
                            run_id=(session.run_id if session else None),
                            initial_messages=(conversation or None) if resume_state is None else None,
                            resume_state=resume_state)
            result = runner.run()
            resume_state = None
        except KeyboardInterrupt:
            print("\n[interrupted]", file=sys.stderr)
            continue
        except LiteError as exc:
            print(f"Task stopped: {exc}. Use /status, /connect or /new to continue.", file=sys.stderr)
            continue
        conversation = result.messages
        streamed = getattr(user_printer, "streamed", {}).get("text") or ""
        if not streamed.strip() or streamed.strip() != (result.text or "").strip():
            print(result.text)
        print(_summary(result), file=sys.stderr)
    box.close()
    if events_file and hasattr(events_file, "close"):
        events_file.close()
    if session:
        session.close()
    return 0


# --------------------------------------------------------------------------- job
def cmd_job(args) -> int:
    from .job import example_job, job_schema, run_job, validate_job

    if args.job_command == "schema":
        print(json.dumps(job_schema(), indent=2))
        return 0
    if args.job_command == "example":
        print(json.dumps(example_job(args.workdir or "."), indent=2))
        return 0
    if not args.file:
        raise ConfigError("job requires --file (use - for stdin)")
    try:
        raw = sys.stdin.read() if args.file == "-" else Path(args.file).expanduser().read_text("utf-8")
        data = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ConfigError(f"cannot read job: {exc}") from exc
    if args.job_command == "validate":
        job = validate_job(data)
        print(json.dumps({"valid": True, "job": job}, indent=2, default=str))
        return 0
    # run
    if args.max_cost:
        data.setdefault("spend", {})["max_cost_usd"] = args.max_cost
        data.setdefault("limits", {})["max_cost_usd"] = args.max_cost
    events = _emit_events_file(args.events)
    result = run_job(data, on_event=events, allow_paid=args.allow_paid)
    if events and hasattr(events, "close"):
        events.close()
    text = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).expanduser().write_text(text + "\n", "utf-8")
    print(text)
    return exit_code_for(result.get("stop_reason"))


# --------------------------------------------------------------------------- config
def cmd_config(args) -> int:
    cfg = load_config()
    sub = args.config_command
    if sub == "path":
        print(cfg["_path"])
        return 0
    if sub == "show":
        print(json.dumps(config_summary(cfg), indent=2))
        return 0
    if sub == "init":
        path = Path(cfg["_path"])
        if path.exists() and not args.force:
            raise ConfigError(f"config already exists at {path}; use --force to overwrite")
        save_config(cfg)
        print(f"wrote {path}")
        return 0
    if sub == "set-key":
        key = args.key
        if args.stdin:
            key = sys.stdin.read().strip()
        elif args.key_file:
            key = Path(args.key_file).expanduser().read_text("utf-8").strip().splitlines()[0]
        if not key:
            raise ConfigError("provide --key, --key-file or --stdin")
        path = store_provider_key(args.provider, key)
        print(f"stored key for {args.provider} in {path} (0600)")
        return 0
    if sub == "set-price":
        prices = cfg.setdefault("prices", {})
        prices[args.model] = {"input": args.input, "output": args.output}
        if args.cache_read is not None:
            prices[args.model]["cache_read"] = args.cache_read
        if args.cache_write is not None:
            prices[args.model]["cache_write"] = args.cache_write
        save_config(cfg)
        print(f"priced {args.model}")
        return 0
    if sub in ("authorize", "revoke"):
        providers = cfg.setdefault("providers", {})
        entry = providers.setdefault(args.provider, {})
        if sub == "revoke":
            entry["authorized"] = False
            save_config(cfg)
            print(f"revoked {args.provider}")
            return 0
        if not (args.daily > 0 and args.total > 0):
            raise ConfigError("--daily and --total must be positive USD values")
        entry["authorized"] = True
        entry["caps"] = {"daily_usd": args.daily, "total_usd": args.total,
                         "per_request_usd": args.per_request or args.daily}
        save_config(cfg)
        print(f"authorized {args.provider}: daily ${args.daily}, total ${args.total}, "
              f"per-request ${entry['caps']['per_request_usd']}")
        return 0
    if sub == "set-default":
        spec = args.model
        if ":" in spec:
            provider, model = spec.split(":", 1)
        elif "/" in spec:
            provider, model = spec.split("/", 1)
        else:
            raise ConfigError("default must be PROVIDER:MODEL")
        cfg["default_model"] = f"{provider}/{model}"
        save_config(cfg)
        print(f"default model: {provider}/{model}")
        return 0
    if sub == "providers":
        print(json.dumps(config_summary(cfg)["providers"], indent=2))
        return 0
    raise ConfigError("unknown config command")


def cmd_init(args) -> int:
    """One model binding, existing configuration format, no inference/spending."""
    from .config import default_config
    path = config_home() / 'config.json'
    if path.exists():
        raise ConfigError(f'config already exists at {path}; inspect it with config show')
    cfg = default_config()
    provider = args.provider or 'custom'
    cfg['providers'][provider] = {'base_url': args.base_url, 'protocol': args.protocol,
                                  'context_window': args.context_window}
    if args.api_key_file:
        key = Path(args.api_key_file).expanduser().resolve()
        if not key.is_file():
            raise ConfigError('API key file does not exist')
        cfg['providers'][provider]['key_file'] = str(key)
    cfg['default_model'] = {'provider': provider, 'model': args.model}
    # Validate URL and local credentials; a cloud route remains unauthorized
    # until the operator uses the existing explicit authorization command.
    from urllib.parse import urlparse
    url = urlparse(args.base_url)
    if url.scheme not in ('http', 'https') or not url.hostname or url.username or url.password:
        raise ConfigError('base URL must be http(s) without inline credentials')
    if url.hostname not in ('127.0.0.1', 'localhost', '::1') and url.scheme != 'https':
        raise ConfigError('remote endpoint requires HTTPS')
    save_config(cfg)
    print(json.dumps({'configured': str(path), 'model': args.model,
                      'next': 'ordivyr-lite run --workdir PATH "TASK"'}, indent=2))
    return 0


# --------------------------------------------------------------------------- doctor / models / usage / sessions
def cmd_doctor(args) -> int:
    cfg = load_config()
    report = doctor_report(cfg)
    if args.probe:
        report["local_probes"] = [probe_local(p["base_url"]) for p in report.get("local_presets", [])
                                  if p.get("base_url")]
    print(json.dumps(report, indent=2))
    ok = report.get("ok") and (not args.probe or any(p.get("reachable") for p in report.get("local_probes", [])))
    return 0 if ok else 1


def cmd_models(args) -> int:
    cfg = load_config()
    route = _route(cfg, args)
    names = ModelClient(route).list_models()
    print(json.dumps({"provider": route.provider, "base_url": route.base_url,
                      "models": names, "count": len(names)}, indent=2))
    return 0


def cmd_usage(args) -> int:
    ledger = UsageLedger()
    since = parse_since(args.since)
    report = ledger.report(since)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(format_report(report))
    return 0


def cmd_sessions(args) -> int:
    if args.sessions_command == "list":
        print(json.dumps(list_sessions(), indent=2))
        return 0
    if args.sessions_command == "show":
        s = Session.open(args.path)
        print(json.dumps(s.info(), indent=2))
        return 0
    if args.sessions_command == "prune":
        removed = prune_sessions(args.keep)
        print(json.dumps({"removed": [str(p) for p in removed]}, indent=2))
        return 0
    raise ConfigError("unknown sessions command")


def cmd_resume(args) -> int:
    args.session = args.path
    args.resume = True
    if not hasattr(args, "no_session"):
        args.no_session = False
    return cmd_run(args)


# --------------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ordivyr-lite",
        description="ORDIVYR Lite - a terminal-first, standalone AI execution engine.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--home", help="state directory (default ~/.ordivyr-lite)")
    p.add_argument("--debug", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    native = sub.add_parser('native', help='open an installed native CLI with its own login and controls')
    native.add_argument('client', choices=('claude','codex','gemini'))
    native.add_argument('--workdir', default='.')
    native.add_argument('--check', action='store_true', help='show availability without starting a CLI or model request')
    init = sub.add_parser('init', help='configure one local or cloud model without a model request')
    init.add_argument('--model', required=True); init.add_argument('--base-url', required=True)
    init.add_argument('--provider', default='custom'); init.add_argument('--api-key-file')
    init.add_argument('--protocol', choices=('openai', 'anthropic'), default='openai')
    init.add_argument('--context-window', type=int, default=32768)
    run = sub.add_parser("run", help="run one task")
    run.add_argument("prompt", nargs="?", help="task text (or stdin / --prompt-file)")
    run.add_argument("-f", "--prompt-file", help="read the task from a file (- for stdin)")
    run.add_argument("--workdir", default=".", help="working directory (default .)")
    run.add_argument("--system", help="system prompt override")
    run.add_argument("--system-file", help="read the system prompt from a file")
    run.add_argument("--validate", choices=("off", "syntax", "auto", "tests"), help="validator mode")
    run.add_argument("--check", help="validator command to run on completion")
    run.add_argument("--scope", action="append", metavar="PATH",
                     help="restrict file writes to this relative path prefix (repeatable)")
    run.add_argument("--plan", action="store_true",
                     help="expose update_plan so the model tracks milestones (opt-in; changes the tool list)")
    run.add_argument("--validate-timeout", type=int)
    run.add_argument("--session", help="session journal path")
    run.add_argument("--no-session", action="store_true", help="do not persist a session")
    run.add_argument("--resume", action="store_true", help="resume the latest/selected session")
    _common_model_args(run)
    _common_limit_args(run)
    _common_permission_args(run)
    _common_output_args(run)

    chat = sub.add_parser("chat", aliases=["start", "setup", "connect"],
                         help="interactive terminal session; setup/connect guide model connection")
    chat.add_argument("--no-chat", action="store_true", help="configure without starting chat")
    chat.add_argument("--workdir", default=".", help="working directory (default .)")
    chat.add_argument("--system", help="system prompt override")
    chat.add_argument("--system-file", help="read the system prompt from a file")
    chat.add_argument("--validate", choices=("off", "syntax", "auto", "tests"), help="validator mode")
    chat.add_argument("--check", help="validator command to run on completion")
    chat.add_argument("--validate-timeout", type=int)
    chat.add_argument("--session", help="session journal path")
    chat.add_argument("--no-session", action="store_true", help="do not persist a session")
    chat.add_argument("--resume", action="store_true", help="resume the latest/selected session")
    _common_model_args(chat)
    _common_limit_args(chat)
    _common_permission_args(chat)
    _common_output_args(chat)

    job = sub.add_parser("job", help="versioned JSON job interface")
    jsub = job.add_subparsers(dest="job_command", required=True)
    jrun = jsub.add_parser("run", help="run a job file")
    jrun.add_argument("--file", "-f", required=True, help="job JSON file (- for stdin)")
    jrun.add_argument("--output", help="write the result JSON here too")
    jrun.add_argument("--events", help="write JSONL progress events here")
    jrun.add_argument("--allow-paid", action="store_true", help="authorise paid providers for this job")
    jrun.add_argument("--max-cost", type=float, default=0.0, help="override the job cost cap")
    jval = jsub.add_parser("validate", help="validate a job file")
    jval.add_argument("--file", "-f", required=True)
    jsub.add_parser("schema", help="print the job JSON schema")
    jex = jsub.add_parser("example", help="print an example job")
    jex.add_argument("--workdir", default=".")

    cfgp = sub.add_parser("config", help="configuration")
    csub = cfgp.add_subparsers(dest="config_command", required=True)
    csub.add_parser("show", help="print the redacted config")
    cinit = csub.add_parser("init", help="write the default config")
    cinit.add_argument("--force", action="store_true")
    csub.add_parser("path", help="print the config path")
    csub.add_parser("providers", help="list provider presets/configuration")
    ckey = csub.add_parser("set-key", help="store a provider key in secrets.json (0600)")
    ckey.add_argument("provider")
    group = ckey.add_mutually_exclusive_group(required=True)
    group.add_argument("--key")
    group.add_argument("--key-file")
    group.add_argument("--stdin", action="store_true")
    cprice = csub.add_parser("set-price", help="set a paid model price (USD per million tokens)")
    cprice.add_argument("model")
    cprice.add_argument("--input", type=float, required=True)
    cprice.add_argument("--output", type=float, required=True)
    cprice.add_argument("--cache-read", type=float)
    cprice.add_argument("--cache-write", type=float)
    cauth = csub.add_parser("authorize", help="authorise a paid provider with caps")
    cauth.add_argument("provider")
    cauth.add_argument("--daily", type=float, required=True)
    cauth.add_argument("--total", type=float, required=True)
    cauth.add_argument("--per-request", type=float)
    crev = csub.add_parser("revoke", help="revoke a paid provider")
    crev.add_argument("provider")
    cdef = csub.add_parser("set-default", help="set default_model = PROVIDER:MODEL")
    cdef.add_argument("model")

    doc = sub.add_parser("doctor", help="environment and provider checks")
    doc.add_argument("--probe", action="store_true", help="probe local endpoints (HTTP, no model call)")
    doc.add_argument("--json", action="store_true", help="JSON output (the default)")

    mod = sub.add_parser("models", help="list models for a provider (token-free)")
    _common_model_args(mod)

    use = sub.add_parser("usage", help="token/cache/cost report from the ledger")
    use.add_argument("--since", help="e.g. 24h, 7d, today")
    use.add_argument("--json", action="store_true")

    sess = sub.add_parser("sessions", help="session journals")
    ssub = sess.add_subparsers(dest="sessions_command", required=True)
    ssub.add_parser("list")
    sshow = ssub.add_parser("show")
    sshow.add_argument("path")
    sprune = ssub.add_parser("prune")
    sprune.add_argument("--keep", type=int, default=20)

    res = sub.add_parser("resume", help="resume a session (alias for run --resume --session)")
    res.add_argument("path")
    res.add_argument("prompt", nargs="?", help="optional new instruction appended to the session")
    res.add_argument("--workdir", default=".")
    res.add_argument("--system")
    res.add_argument("--system-file")
    res.add_argument("--validate", choices=("off", "syntax", "auto", "tests"))
    res.add_argument("--check")
    res.add_argument("--validate-timeout", type=int)
    res.add_argument("--no-session", action="store_true")
    _common_model_args(res)
    _common_limit_args(res)
    _common_permission_args(res)
    _common_output_args(res)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        if not sys.stdin.isatty():
            parser.print_help()
            return 0
        argv = ["start"]
    args = parser.parse_args(argv)
    try:
        _set_home(getattr(args, "home", None))
        if args.command == 'native':
            from .native import launch
            return launch(args.client,args.workdir,check=args.check)
        if args.command == "run":
            return cmd_run(args)
        if args.command in ("chat", "start", "setup", "connect"):
            return cmd_start(args)
        if args.command == "job":
            return cmd_job(args)
        if args.command == 'init':
            return cmd_init(args)
        if args.command == "config":
            return cmd_config(args)
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "models":
            return cmd_models(args)
        if args.command == "usage":
            return cmd_usage(args)
        if args.command == "sessions":
            return cmd_sessions(args)
        if args.command == "resume":
            return cmd_resume(args)
        parser.error("unknown command")
        return 2
    except LiteError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "debug", False):
            import traceback
            traceback.print_exc()
        else:
            print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 8


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
