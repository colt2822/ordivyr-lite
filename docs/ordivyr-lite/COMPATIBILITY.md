# ORDIVYR Lite compatibility contract

Lite exposes one versioned, machine-readable interface for Board (or any other
orchestrator): a **job** goes in, a **result** comes out. Lite never calls back
into Board, never opens Board's database, and never requires a Board process to
be running.

```
Board (orchestration, teams, scheduling, memory, hosting, billing)
        │
        │  job JSON (stdin/file)                         result JSON (stdout/file)
        ▼
  ordivyr-lite job run  ──►  execution loop  ──►  files / sandbox / model routes
        │
        └── session journal (optional, for crash recovery/resume)
```

## Boundary

Lite owns:

- the execution loop (model turns, tool calls, breakers, validation/repair);
- context management (append-only history, compaction, cache-friendly prefix);
- tools and permissions (file tools, sandboxed shell, web fetch);
- model routing, BYOK, output limits and fail-closed spend caps;
- recovery (session journal, resume);
- transparent usage/cost accounting.

Board owns: orchestration, teams, scheduling, durable memory, hosting, billing,
UI, and decisions about which jobs to launch. Lite has no Board imports and no
Board code was copied into it. Board should integrate by launching the CLI (or
calling `ordivyr_lite.job.run_job`) — not by re-implementing the loop.

## Versioning

- Job protocol: `"protocol": "ordivyr-lite/job"`, `"version": 1`
- Result protocol: `"protocol": "ordivyr-lite/result"`, `"version": 1`
- `JOB_PROTOCOL_VERSION` is `1`. A mismatch is rejected with
  `unsupported protocol version`; the CLI exits `2` and writes no result.
- Unknown job fields are rejected in v1. Adding a field requires a protocol
  version bump (or a compatible v1.1 negotiation in a future release).
- `lite_version` in the result identifies the implementing release.

Machine-readable schemas: [`job.schema.json`](job.schema.json) and
[`result.schema.json`](result.schema.json). A working example is in
[`example-job.json`](example-job.json).

## Job schema (v1)

```json
{
  "protocol": "ordivyr-lite/job",
  "version": 1,
  "job_id": "board-job-42",
  "created_at": "2026-11-07T12:00:00Z",
  "task": {
    "prompt": "Fix the failing tests and do not touch src/legacy.py.",
    "workdir": "/srv/work/job-42",
    "system": null
  },
  "model": {
    "provider": "llamacpp",
    "model": "local-3060",
    "base_url": "http://127.0.0.1:8181/v1",
    "api_key_env": "LOCAL_LLM_KEY",
    "context_window": 16384,
    "timeout_s": 120,
    "max_retries": 2
  },
  "limits": {
    "max_turns": 64,
    "max_tool_calls": 400,
    "max_seconds": 1800,
    "max_cost_usd": 0,
    "max_tokens": 0,
    "max_repairs": 2,
    "output_tokens": 16384,
    "compact_at": 48000
  },
  "permissions": {
    "profile": "workspace",
    "tools": null,
    "sandbox": "auto",
    "network": false,
    "allow_shell": null,
    "allow_web": null
  },
  "validation": {"mode": "auto", "command": null, "timeout_s": 120},
  "session": {"enabled": true, "path": null, "resume": false},
  "spend": {"allow_paid": false, "max_cost_usd": 0},
  "metadata": {"board": "engineering", "team": "backend"}
}
```

### Field notes

- `task.prompt` (required) — the task text. `task.workdir` defaults to the
  process cwd; it must exist. `task.system` overrides Lite's default system
  prompt.
- `model` — `provider` and `model` are required in practice; `base_url` may be
  supplied instead of a known provider. Secrets must be referenced via
  `api_key_env`/`api_key_file`; **inline `api_key`/`key`/`token` fields are
  rejected**.
- `limits` — all optional. `max_tokens` is a **fresh-token** budget (cached
  reads are reported, not charged). `max_cost_usd` is enforced against both the
  per-run estimate and actual cost.
- `permissions.profile` — `read-only`, `workspace`, `trusted`, `full`.
  Omitted `allow_shell`/`allow_web` inherit the profile. Explicit booleans
  enable or deny the individual capability. `read-only` cannot be widened;
  `workspace` plus `allow_shell: true` is a scoped sandboxed shell grant.
- `permissions.sandbox` — `auto` (bwrap if available), `bwrap`, `process`
  (explicitly non-isolated), `none` (unsafe; operator only).
- `validation.mode` — `off`, `syntax`, `auto` (syntax + detected Python tests),
  `tests`, `command`. Failed validation is fed back to the model for at most
  `max_repairs` rounds; the same normalised failure twice stops with
  `repeat_failure`.
- `session` — Lite writes a JSONL journal under `$ORDIVYR_LITE_HOME/sessions`
  or `session.path`. `resume: true` continues from the latest snapshot. The
  journal is Lite-local; Board does not need to read it.
- `spend.allow_paid` — advisory metadata. **A job cannot authorise paid
  spending by itself.** The invoking process must pass `--allow-paid` with a
  positive `--max-cost` (or run against a local endpoint), otherwise the job is
  refused before any network I/O. This is deliberate: a queue entry must not
  silently spend the owner's money.

## Result schema (v1)

```json
{
  "protocol": "ordivyr-lite/result",
  "version": 1,
  "lite_version": "0.1.0",
  "job_id": "board-job-42",
  "created_at": "2026-11-07T12:00:41Z",
  "status": "succeeded",
  "stop_reason": "done",
  "error": null,
  "text": "Fixed the failing tests; 7 passed.",
  "provider": "llamacpp",
  "model": "local-3060",
  "route": "llamacpp/local-3060",
  "cost_usd": 0.0,
  "usage": {
    "input_tokens": 1200,
    "cache_read_tokens": 18000,
    "cache_write_tokens": 0,
    "output_tokens": 430,
    "context_tokens": 19200,
    "peak_context_tokens": 19200,
    "billable_tokens": 1630,
    "total_tokens": 19630,
    "cache_hit_rate": 0.9375,
    "model_calls": 9,
    "tool_calls": 14,
    "tool_errors": 0,
    "tool_success_rate": 1.0
  },
  "stats": { "...": "full counters, compactions, repairs, routes, latency" },
  "completion": {
    "model_said_done": true,
    "code_written": true,
    "syntax_ok": true,
    "tests_passed": true,
    "independent_review": null
  },
  "tool_events": [{"turn": 1, "tool": "read_file", "ok": true, "duration_ms": 0.4}],
  "artifacts": ["src/fix.py"],
  "session": {"path": "/home/u/.ordivyr-lite/sessions/board-job-42.jsonl",
               "run_id": "board-job-42", "resume": false},
  "limits": {"max_turns": 64},
  "duration_ms": 41234.5,
  "started_at": "2026-11-07T12:00:00Z",
  "ended_at": "2026-11-07T12:00:41Z"
}
```

`status` is one of `succeeded`, `failed`, `cancelled`, `refused`.
`stop_reason` is the precise reason: `done`, `steps`, `tool_calls`, `time`,
`budget`, `cost`, `malformed`, `repeat_failure`, `unverified`, `cancelled`,
`auth`, `rate_limit`, `quota`, `network`, `timeout`, `bad_request`, `server`,
`paid_not_authorized`, `unpriced_model`, `provider_error`.

`completion` is deliberately not a boolean: `tests_passed` is `null` unless a
test/command validator actually ran, and `independent_review` is `null` because
Lite does not perform one. Do not infer more than the fields say.

## CLI exit codes

| code | meaning |
|---|---|
| 0 | `done` (succeeded) |
| 2 | invalid usage/config/job (no model call) |
| 3 | permission or sandbox refusal |
| 4 | provider/model failure (auth, network, rate limit, server) |
| 5 | budget/cost cap reached |
| 6 | incomplete (malformed loop, repeat failure, unverified, turn/tool/time limit) |
| 7 | cancelled |
| 8 | internal error |

For jobs, the result JSON is still written for execution failures that happen
after validation; the exit code mirrors `stop_reason`.

## Board integration sequence

```python
import json, subprocess

job = {
    "protocol": "ordivyr-lite/job", "version": 1,
    "job_id": f"board-{task_id}",
    "task": {"prompt": mission, "workdir": str(workdir)},
    "model": {"provider": "llamacpp", "model": "local-3060",
              "base_url": "http://127.0.0.1:8181/v1"},
    "permissions": {"profile": "workspace"},
    "validation": {"mode": "auto"},
    "session": {"enabled": True},
}
p = subprocess.run(
    ["ordivyr-lite", "job", "run", "-f", "-"],
    input=json.dumps(job), text=True, capture_output=True, timeout=1900,
)
result = json.loads(p.stdout)
if result["status"] != "succeeded":
    ...
```

- Run one job per workdir; Lite does not coordinate parallel jobs.
- A crash/retry should pass the same `job_id` and a `session.path` (or let Lite
  choose one) with `resume: true`; Lite resumes from the last completed turn.
- Board should treat `usage`/`cost_usd` as authoritative for the job and may
  reconcile them with its own billing ledger.
- No production Board deployment was changed by this branch; this contract is
  the migration seam.

## Additive changes in 0.2.0

The protocols above are unchanged: a v1 job is still a v1 job and a v1 result is
still a v1 result. What 0.2.0 adds is additive and safe to ignore:

- CLI: `--no-stream` on `run`/`chat`/`job run` (streaming is the default).
- Result/stat fields: `stats.first_token_ms`, `stats.streamed_turns`,
  `stats.recovered_calls`, `stats.reused_reads`, `stats.skipped_validations`,
  `stats.state_changed_turns`. Consumers that whitelist keys must add them.
- Event stream: a new `delta` event (`{"type":"delta","kind":"text"|"reasoning"|"tool_call","text":...}`)
  when a caller passes an event sink. It is not written to the session journal.
- Ledger: paid calls now write a `reserved` row before the request and a `settled`
  row after it, so `usage` and the caps are correct with concurrent processes.
  Rows without `reserve_id` are unchanged; `usage report` ignores lifecycle rows.
- Environment: `ORDIVYR_LITE_STREAM` (`0`/`1`) overrides the streaming default.

## Compatibility checks

Run `ordivyr-lite job schema` for the machine-readable JSON Schema and
`ordivyr-lite job example` for a working example. `job validate` performs the
same validation as `job run` without touching a model.
