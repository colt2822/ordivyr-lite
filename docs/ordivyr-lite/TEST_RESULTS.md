# Test results

Host: Linux, Python 3.12.3, 12 CPU threads. Commands run from the
`feature/ordivyr-lite` worktree on 2026-11-07.

## ORDIVYR Lite

```bash
python3 -m pytest ordivyr_lite/tests -o addopts=""
```

**286 passed in 22 s** (0.1.0: 98 passed in 7.50 s). The suite grew
with streaming, tokenizer, memory, sandbox-session, edit-ladder, efficiency and
benchmark-contract coverage.

Coverage includes:

- config/BYOK/pricing/paid-authorization/output limits;
- OpenAI + Anthropic wire parsing, malformed arguments, truncation,
  retries, cache telemetry and cache breakpoints;
- paginated read, exact/tolerant/CRLF edit, symlink and path confinement,
  grep/find, spill;
- permission profiles, tool validation, bwrap/process sandbox, timeout, SSRF
  and session round-trip;
- 20-turn × 3-tool sessions, malformed-call refusal, truncated-turn refusal,
  validator repair loops, repeated-failure breaker, fresh-token budget,
  tool-call limit, duplicate pointers, compaction/checkpoint, output-limit
  halving, session resume;
- job protocol validation (version, inline-secret rejection, unknown fields,
  result shape, paid-spend refusal, resume);
- router fallback and paid-fallback guard, spend caps;
- CLI end-to-end (run, chat, job, usage, doctor, config, sessions, resume,
  dry-run, fast version path);
- streaming: SSE frame parsing, incremental deltas, tool-call fragments by
  index, cut-stream refusal (also at loop level), Anthropic events, usage and
  llama.cpp timing accounting, capability degradation, text tool-call recovery
  and its safety rule, stale keep-alive reconnect, one capability probe per
  route;
- tokenizer: family detection, bands, monotonicity, cache-block alignment,
  calibration, corrupt cache, real-BPE absence, import cost;
- memory: ranking, incremental refresh, cache reload, symlink/oversize/binary
  refusal, compression that protects system + recent messages, context store
  relevance/bounds/corruption/cross-root isolation;
- efficiency: reservations (single process, expiry, settlement, six concurrent
  processes), cap estimate conservatism, read dedupe and invalidation,
  validation reuse, cut-answer retry, output allocation, sandbox session reuse,
  relevance-ordered compression;
- edits: 32 cases across the four-stage ladder, ambiguity refusal at every
  stage, CRLF/LF preservation, no-write-on-failure, 1 MB timing;
- sandbox session: confinement argv parity, env/cwd reset, sentinel injection,
  timeout, no leftover children, fallback, one-session guard;
- benchmark contract: worker/tasks CLI, cost arithmetic, stub SSE, report
  rendering, engine fingerprints.

## Existing optimized harness (regression)

```bash
python3 -m pytest ordivyr/tests -p no:cacheprovider
```

**407 passed, 23 skipped in 144.15 s** (re-run for 0.2.0; identical to the
0.1.0 baseline). The 23 skips are pre-existing
environmental skips; no original `ordivyr/` runtime file was modified by this
branch.

Targeted subset (agent loop, perf passes, gateway, boards, security/enforcement,
agent jail):

```bash
python3 -m pytest ordivyr/tests/test_agentloop.py ordivyr/tests/test_perf.py \
  ordivyr/tests/test_perf_pass1.py ordivyr/tests/test_perf_pass2.py \
  ordivyr/tests/test_perf_pass3.py tests/test_selfhost_gateway.py \
  tests/test_selfhost_boards.py tests/test_security_sweep.py \
  tests/test_enforcement_sweep.py tests/test_agent_jail.py -p no:cacheprovider
```

**151 passed in 54.36 s.**

## Installation

```bash
python3 -m venv /tmp/lite-venv
/tmp/lite-venv/bin/python -m pip install --no-deps .
/tmp/lite-venv/bin/ordivyr-lite --version
# ordivyr-lite 0.1.0
```

The package has no runtime dependencies; the wheel install pulls only the
`ordivyr_lite` package and the console script.

## Benchmark validation

| Artifact | Result |
|---|---|
| `bench/results/tasks-scripted-dev.json` | 42/42 attempts pass, both engines (14 tasks × 3) |
| `bench/results/tasks-scripted-heldout.json` | 42/42 attempts pass, both engines (held-out seed 1000) |
| `bench/results/tasks-local-smoke-pass3.json` | original 0/3 tasks, Lite 0/3 tasks (9 attempts each, exact system parity) |
| `bench/results/tasks-local-heldout8.json` | original 2/8, Lite 1/8 (8 tasks, 1 attempt) |
| `bench/results/mechanics.json` | 8 scenarios, original vs Lite |
| `bench/results/recovery.json` | SIGKILL after turn 1, resumed, final file correct |
| `bench/results/edits-baseline.json` / `-new.json` | deterministic edit corpus: 10/14 -> 11/14 applied, 0 unsafe writes, 3/3 refusals |
| `bench/results/sandbox.json` | one-shot 6.72 ms vs warm session 3.91 ms per command (41.8%) |
| `bench/results/mechanics-0.1.0.json` / `mechanics-0.2.0.json` | harness mechanics and start-up, 0.1.0 vs 0.2.0 |
| `/tmp/ordivyr-bench/final-dev.json`, `final-heldout.json` | 3-engine A/B campaign (unpublished) |

The scripted-oracle suites are saturated by design (the oracle knows the tool
sequence); they validate the harness, not model quality. The live local 7B
results are an engineering signal, not a claim about frontier models.

## Not measured

- Paid cloud models: **NOT MEASURED** (no paid API was authorized or used).
- Real provider bills, cache discounts, tokenizer-accurate billing: **NOT
  MEASURED**; local/scripted costs are $0.
- Windows/macOS, 100k+ token contexts, multi-hour sessions, concurrent jobs:
  **NOT MEASURED**.

See the (unpublished) benchmark records for the tables and limitations.
