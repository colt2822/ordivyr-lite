# ORDIVYR Lite changelog

## 0.2.0 — feature/ordivyr-lite (2026-11-08)

Same job/result protocol (v1) and the same CLI surface, plus `--no-stream`.
See PERFORMANCE.md for the measured effect and benchmark records for the raw runs.

### Added

- **True token streaming** for OpenAI-compatible and Anthropic endpoints:
  incremental text/reasoning deltas, per-index tool-call assembly, truncation
  detection that executes nothing from a cut stream, time-to-first-token per
  turn, `delta` events, and `--no-stream` / `ORDIVYR_LITE_STREAM=0`.
- **Provider adaptation** (`profiles.py`): DeepSeek (incl. reasoner text-tool
  protocol), Qwen/llama.cpp (`cache_prompt`, thinking off), Claude cache
  breakpoints, OpenAI `prompt_cache_key`/`parallel_tool_calls`, o-series
  `max_completion_tokens`; a 400-triggered degrade to the core body.
- **Tokenizer** (`tokens.py`): per-family estimation, a real byte-level BPE when a
  `tokenizer.json` is available, calibration from provider-reported counts, and
  cache-block alignment; used for budgeting, allocation and spend caps.
- **Context**: relevance-ordered compression, a persisted context store, a
  retrieval hint for larger trees, and exact llama.cpp `cache_n`/`prompt_n`
  accounting instead of an estimate.
- **Economy**: unchanged-read dedupe, validation reuse by file signature, a
  smaller-budget retry for a truncated answer, per-turn output allocation, and
  a warm bubblewrap session shared by `run_command` and the validator.
- **Spending**: atomic cross-process reservations in the usage ledger.
- **Editing**: a four-stage unique-match ladder in `edit_file` with per-stage
  counters.
- **Bench harness**: `--baseline-root`/`lite-baseline` engine, `bench/stream.py` (TTFT),
  `bench/cost.py` (price proxy), `bench/sandbox_startup.py`,
  `bench/edits.py` (deterministic edit corpus), engine fingerprints.

### Tested

- `ordivyr_lite/tests`: 285 passed (0.1.0: 98).
- Deterministic edit corpus: 10/14 → 11/14 successful edits, 0 unsafe writes,
  all ambiguous requests still refused.
- Warm sandbox session: 7.0 → 4.6 ms per command (1.49x) on this host.
- A/B task campaign against a frozen 0.1.0 tree: unpublished.

### Not measured

- Paid/frontier-model Pass@k and real bills: still NOT MEASURED (no paid API
  authorized). Local-model costs in this release are a price proxy over measured
  token counts.

## 0.1.0 — feature/ordivyr-lite (2026-11-07)

Base: `4505213` (`performance/ordivyr-harness-optimization`), worktree
`feature/ordivyr-lite`.

### Added

- Standalone `ordivyr_lite` package with no Board/web/database runtime imports.
- Terminal-first CLI: `run`, `chat`, `job`, `config`, `doctor`, `models`,
  `usage`, `sessions`, `resume`, `--json`, `--dry-run`, fast `--version`.
- Optimized single-agent loop: turns separated from tool calls, 64-turn default,
  model-aware output limits, bounded output-limit halving, fresh-token budget,
  wall/cost/tool limits.
- Reliable tool calling: typed validation before execution, malformed and
  truncated calls never run, unknown/permission-denied tools refused, parallel
  read-only calls, repeated-failure breakers.
- Context efficiency: append-only history below pressure, one-batch compaction
  with deterministic checkpoint, duplicate pointers, head+tail retention and
  task-local spill, Anthropic cache breakpoints, OpenAI/DeepSeek cache telemetry.
- Precise editing: paginated line-numbered `read_file`, exact/unique
  `edit_file` (tolerant only if unique, CRLF preserved, atomic writes),
  sandboxed `grep`/`find_files`.
- Validation/repair: syntax, detected Python tests or operator command; exact
  report fed back for at most `max_repairs`; same failure twice stops.
- Crash recovery: append-only JSONL session + per-turn snapshots; promptless
  `resume`; atomic file edits.
- Provider routing/BYOK: OpenAI-compatible and Anthropic, local/cloud presets,
  explicit fallback chain, fail-closed paid authorization, operator prices,
  per-request/daily/total caps, no silent paid fallback.
- Permissions/sandbox: read-only, workspace, trusted, full; bubblewrap shell
  with namespace/workdir/tmpfs/no-network/capability-drop/rlimits; explicit
  non-isolated process mode; SSRF-guarded `fetch_url`.
- Transparent usage: JSONL ledger, token/cache/latency/cost report, per-result
  usage and completion levels.
- Versioned `ordivyr-lite/job` → `ordivyr-lite/result` protocol (v1) with JSON
  Schema, example, exit codes and Board integration notes.
- `bench/` paired harness: original loop vs Lite on Arena dev/held-out suites,
  scripted oracle, live local 7B, mechanics and crash recovery.
- Docs: installation/usage, compatibility contract, architecture, security,
  benchmarks/results with limitations, test results, performance bottlenecks.

### Tested

- `ordivyr_lite/tests`: 98 passed.
- `ordivyr/tests`: 407 passed, 23 skipped (existing harness; no runtime file
  changed).
- Install smoke in a fresh venv.
- Scripted dev/held-out: both engines 42/42 attempts and essentially token-equal
  (+0.2% Lite). Live local 7B exact-parity: smoke 0/3 both; held-out 8 original
  2/8 vs Lite 1/8. Repeated local runs ranged 0-2/3 and 1-2/8 for both; this is
  small-sample noise, not a quality ranking. Recovery SIGKILL + resume
  succeeded.

### Not measured

- Paid/frontier-model Pass@k, real bills and cache discounts: NOT MEASURED.
- Windows/macOS, 100k+ contexts, multi-hour sessions, concurrent jobs: NOT
  MEASURED.
