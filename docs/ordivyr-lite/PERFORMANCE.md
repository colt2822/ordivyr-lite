# Performance notes and remaining bottlenecks

Measured on the development host (12 threads, Linux, Python 3.12) against the
existing authorized local llama.cpp `qwen2.5-7b-instruct` server (port 8181,
16k context, sleep-idle 45 s) and a frozen copy of 0.1.0 as the baseline. No
paid API was used. Read the (unpublished) benchmark records for the tables and the
limitations before quoting any number.

## What 0.2.0 changed, and what it measured

| Lever | Mechanism | Measured effect on this host |
|---|---|---|
| Streaming | SSE on both protocols, per-index tool-call assembly, truncation that executes nothing, TTFT per turn | First character of the answer 1.53 s instead of 2.62 s (42% sooner) with the same build and prompt; identical trajectories with streaming on and off |
| Warm sandbox | one bwrap session per process, framed commands | 6.72 -> 3.91 ms per command (41.8%, 1.72x); break-even 4 commands. Inside a real ToolBox: 6.8 -> 4.6 ms |
| Exact edits | four-stage unique-match ladder | 10/14 -> 11/14 deterministic corpus cases, 0 unsafe writes, all 3 ambiguous requests still refused |
| Redundant reads | run-local path-signature cache | `stats.reused_reads`; a repeated read of an unchanged file no longer touches the filesystem tool at all |
| Redundant validation | reuse by file signature | `stats.skipped_validations`; a repair loop does not run the same failing suite twice |
| Truncated answers | smaller-budget retry | a cut-off answer is retried shorter instead of being accepted as final |
| Output allocation | per-turn budget from the context window and the remaining token budget | fewer provider `token_limit` rejections, each of which costs a whole call |
| Tokenizer | family-aware estimate, optional real BPE, calibration from observed counts, cache-block alignment | full count 3.8 ms at 40k chars / 15 ms at 120k / 41 ms at 400k; a per-message memo makes a turn cost O(new messages) |
| Cache accounting | llama.cpp `timings.cache_n`/`prompt_n` | local cache-hit numbers are measured, not derived |
| Spend safety | atomic reservation rows | 6 concurrent processes, exactly one reservation is allowed (test) |
| Harness overhead | memo + cheapest paths on every turn | 20-turn stub session +7.7% CPU vs 0.1.0 (was +97% before the memo); `--version` -2.3%; CLI import -1.1% |

Streaming is an interactivity win, not a generation-time win: the server still
produces the same tokens, so a wall-clock comparison of whole runs is unaffected
by streaming itself.

## Where the local-model numbers stand

The task campaign uses the local 7B model on 14 tasks with 2 attempts per engine.
Pass-rate differences of one or two tasks are inside the noise of that sample,
and a failing run usually costs more turns than a passing one, so tokens per
attempt move with the pass rate. What the campaign does show, on the same task,
prompt, model and host, is that 0.2.0 does not trade tokens or latency for its
new features, and that tool-call reliability improved slightly. See the deltas
in the (unpublished) benchmark records.

## Remaining bottlenecks

1. **Local model capability dominates.** With a 7B model on multi-file tasks,
   most failures are model errors, not harness errors. A frontier or a larger
   coding model is the next measurement, and it needs paid authorization.
2. **No paid/frontier measurement.** Pass@1/Pass@3 and cost per successful task
   for cloud models are **NOT MEASURED**. Costs in this release are a price
   proxy over measured token counts.
3. **Token counting is still an estimate.** The family tables are calibrated
   from provider-reported counts, but only an exact tokenizer
   (`tokenizer.json`) makes the pre-call budget exact; cap checks
   deliberately take the larger of two independent estimates.
4. **Session snapshots are full-history JSON.** After every completed turn the
   journal rewrites the whole message list. At 48k tokens that is a few hundred
   KB per turn; incremental snapshots remain the next step.
5. **Compression is lexical, not semantic.** Old tool results are ranked by a
   local lexical index against the task text. It preserves the ones the task is
   about, but it cannot tell that two differently-worded results mean the same
   thing.
6. **No cross-run KV pinning.** Prefix reuse is what the provider gives us while
   its slot stays warm (`--sleep-idle-seconds 45` on this host). Lite keeps the
   prefix byte-stable and asks for cache use, but it cannot hold a remote KV
   cache open between runs.
7. **Warm-sandbox capture divergence.** A single command that writes more than
   the 100 MB file-size rlimit to stdout fails in the warm-session path where the
   one-shot pipe path would clip it. The session is stricter, never looser, and
   the divergence is documented in SECURITY.md.
8. **Routing is primary-first.** Cache awareness lives in the request (stable
   prefix, `prompt_cache_key`, `cache_prompt`, block-aligned budgets), not in choosing
   another provider: a different provider cannot reuse another's cache.
9. **Tool output caps are per result, not per turn.** A turn that calls ten
   tools still appends ten results; compaction only bounds the total under
   pressure.
10. **No speculative tool execution.** Read-only prefetch could hide latency but
    risks wasted work and duplicate side effects; not attempted.

## Next measurements that need authorization

The biggest unknown remains frontier-model quality per dollar: the same paired
design on an authorized paid model (with real prices and cache telemetry), on
the dev and held-out suites, at an attempt count large enough to separate a
one-task difference from noise. Until then, paid-model Pass@k and cost per
successful task are **NOT MEASURED**.
