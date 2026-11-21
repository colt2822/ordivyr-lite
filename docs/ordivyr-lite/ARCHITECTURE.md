# Architecture

Lite is the optimized execution loop extracted into a standalone product. The
Board's web/runtime layer is not imported. The seam between the two products is
the versioned job/result protocol, not shared process state.

## Execution loop

One iteration:

1. **Budget gate** — cancel flag, wall clock, cost cap. Build the request view
   and size the request with the calibrated tokenizer.
2. **Model turn** — one streamed request to a route. Transient failures retry
   within the route and then fall to an explicit fallback route. A provider that
   rejects the requested output length triggers bounded halving (down to 512).
3. **Account** — record fresh input, cache read, cache write, output, context,
   model latency, time-to-first-token and cost in the ledger and run stats, then
   feed the observed token count back into the tokenizer calibration.
4. **No tool calls** — run the validator. On failure, append the exact checker
   report and continue for at most `max_repairs` (same normalised failure twice
   → `repeat_failure`; still failing → `unverified`). Otherwise finish.
5. **Tool calls** — validate each call *before* execution:
   - a truncated turn executes nothing;
   - malformed JSON, non-object arguments, missing/wrong-typed required
     parameters and unknown tools are refused with a structured error;
   - permission refusals are explicit.
   Independent read-only calls run concurrently (max 8); mutating calls run in
   order.
6. **Append** the assistant message and the tool results. A duplicate result is
   replaced by a pointer (on the new copy, so earlier messages never change).
7. **Breakers** — 3 all-refused turns or an identical malformed call more than
   twice → `malformed`; an identical executed call failing 3× is nudged, 5×
   stops the run; repeated validator signature stops.
8. **Context pressure** — only when context occupancy reaches `compact_at`
   (default 48k tokens) is old tool output elided in one batch and a
   deterministic checkpoint written. Below that the history is append-only, so
   provider prefix caches can hit.
9. **Limits** — turns, tool executions, wall clock, fresh-token budget, cost.

`steps` always means model turns. `tool_calls` is counted separately. A turn
with 20 tools is one step and 20 tool executions.

## Streaming

Every request is sent with `stream: true` and assembled incrementally:

- text deltas reach the caller as they arrive (the CLI prints them; events carry
  them as `delta` and are never written to the session journal);
- reasoning deltas (`reasoning_content`, `thinking_delta`) are kept out of the answer;
- tool calls are assembled by index from argument fragments and are only
  convertible into a runnable call when the fragment stream produced valid JSON;
- a stream that ends without a finish reason, or a partial argument payload, is
  marked truncated and makes the loop execute **nothing** from that turn;
- `time-to-first-token` and total latency are recorded per turn, so streaming is
  measurable rather than asserted.

A provider that answers a streaming request with a plain JSON body is handled by
the same client (the assembler falls back), and `--no-stream` or
`ORDIVYR_LITE_STREAM=0` disables SSE entirely.

## Provider adaptation

`profiles.py` maps a route to a model family (DeepSeek, Qwen, Claude, OpenAI,
generic) and shapes the request for it:

| Family | Request shaping |
|---|---|
| DeepSeek | tools + `tool_choice`, no `parallel_tool_calls`, 64-token cache blocks, `deepseek-reasoner` gets no tools array and a text tool protocol instead |
| Qwen | `cache_prompt` on llama.cpp/vLLM, `chat_template_kwargs.enable_thinking=false`, text tool-call recovery |
| Claude | system/tools/end-of-turn cache breakpoints, Messages API stream events |
| OpenAI | `prompt_cache_key` for cache routing, `parallel_tool_calls`, `max_completion_tokens` and no temperature on o-series |

A strict endpoint that rejects a family-specific field is detected by the 400 and
retried once with the core body only; the degradation is recorded on the client
and not repeated for that run.

## Context and caching

- The canonical history is OpenAI-shaped; the Anthropic adapter converts it on
  each request and applies up to 4 `cache_control` breakpoints (tools, system,
  current end, previous request end).
- Tool results longer than 8,000 chars keep head and tail; the full text is
  written to a task-local `spill:N` that `read_file` can page.
- `read_file` is paginated and line-numbered; `grep`/`find_files` are bounded
  and skip build/VCS/binary trees; `edit_file` walks a four-stage ladder
  (exact, trailing-whitespace, indentation, internal-whitespace) and accepts a
  stage only when it matches exactly once, so it is still unique, atomic and
  CRLF/symlink-safe. Every success records which stage applied.

## Sandbox

`run_command` and the validator run through bubblewrap with the same mounts,
namespaces, capability drop and rlimits as before. Within one process the
sandbox is kept warm: a single bwrap session runs a fresh shell per command
(cwd and env reset, per-command timeout kills the process group), removing the
per-command start-up cost; any protocol anomaly falls back to the one-shot path
for the rest of the run.
- Duplicate tool results become pointers; a repeated read of an unchanged file
  is answered from the run-local read cache without touching the filesystem.
  Old results are elided only under pressure, with a deterministic checkpoint
  (files read/written, recent errors) in the first system message.
- Compression is task-aware: old tool results are ranked against the task prompt
  with the local lexical index, so the results the task is about survive and the
  irrelevant ones are elided first (the newest results are always kept).
- A bounded persistent context store (`memory.ContextStore`, under
  `$ORDIVYR_LITE_HOME`) recalls relevant facts from earlier runs on the same
  tree, and a retrieval hint is added only for trees with 20+ indexed files.
- llama.cpp's reported `timings.cache_n`/`prompt_n` replace the estimated cache split,
  so local cache-hit numbers are measured, not derived.
- A local endpoint without provider cache telemetry is charged only the
  incremental input each turn (the prefix is reused), matching the original
  optimized loop. Cloud providers supply real cache telemetry.

## Routing, BYOK and cost

`ModelRouter` holds a primary route and an optional explicit fallback chain.
Each route has its own base URL, protocol, key source, price, caps, timeout and
context window. Spend caps are checked before each paid request; unpaid local
routes are free. A paid route requires stored authorization with positive caps,
or `--allow-paid --max-cost`; a paid model also requires an operator-supplied
price. There is no hidden default discount and no silent paid fallback.

Output limits are model-aware (`gpt-3.5` 4k, `gpt-4o` 16k, Claude 3.5 8k, etc.),
never above half a known context window, and reachable through a bounded
provider-error retry.

## Recovery

`session.py` writes append-only JSONL plus a full snapshot after every completed
model turn. `resume` loads the latest snapshot and continues. File edits are
atomic (temp + rename), so a crash cannot leave a half-written file. The
at-least-once boundary is the interrupted turn: it may be replayed, never
silently skipped.

## Why there is one loop

Lite's `loop.py` is the canonical optimized execution path in this branch. The
Board's older `ordivyr/agentloop.py` remains untouched on its own branch; Board
is expected to migrate to Lite through `ordivyr_lite.job` rather than keep a
second implementation. No Board code was copied into Lite and Lite does not
import `ordivyr`, `selfhost`, `core`, or any web/runtime module.
