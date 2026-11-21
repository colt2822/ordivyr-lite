# ORDIVYR Lite

<img src="../../ordivyr_lite/assets/ordivyr-lite.png" alt="ORDIVYR Lite — Elyrea in black and white" width="220">

A terminal-first, standalone AI execution engine. Lite owns the agent loop,
context management, tools, model routing, validation and crash recovery. It has
**no runtime dependency on the ORDIVYR Board** — no web server, database,
scheduler, team graph or billing service is imported or required.

It is pure Python 3.11+ standard library: no third-party runtime dependencies,
no dashboard, no subscription system, no hosted control plane.

## Install

```bash
# from the repository root
python3 -m pip install -e .     # from a checkout; or install the wheel from the GitHub Release (verify with SHA256SUMS)
ordivyr-lite --version
```

If you do not want to install, run it in place:

```bash
PYTHONPATH=. python3 -m ordivyr_lite --version
```

The default state directory is `~/.ordivyr-lite` (0600/0700). Override it with
`--home DIR` or `ORDIVYR_LITE_HOME`.

## Quick start

### Guided connection

```bash
cd ~/your-project
ordivyr-lite
# Or explicitly change a connection:
ordivyr-lite setup
# Configure only, without starting chat:
ordivyr-lite setup --no-chat
```

The first terminal launch guides provider selection, credentials, model IDs and
explicit cloud spending limits. Cloud and local connections have no preselected
provider. `Claude` selects Anthropic. Built-in provider URLs are supplied
automatically. Optionally fetch the model list to check metadata access without
an inference request; a failed lookup does not establish a working connection.
Prices are entered from the provider's current rates; no price catalog is assumed.

No Elyraon account is required. An existing environment variable or key file can
be used; pasted keys are stored unencrypted in a local owner-only file (0600).
Setup previews the connection and limits before saving, preserves a backup of
existing configuration, and leaves unsaved choices untouched on cancellation.
Lite does not import CLI/subscription OAuth tokens into its engine. API billing
may be separate from a chat or coding subscription.

Saved connections start chat directly on later launches. Inside chat, `/connect`
opens setup, `/models` lists models, and `/model PROVIDER:MODEL` changes the
active model while retaining the conversation. Explicit credentials, endpoint
overrides and one-run paid authorization are never transferred to the new
provider. Configure and authorize the destination first. Retained conversation
will be sent to the selected provider on the next task. `/new` starts a fresh
conversation and journal while preserving the previous session on disk.

The default tool-enabled prompt loads a regular, nonsymlink `AGENTS.md` from the
workspace root, bounded to 16 KiB. It does not search parent directories or load
nested guidance automatically. Explicit system overrides and tool-free requests
remain unchanged. Project instructions cannot grant tools or spending permission.

### Use an installed native CLI

```bash
ordivyr-lite native claude --check
ordivyr-lite native claude --workdir ~/project
# Other installed clients:
ordivyr-lite native codex --workdir ~/project
ordivyr-lite native gemini --workdir ~/project
```

This explicitly opens the installed, unmodified CLI in your terminal. That CLI owns its login, tools, permissions, billing and sessions; Lite's limits do not apply. No credential is copied into Lite, no prompt or permission-bypass flag is inserted, and no CLI is downloaded. `--check` only resolves availability and starts no CLI or model.

Use `ordivyr-lite setup` for Lite's own engine and its validation/recovery loop. The native handoff runs the chosen client's engine. Claude subscription login completes through Claude Code's own flow; Lite does not turn a subscription token into an API key.

### Local inference (free, no key)

```bash
# llama.cpp server, LM Studio, Ollama, vLLM: any OpenAI-compatible endpoint
ordivyr-lite run "Read README.md and list the three most important commands." \
  --workdir . \
  --provider custom \
  --base-url http://127.0.0.1:8181/v1 \
  --model local-3060 \
  --api-key-file ~/path/to/local.key \
  --permission read-only
```

`--provider llamacpp`, `ollama`, or `lmstudio` picks the preset base URL.

### Interactive session

```bash
ordivyr-lite chat --workdir ~/project --provider llamacpp --model local-3060
```

`/help`, `/status`, `/connect`, `/models`, `/model PROVIDER:MODEL`, `/cost`,
`/tools`, `/clear`, `/new`, `/session`, `/exit`.

### One task, machine-readable

```bash
ordivyr-lite run "Fix the failing tests" --workdir . \
  --provider llamacpp --model local-3060 \
  --permission workspace --validate auto --json
```

`--validate tests` and `auto` run the project's own tests. For Python projects, **`pytest` must be
installed in the same environment as Lite** (`pip install pytest`); otherwise verification fails
closed with `validator_dependency_missing` and the result is reported as `unverified`, even if the
code is correct. Node projects need a real `node` binary on `PATH`.

### Scripted / Board invocation

```bash
ordivyr-lite job example --workdir . > job.json
ordivyr-lite job validate -f job.json
ordivyr-lite job run -f job.json --json
```

The job/result protocol is versioned and documented in
[COMPATIBILITY.md](COMPATIBILITY.md).

## Permissions

| Profile | Read | Write/edit | Sandboxed shell | Web fetch |
|---|---|---|---|---|
| `read-only` | yes | no | no | no |
| `workspace` (default) | yes | yes | no | no |
| `trusted` | yes | yes | yes | yes |
| `full` | yes | yes | yes | yes |

`run_command` is offered by `trusted`/`full`, or by `workspace` with the
explicit `--allow-shell` grant, and runs under bubblewrap by
default: workdir at `/workspace`, toolchain read-only, tmpfs `$HOME`, no
network unless `--network`, capabilities dropped. If `bwrap` is missing the
tool is disabled unless the operator explicitly chooses `--sandbox process`
(non-isolated, clearly reported). See [SECURITY.md](SECURITY.md).

## Model routing and BYOK

- OpenAI-compatible presets: OpenAI, OpenRouter, DeepSeek, Groq, Mistral, xAI,
  Together, Google's OpenAI-compatible endpoint, Ollama, LM Studio, llama.cpp,
  and any `--base-url`.
- Anthropic Messages protocol (including prompt-cache breakpoints).
- Keys come from `--api-key-env`, `--api-key-file`, a provider-specific
  environment variable, or `ordivyr-lite config set-key` (0600 file). A key is
  never printed or written into a result.
- A non-loopback endpoint is **paid** and fails closed: it needs
  `--allow-paid --max-cost USD`, or a stored authorization with positive
  daily/total/per-request caps. A paid model also needs an operator price
  (`config set-price`) before any request; local endpoints are free.
- Fallback routes are explicit (`--fallback PROVIDER:MODEL`); a paid fallback
  after a free route additionally needs `--allow-paid-fallback`.

## Streaming and provider adaptation

A model turn is streamed: text appears as it is generated, reasoning stays out
of the answer, and tool calls are assembled from argument fragments and are only
runnable once the JSON is complete. A cut-off stream executes nothing.
`--no-stream` (or `ORDIVYR_LITE_STREAM=0`) waits for each whole turn instead.

The request body is shaped per model family, so one engine drives each provider
well without a per-provider code path:

- **DeepSeek** — tools and `tool_choice`, no `parallel_tool_calls`, 64-token prefix-cache
  blocks, and `deepseek-reasoner` (which has no function calling) gets a text tool
  protocol plus recovery of `<tool_call>{"name":...}</tool_call>` blocks.
- **Qwen** (and llama.cpp/vLLM generally) — `cache_prompt` for warm prefix reuse and
  `chat_template_kwargs.enable_thinking=false` so a coding turn is not spent on a
  thinking block. A strict endpoint that rejects either field is retried once
  with the core body and the degradation is remembered for the run.
- **Claude** — Messages cache breakpoints on system, tools and end of turn; SSE
  event assembly.
- **OpenAI** — `prompt_cache_key` for cache routing, `parallel_tool_calls`; o-series
  uses `max_completion_tokens` and no temperature.

## Doing more per token

- A repeated read of a file that has not changed is answered from the run-local
  read cache instead of the filesystem, and the model is told why.
- Validation can be reused only with a complete unchanged content identity;
  partial repository manifests require fresh checks. A repair
  loop does not pay for the same failing suite twice.
- A truncated answer is retried with a smaller output budget instead of being
  accepted as final.
- Requested output tokens are sized to the remaining context window and to the
  remaining token budget each turn.
- Old tool results are elided by relevance to the task first (a local lexical
  index, no network), keeping the ones the task is about.
- A bounded context store under `$ORDIVYR_LITE_HOME` recalls facts from earlier runs on
  the same tree.
- `run_command` and validation share one warm bubblewrap session per process, which
  removes the per-command start-up cost; confinement and rlimits are unchanged.
- Coding tasks get one verification checkpoint while there is still room to
  repair: short turn limits and observed token, cost or time pressure can move
  it earlier. Green tests also prompt review of unchecked behavior before the
  last reply. These are planning estimates, not a guarantee about future usage.
- All named items in a requirement remain in its checklist. Presence checks
  omitted by older saved sessions are restored from the retained task text;
  existing checks are kept. Workspace tests still need separate task acceptance
  for a quality claim.

## Cost and token reporting

Every model call appends one line to `$ORDIVYR_LITE_HOME/usage.jsonl`:

```bash
ordivyr-lite usage            # today/all, by model and provider
ordivyr-lite usage --since 7d --json
```

The result JSON reports fresh input, cache read, cache write, output,
context occupancy, cache-hit rate, model latency, wall time, tool success rate
and cost. Costs use operator-supplied prices per million tokens; no discount is
invented. Local runs report `$0`.

## Recovery

Each run writes an append-only session journal with a snapshot after every
completed model turn. If the process dies:

```bash
ordivyr-lite sessions list
ordivyr-lite resume ~/.ordivyr-lite/sessions/run-....jsonl
```

Resume retains the original task and validation choice. You can append a new
instruction with `ordivyr-lite resume SESSION "Also add the regression test"`;
it becomes part of the task and invalidates a pending final answer. Existing
preservation rules remain binding. Final verification checks the current files.

Interrupted file mutations are not replayed: Lite acknowledges only an exact
expected postimage and preserves conflicting bytes. Unknown shell or external
effects are not automatically repeated. An interrupted custom `--check` stops
for inspection and stays held across resumes, including `--validate off`.
Inspect its effects and run the check manually before starting a new session.
Normal detected test suites may run again after restart; this is not an
exactly-once guarantee for test side effects. Older pending-final sessions
without a saved validation policy require an explicit validation choice.
Use `--no-session` to disable journaling.

## Layout

```
ordivyr_lite/
  cli.py           command line (run, chat, job, config, doctor, ...)
  loop.py          the execution loop (turns, tools, repair, breakers)
  providers.py     OpenAI/Anthropic wire clients + usage normalisation
  router.py        primary/fallback routing and spend checks
  context.py       append-only history, compaction, checkpoint
  tools.py         permission profiles + confined file/shell/web tools
  filetools.py     read/edit/grep/find + spill, four-stage edit ladder
  profiles.py      per-family request shaping (DeepSeek/Qwen/Claude/OpenAI)
  tokens.py        tokenizer + per-family calibration + cache-block alignment
  memory.py        lexical index, relevance compression, persistent context
  sandbox.py       bubblewrap/process isolation + warm session
  validators.py    syntax/test/command validation
  session.py       crash-recovery journal
  job.py           versioned Board <-> Lite protocol
  usage.py         ledger, pricing, caps
  config.py        presets, BYOK, model-aware limits
```

## Tests

```bash
python3 -m pytest ordivyr_lite/tests -q
```

## Status and limits

Alpha. Lite has not been shown to outperform other agent harnesses: in the project's own
same-cohort comparison its repair correctness was below the competitors', and no paid model
endpoint was benchmarked. Treat it as an experimental, durable, local-first execution engine.
Test counts are in [TEST_RESULTS.md](TEST_RESULTS.md); remaining bottlenecks in
[PERFORMANCE.md](PERFORMANCE.md).

## License

Apache-2.0; see `LICENSE` and `NOTICE`.
