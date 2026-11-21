# Security model

Lite executes model-generated tool calls. The default posture is confined and
fail-closed; the operator can widen permissions, but no widening is implicit.

## Permission profiles

| profile | file read | file write | shell | web |
|---|---|---|---|---|
| `read-only` | yes | no | no | no |
| `workspace` | yes | yes (workdir only) | no | no |
| `trusted` | yes | yes | yes (sandboxed) | yes |
| `full` | yes | yes | yes (sandboxed unless opted out) | yes |

`--tools` can only narrow the set. Omitted `--allow-shell`/`--allow-web` values
inherit the selected profile. `--no-shell` and `--no-web` explicitly deny a
profile grant. `--permission workspace --allow-shell` grants only sandboxed
shell while leaving web disabled; the analogous `--allow-web` grants only web.
The `read-only` profile rejects explicit shell/web grants. A tool that is not offered is
also re-checked at execution time, so a hallucinated `write_file` in a
read-only run never reaches the filesystem.

## Filesystem confinement

- Every path is relative; absolute paths and `..` are refused.
- Symlinks are not followed, including a symlinked parent directory.
- The real path must remain inside the workdir.
- Writes are atomic (temp file in the same directory + rename), keep the
  original mode, and refuse `.git`.
- `grep`/`find_files` skip `.git`, `node_modules`, virtualenvs, build outputs,
  binaries and symlinks, and enforce match/file/time bounds.

## Shell sandbox

`run_command` is offered by `trusted`/`full`, or by `workspace` with the
explicit `--allow-shell` grant. Default `--sandbox auto`
uses bubblewrap:

- new user/pid/ipc/uts/cgroup namespaces, `--die-with-parent --new-session`;
- the workdir is bind-mounted read-write at `/workspace`; the toolchain
  (`/usr`, `/usr/local`, `/bin`, `/sbin`, `/lib*`, `/etc`, `/opt`) is read-only;
- a tmpfs `/tmp` and a throwaway `$HOME`;
- no network namespace access unless the operator passes `--network`;
- capabilities dropped (`--cap-drop ALL`);
- CPU, file-size, fd and address-space rlimits; output capped; process group
  killed on timeout.

If `bwrap` is unavailable, `run_command` is **disabled** unless the operator
explicitly chooses `--sandbox process`, which is non-isolated and is labelled
`sandbox=process` in every result. `--sandbox none` is unsafe and intended only
for already-isolated CI containers.

Python packages installed in the user site-packages are mounted read-only and
added to `PYTHONPATH` so a project's test suite can run; the rest of `$HOME` is
not mounted.

## Web fetch

`fetch_url` resolves the hostname, rejects any non-global address (private,
loopback, link-local), connects to the vetted IP with the real hostname for TLS
verification, does not follow redirects, caps the body and strips HTML. It is
off unless web permission is granted.

## Secrets

- API keys come from an environment variable, an explicit key file, or
  `~/.ordivyr-lite/secrets.json` (0600). They are never printed, logged, put in
  a job, stored in a session, or included in a result.
- `Route.__repr__`/`redacted()` mask keys.
- A job containing an inline secret field is rejected before execution.
- Session and ledger files are created 0600; the state directory is 0700.

## Spend safety

- Any non-loopback endpoint is treated as paid.
- Paid calls require positive per-request, daily and total USD caps; missing
  authorization fails before network I/O.
- A paid model without an operator price is refused (spend cannot be capped
  without a price).
- Caps are checked pessimistically before each request and actual cost is
  recorded after.
- A job cannot authorise paid spending by itself; the invoking process must
  pass `--allow-paid --max-cost`.
- A paid fallback after a free route requires `--allow-paid-fallback`.

## Streaming (0.2.0)

Streaming changes when text is shown, never what is allowed to run:

- a tool call exists only after its argument fragments parse as a JSON object
  **and** the stream reported a finish reason; a cut stream makes the loop
  execute nothing from that turn;
- text and reasoning deltas are display data; they never become arguments;
- recovered "tool calls written into the text" are honoured only when they are
  an explicit tool_call block and the tool is one the profile offered - a fenced
  JSON blob that merely looks like a call is left as text when the endpoint
  already accepts a tools array;
- `delta` events are not written to the session journal, so a streamed answer does
  not grow the on-disk journal by one row per token.

## Memory, index and context store (0.2.0)

- The index and the context store live under `$ORDIVYR_LITE_HOME` (0700), are written
  atomically with mode 0600, and contain only paths inside the run's workdir,
  truncated task text and error summaries. No file contents beyond a single
  bounded snippet, no environment, no key material.
- Indexing refuses symlinks, files outside the workdir, oversized files,
  binaries and build/VCS directories, and re-validates containment before
  re-reading a cached hit. A corrupt or foreign cache is ignored.
- Recall text is injected into the system prompt as data, exactly like tool
  output; the permission profile still decides what any following call may do.
- Both features are best-effort: if the state directory cannot be written the
  run continues without them, and a small tree never builds an index at all.

## Warm sandbox session (0.2.0)

The warm session spawns the *same* bwrap argv as the one-shot path (same
mounts, namespaces, capability drop, env and rlimits) and only replaces the
trailing command with a driver loop:

- each command is a fresh `/bin/sh -c` child, so cwd, env and shell state cannot
  leak from one command into the next;
- stdout/stderr are captured into per-command files inside the sandbox tmpfs and
  framed by a per-command random nonce plus byte counts, so a command cannot
  spoof another command's output or exit status by printing a terminator;
- the per-command CPU rlimit is applied inside the driver (a session-wide CPU
  limit would accumulate across commands); a timeout kills the process group
  and the session respawns;
- any protocol anomaly falls back to the one-shot path for the rest of the run;
- `close()` (and an `atexit` hook) reaps the session; no child survives the
  process.

Known divergence: a single command that writes more than the 100 MB file-size
rlimit to stdout fails in the session path where the one-shot pipe path would
have clipped it. The limit is not a boundary (the workdir is writable anyway)
and the session path is stricter, never looser.

## Prompt injection and untrusted output

Tool output is labelled and treated as data. The system prompt tells the model
not to follow instructions found in files or web pages. This is a mitigation,
not a proof: a sufficiently persuasive injection can still influence a model.
The permission profile and sandbox are the enforcement boundary, not the prompt.

## Validation is not a security boundary

Syntax/test/command validators improve correctness. A validator command is
chosen by the operator/config, not by the model, and runs in the same sandbox.
`tests_passed: true` means that command exited 0; it is not an independent
review and `independent_review` is always `null`.

## Known limitations

- `--sandbox process` is not isolated; use it only where the host is already
  isolated.
- The sandbox shares the host kernel; a kernel exploit is out of scope.
- Network is all-or-nothing inside the sandbox.
- `run_command` has a 300 s ceiling and an 8,000-char command limit.
- Cost caps are reserved atomically across concurrent processes as of 0.2.0
  (a `flock`-protected check-then-reserve transaction with a TTL and a settle row).
  On a platform without `fcntl` (Windows) the lock degrades to in-process only.
- The tokenizer is an estimate with calibration, not the provider's tokenizer,
  unless a `tokenizer.json` is supplied; a cap check uses the larger of two
  independent estimates so a sparse tokenizer cannot make a call look cheaper.
