# Interrupted file recovery

The `reconcile` policy checks recorded full-file identity and never dispatches an
interrupted `write_file` or `edit_file` operation. An exact postimage can be
acknowledged. A preimage, conflict, unsupported identity or missing legacy
metadata produces an error/unknown observation and preserves current bytes.
Unapplied operations require inspection and a new explicit tool call. There is
no automatic replay of shell, MCP, delegate or other unknown external effects.
Read-only tool recovery remains available under current permissions.

The `strict` policy still stops when any call lacks a durable result, including
when its current postimage happens to match. Durable file results reconciled from
an interrupted assistant turn also require current exact content verification;
a historical success string alone is insufficient. A legacy completed file
result without adequate identity is conservatively unknown too.

## Versioned evidence

New built-in file intents/results carry `file_identity` with format
`ordivyr-lite/file-intent/1`, algorithm `sha256`, bounded session run/tool ids,
tool name, digests of the canonical scoped root/target and canonical JSON
arguments, operation semantics, and `pre`/`post` images. Images contain
`exists`, byte count and full byte digest (null digest for absence). Images are
bounded to 8,000,000 bytes; unsupported plans retain no usable identity and
cannot be automatically acknowledged. New metadata contains neither plaintext
file contents nor additional plaintext paths. Existing journal arguments and
outputs keep their existing storage/privacy behavior.

`write:utf8/1` means whole-file UTF-8 bytes. Edit semantics record the actual
`exact`, `rstrip`, `indent` or `collapse` match stage and `one`/`all` replacement.
Both intent calculation and normal editing use the same file-tool planner,
including CRLF normalization and tolerant matching. Recovery never applies that
planner to infer history from current substring presence. Custom handlers that
override built-in tool names do not inherit built-in recovery semantics.

Intents are flushed before ordinary dispatch. Ordinary successful file results
with usable identity are checked against the intended postimage before success
is journaled. Recovery classifications/results are appended and flushed before
callbacks, snapshots and the next model turn. Required flush exceptions propagate.
Permission narrowing, current allow-lists, current declared scope, run/tool
binding, target/existence checks and no-follow path checks apply to verification.
Repeated resumes consume persisted reconciliation evidence without redispatch.

## Writer and durability boundaries

There is no exclusive-writer contract. Recovery therefore performs no file
mutation, even if the recorded preimage is still present. This avoids a
hash-then-replace race entirely for recovery; no compare-and-swap, advisory-lock
exclusion or rollback of independent writers is claimed. A postimage digest
attests to bytes observed during verification, not their continued existence or
which actor wrote them. Later edits remain untouched. Ordinary new tool dispatch
retains the existing writer-concurrency assumptions; this patch does not add a
workspace transaction or lock service.

Directory descriptors and bounded reads reject symlinks and observed target
replacement/change during verification. Linux/POSIX is the tested platform.
Journal files remain trusted local state, not authenticated against malicious
rewriting. Power-loss durability, remote exactly-once effects and arbitrary
concurrent filesystem mutation are not certified by these tests.

At the 021A checkpoint, this was the content-reconciliation fix. The separate divergent fsync/delta
branch was not integrated. The research's known short-`os.write` acceptance in
`Session.append`, storage fault matrix and delta-snapshot concerns remain a
separate 021B task. A successful flush is not evidence that a short append was
complete. This patch does not claim to close that durability defect or authorize
deployment. The subsequent 021B correctness fix and its remaining durability
limits are documented in [JOURNAL_DURABILITY.md](JOURNAL_DURABILITY.md); original
021A reports and evidence remain historical and unchanged.

## Reproduction

Baseline evidence and the unchanged research reproducer live in
`evidence/recovery-integrity/`. It reproduced all three documented failures
against `9974497` before source edits. Its defect assertions intentionally do not
pass after the fix; preserve its baseline output rather than rewriting it.

Run the independent acceptance script with
`PYTHONDONTWRITEBYTECODE=1 python3 evidence/recovery-integrity/verify-independent.py`.
It uses independently written expected bytes/digests and no recovery helper or
test imports; it is not an independent human review. Run the full suite with
`PYTHONDONTWRITEBYTECODE=1 python3 -m pytest ordivyr_lite/tests -o addopts='' -q -p no:cacheprovider`.
Tests exercise actual Runner preparation, interrupted dispatch, exact/fuzzy/all
edits, legacy records, changed existence, scope/permission narrowing, repeated
resume and a disposable child killed after its file effect but before result
commit. No model inference, production service or deployment is required.
