# Journal durability: 021B

Continue from 021A's read-only reconciliation contract. This changes journal
correctness, not its snapshot strategy: full snapshots and synchronous required
barriers remain. No fsync batching or delta code from the divergent hardening
branch is imported.

## Storage behavior

`Session.append` checks the exact `os.write` byte count. A short/zero append or
write error raises `OSError` (`JournalError` for short counts). No fragment retry
is attempted after a torn record. The session becomes unusable for later writes
or barriers; catching a diagnostic error cannot make a later required barrier
succeed. Inspect/reopen and validate explicitly after a failure. `close` releases
the descriptor even when reporting a prior failure.

`flush` always calls fsync and propagates its error. `close` syncs dirty records
and surfaces failures. The Runner no longer swallows required snapshot,
request-commit/abandon, or diagnostic append failures. A diagnostic append failure
can damage required record framing, so it cannot be ignored. Ordinary diagnostic
records still share the next explicit barrier; this is not fsync-per-diagnostic.
Successful run completion requires the final stop-record sync to succeed.
Required storage exceptions abort rather than returning a successful RunResult;
the CLI reports an error instead of printing a success JSON result.

Resume validates the complete journal and establishes a fresh sync barrier before
consuming its saved state or acknowledging any tool result. A record visible in
the page cache does not prove an earlier fsync returned successfully. A later
successful barrier establishes a current journal barrier, not retroactive proof.
021A still verifies exact current file postimages and never replays file mutations
or unknown external operations.

## Framing and conservative reads

Each new JSONL record retains its event fields and adds reserved `_journal`
metadata with format `ordivyr-lite/journal-record/1`: positive sequence number,
canonical payload byte count/SHA-256 digest and previous physical-record digest.
Payload encoding is UTF-8 JSON with sorted keys, compact separators, ASCII escapes
and finite numbers. The digest excludes `_journal`; the previous digest covers
the complete preceding physical line including its newline. Sequence and digest
checks detect interior gaps, duplication, reordering and changed framed content.
One Session serializes appends using a thread lock.

Complete legacy JSONL prefixes remain readable. The first framed record has seq=1
and binds the previous legacy line; after framing begins, an unframed downgrade
is rejected. Unknown framing versions, duplicate keys, malformed records and
unterminated lines stop reading at the first invalid record. Suffix successes are
never skipped into the result stream. `events()` and `latest_snapshot()` expose
the valid prefix for inspection; `integrity_error` identifies an invalid suffix.
That inspection is not a durability acknowledgement. `require_healthy`, automatic
resume and opening for further append refuse damaged journals. Raw bytes remain
untouched; there is no automatic truncate, repair or suffix deletion.

## Limits

- Checksums are integrity framing, not signatures or authentication against a
  malicious writer who can recompute the chain. Multiple processes/Session
  instances writing one journal are unsupported; the thread lock covers one
  Session only. Uncooperative file/journal mutation is not excluded by a CAS.
- A clean loss of the entire final suffix has no external high-water mark and
  cannot always be detected. No historical fsync-success proof is encoded.
  An interrupted unknown effect remains unknown; current file bytes are verified
  by 021A before content can be acknowledged.
- Legacy records cannot gain retroactive gap/checksum guarantees. Older binaries
  and the preserved hardening branch do not enforce this new framing; do not use
  them to resume these journals. No automatic downgrade is supported.
- File fsync is not device/power-loss certification. Directory-entry fsync, target
  file/directory crash durability, ENOSPC on an actual full device, NFS semantics,
  kernel/device faults and remote exactly-once execution are unverified. The
  existing atomic file tools are not a workspace transaction.
- Tests use deterministic injected EIO/ENOSPC, real short byte appends and explicit
  suffix-loss simulations on disposable files. The inherited 021A SIGKILL test
  is process-death evidence, not an actual power outage. No live systems or
  credentials are involved, and no deployment is authorized by these results.

Regression cases: `ordivyr_lite/tests/test_journal_durability.py`. Separate
acceptance program: `evidence/journal-durability/verify-independent.py`; it imports
neither tests nor journal/recovery helpers. It is same-author verification, not
independent human review. Before/after evidence and exact source identities are
indexed in `evidence/journal-durability/REPORT.md` and `MANIFEST.json`.
