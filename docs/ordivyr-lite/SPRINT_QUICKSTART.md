# Standalone Lite

Python 3.11 or newer. Install into a virtual environment:

```sh
python3 -m venv .venv
.venv/bin/pip install '.[dev]'
.venv/bin/ordivyr-lite init --model MODEL --base-url http://127.0.0.1:8080/v1
.venv/bin/ordivyr-lite run --workdir /path/to/repository 'Implement the requested change and verify it'
.venv/bin/ordivyr-lite sessions list
.venv/bin/ordivyr-lite resume /path/to/session.jsonl
```

`init` refuses to replace existing configuration. `--api-key-file FILE` binds a
local credential file without putting its contents in argv. Cloud routes require
HTTPS, model prices, authorization and positive spending caps through the
existing `config set-price`, `config authorize` commands. Configuration alone
does not authorize spending. `--home DIR` chooses independent configuration and
usage/session storage. Board is not a dependency.

For Go installed outside the standard system toolchain directories, set
`ORDIVYR_LITE_GO_BINARY=/absolute/path/to/go`. The sandbox mounts only that Go
installation read-only. Python virtual environments are also mounted read-only.
No dependencies are automatically installed into the target repository.

The existing JSON job interface is the minimal harness configuration. Save this
as `harness.json`, select the model/workdir, then run
`ordivyr-lite job validate --file harness.json` and
`ordivyr-lite job run --file harness.json`:

```json
{
  "protocol": "ordivyr-lite/job",
  "version": 1,
  "task": {
    "prompt": "Implement the requested change and its regression tests.",
    "workdir": "/absolute/path/to/repository",
    "system": "Preserve unrelated work. Report verification limits."
  },
  "model": {"provider": "custom", "model": "MODEL", "base_url": "http://127.0.0.1:8080/v1"},
  "permissions": {
    "profile": "workspace", "sandbox": "auto", "network": false,
    "tools": ["read_file", "list_files", "grep", "find_files", "edit_file", "write_file"]
  },
  "limits": {"max_turns": 32, "max_tool_calls": 120, "max_seconds": 300, "max_repairs": 2},
  "validation": {"mode": "auto"},
  "session": {"enabled": true}
}
```

Use `job example` for the installed protocol spelling and current example.
Use explicit `validation.mode=command` for an operator-owned acceptance command.
Workspace test success and model self-review are not independent acceptance.
TypeScript compilation is syntax/type evidence; it is not a test suite.

Interrupted writes are never replayed. Recovery recognizes only an exact bound
postimage. Ambiguous shell/external actions remain unconfirmed. Read identities
survive resume, so edits using stale context fail until the file is read again.
Digest checks are optimistic: they do not provide exclusive-writer locking or
transactional changes across files.

Existing pytest fixtures and test configuration are trusted only at their
captured bytes. Changing them requires operator review or an explicit validation
command; pytest module shadowing remains refused. Mixed-language repositories
currently choose the Python suite first when actual Python tests are present.

# Optional Board adapter

`ordivyr_lite.board_adapter.execute` consumes Board contract 1.1, vendored
byte-for-byte as `board_contract.py`, and calls the existing Lite job/Runner path.
Board remains independently installed. The first adapter supports exact grants
for file tools in personal generated-artifact workspaces. It rejects shell,
delegation, named repositories and external validator commands rather than
silently ignoring unsupported semantics. Model routing is worker-owned.

# Explicit preservation constraints

Recognized instructions such as “Keep existing tests and configuration files
unchanged” capture the original files' byte identities. File tools refuse changes
to those originals; separate NEW regression test files remain writable. Named
files can be protected with “Keep `README.md` unchanged.” Quote extensionless
names and dotfiles. These guards persist through session resume and are checked
before and after verification. An owner change stops completion and is preserved
for inspection; the harness never restores a file automatically.

This is supported-pattern detection, not a general natural-language permission
parser. Preservation exceptions fail with a clear error. While protection is
active, shell, delegation and custom mutation tools are rejected because their
effects cannot be confined to the protected file set by this adapter. Operator
validators still run, and any protected-file mutation invalidates verification.
Protected input checking is bounded to16MB and remains optimistic with respect
to concurrent writers. Use file tools for this workflow.
