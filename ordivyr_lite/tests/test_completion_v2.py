"""Completion engine V2: requirements, bounded manifests, project tests, prefetch, harness
checks and honest post-stop verification."""
import json
import os
from pathlib import Path

import pytest

from ordivyr_lite import requirements as REQ
from ordivyr_lite import validators as V
from ordivyr_lite import verification as VER
from ordivyr_lite.ledger import Ledger
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator


def put(root, files):
    for name, text in files.items():
        path = Path(root) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


# ---------------------------------------------------------------- requirements
def test_requirements_extracted_without_a_model_and_survive_ledger_roundtrip(tmp_path):
    prompt = ("Add a function `parse_flags` and create `utils/flags.py`. It returns a list. "
              "The default is empty. Update all callers. Work in the supplied source repository.")
    items = REQ.extract(prompt)
    assert [i['kind'] for i in items] == ['interface', 'return', 'default', 'caller']
    assert items[0]['status'] == 'open' and len(items[0]['checks']) == 2
    ledger = Ledger.from_prompt(prompt)
    again = Ledger.from_dict(json.loads(json.dumps(ledger.to_dict())))
    assert again.requirements == ledger.requirements
    assert 'acceptance checklist' in again.render()
    put(tmp_path, {'utils/flags.py': 'def parse_flags():\n    return []\n'})
    REQ.evaluate(items, tmp_path)
    assert items[0]['status'] == 'satisfied'
    assert items[1]['status'] == 'unchecked'          # behavior is never promoted to satisfied
    assert REQ.summary(items)['open'] == 0


def test_open_requirement_blocks_done_and_is_repaired_within_budget(tmp_path):
    put(tmp_path, {'app.py': 'X = 1\n'})
    calls = 0

    def policy(body):
        nonlocal calls
        calls += 1
        if calls == 1:
            return openai_reply(text='all done')                       # nothing implemented
        if calls == 2:
            return openai_reply(calls=[('write_file', {'path': 'app.py',
                                                       'content': 'def parse_flags():\n    return []\n'})])
        return openai_reply(text='done now')
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box, 'Add a function `parse_flags` to app.py.',
                    stream=False, validator=Validator('syntax'), limits=RunLimits(max_turns=8)).run()
    assert result.stop_reason == 'done' and calls == 3
    assert result.stats.repairs == 1
    assert result.completion['requirements']['open'] == 0


def test_unmet_requirement_never_reports_done(tmp_path):
    put(tmp_path, {'app.py': 'X = 1\n'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(lambda b: openai_reply(text='finished'))), box,
                    'Add a function `parse_flags` to app.py.', stream=False,
                    validator=Validator('syntax'), limits=RunLimits(max_turns=8, max_repairs=1)).run()
    assert result.stop_reason == 'unverified'
    assert result.completion['verified'] is False
    assert result.completion['requirements']['open'] == 1


# ---------------------------------------------------------------- manifest / reports
def test_manifest_degrades_to_partial_instead_of_raising(tmp_path, monkeypatch):
    put(tmp_path, {f'm{i}.py': f'V={i}' for i in range(6)})
    monkeypatch.setattr(VER, 'MAX_FILES', 2)
    VER._STAT_CACHE.clear()
    with pytest.raises(ValueError):
        VER.inventory(tmp_path)
    got = VER.manifest(tmp_path)
    assert VER.PARTIAL in got and len(got) == 7
    assert VER.select_tests(tmp_path, ['m1.py'], got).broad
    (tmp_path / 'm5.py').write_text('V=99')
    assert VER.manifest(tmp_path) != got


def test_manifest_matches_inventory_within_bounds(tmp_path):
    put(tmp_path, {'a.py': 'A=1', 'b/c.py': 'C=2'})
    assert VER.manifest(tmp_path) == VER.inventory(tmp_path)


def test_change_triggers_force_the_whole_suite(tmp_path):
    put(tmp_path, {'pkg/__init__.py': '', 'pkg/a.py': 'A=1', 'test_a.py': 'from pkg.a import A'})
    inv = VER.inventory(tmp_path)
    for path, kind in (('pkg/__init__.py', 'export'), ('migrations/001.sql', 'migration'),
                       ('requirements.txt', 'dependency'), ('api.schema.json', 'schema')):
        assert kind in VER.change_triggers([path])
    sel = VER.select_tests(tmp_path, ['pkg/__init__.py'], inv)
    assert sel.broad and 'export' in sel.reason


def test_condense_report_keeps_failing_node_assertion_and_frame():
    noisy = ('[focused]\n' + 'collected 3 items\n' + 'pad line\n' * 400 +
             '>       assert f(1) == 2\nE       assert 3 == 2\ntest_a.py:5: AssertionError\n'
             'FAILED test_a.py::test_x - assert 3 == 2\n1 failed, 2 passed in 0.1s\n')
    out = VER.condense_report(noisy, 600)
    assert 'FAILED test_a.py::test_x' in out and 'E       assert 3 == 2' in out
    assert 'test_a.py:5: AssertionError' in out and len(out) < 600
    assert VER.condense_report('short', 600) == 'short'
    assert VER.condense_report('x' * 5000, 600).endswith('x' * 100)    # unstructured: tail, never silence


# ---------------------------------------------------------------- JS / TS / Go
def test_javascript_command_is_frozen_before_the_task(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, 'which', lambda name: '/usr/bin/' + name)
    put(tmp_path, {'package.json': json.dumps({'scripts': {'test': 'node --test'}}), 'index.js': 'x'})
    kind, argv, digest = V.detect_project_tests(tmp_path)
    assert kind == 'javascript' and argv[1:] == ['test', '--silent', '--ignore-scripts']
    v = Validator('auto')
    ran = []

    class Box:
        root = tmp_path

        def sandbox_run(self, command, timeout):
            ran.append(command)
            from ordivyr_lite.sandbox import SandboxResult
            return SandboxResult(0, False, 'ok', '', 'bwrap', True, 0.1, False, False)
    v.prepare(Box())
    assert v.project_def == ('javascript', digest)
    assert v(Box(), ['index.js']).ok is True and ran
    put(tmp_path, {'package.json': json.dumps({'scripts': {'test': 'true'}})})     # agent weakens it
    refused = v(Box(), ['package.json'])
    assert refused.ok is False and 'test_definition_changed' in refused.report and len(ran) == 1


def test_placeholder_npm_script_and_missing_toolchains_detect_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, 'which', lambda name: '/usr/bin/' + name)
    put(tmp_path, {'package.json': json.dumps({'scripts': {'test': 'echo "Error: no test specified" && exit 1'}})})
    assert V.detect_project_tests(tmp_path) is None
    monkeypatch.setattr(V.shutil, 'which', lambda name: None)
    put(tmp_path, {'package.json': json.dumps({'scripts': {'test': 'node t.js'}}),
                   'go.mod': 'module x\n', 'a_test.go': 'package x'})
    assert V.detect_project_tests(tmp_path) is None


def test_go_runs_offline_readonly_and_dependency_change_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(V.shutil, 'which', lambda name: '/usr/local/go/bin/go' if name == 'go' else None)
    put(tmp_path, {'go.mod': 'module x\n', 'a.go': 'package x', 'a_test.go': 'package x'})
    kind, argv, digest = V.detect_project_tests(tmp_path)
    assert kind == 'go'
    joined = ' '.join(argv)
    for flag in ('GOFLAGS=-mod=readonly', 'GOPROXY=off', 'GOTOOLCHAIN=local', 'test ./...'):
        assert flag in joined
    put(tmp_path, {'go.mod': 'module x\nrequire evil v1\n'})
    assert V.detect_project_tests(tmp_path)[2] != digest


# ---------------------------------------------------------------- loop behavior
def test_small_repository_snapshot_replaces_discovery_reads(tmp_path):
    put(tmp_path, {'calc.py': 'def add(a, b):\n    return a - b\n', 'notes.txt': 'owner'})
    seen = {}

    def policy(body):
        seen.setdefault('system', body['messages'][1]['content'])
        n = len(seen.setdefault('turns', [])) + 1
        seen['turns'].append(n)
        if n == 1:
            return openai_reply(calls=[('read_file', {'path': 'calc.py'})])
        return openai_reply(text='ok')
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box, 'Fix add in calc.py so it adds.', stream=False,
                    validator=Validator('syntax'), limits=RunLimits(max_turns=6)).run()
    assert 'Task-start snapshot' in seen['system'] and 'return a - b' in seen['system']
    assert result.stats.prefetched_files >= 1 and result.stats.prefetch_hits == 1
    tool_msg = [m for m in result.messages if m.get('role') == 'tool'][0]['content']
    assert 'unchanged since the task-start snapshot' in tool_msg


def test_edit_gets_a_harness_check_note_and_post_stop_verification_is_honest(tmp_path):
    put(tmp_path, {'calc.py': 'def add(a, b):\n    return a - b\n',
                   'test_calc.py': 'from calc import add\ndef test_add(): assert add(1, 2) == 3\n'})
    calls = 0

    def policy(body):
        nonlocal calls
        calls += 1
        if calls == 1:
            return openai_reply(calls=[('edit_file', {'path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'})])
        return openai_reply(calls=[('read_file', {'path': 'calc.py', 'offset': 1})])    # never finishes
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box, 'Fix add in calc.py so it adds.', stream=False,
                    validator=Validator('auto', sandbox_mode='process'),
                    limits=RunLimits(max_turns=4, max_repairs=1)).run()
    tool_msgs = [m['content'] for m in result.messages if m.get('role') == 'tool']
    assert any('[harness check] tests pass' in t for t in tool_msgs)
    assert result.stop_reason == 'steps' and result.status == 'failed'          # stop reason never rewritten
    after = result.completion['verified_after_termination']
    assert after['ok'] is True and after['tests_run'] and after['model_declared_done'] is False
    assert result.completion['verified'] is False
    assert result.completion['verification_level'] == 'verified_artifact_after_stop'
    assert result.as_dict()['verification']['harness_verified'] is False


def test_failed_artifact_after_stop_is_not_reported_verified(tmp_path):
    put(tmp_path, {'calc.py': 'def add(a, b):\n    return a - b\n',
                   'test_calc.py': 'from calc import add\ndef test_add(): assert add(1, 2) == 3\n'})
    calls = 0

    def policy(body):
        nonlocal calls
        calls += 1
        if calls == 1:
            return openai_reply(calls=[('write_file', {'path': 'notes.md', 'content': 'x'})])
        return openai_reply(calls=[('read_file', {'path': 'calc.py', 'offset': 1})])
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box, 'Fix add in calc.py so it adds.', stream=False,
                    validator=Validator('auto', sandbox_mode='process'), limits=RunLimits(max_turns=4)).run()
    assert result.completion['verified_after_termination']['ok'] is False
    assert result.completion['verification_level'] == 'unverified'


def test_untouched_workspace_is_never_verified_after_stop(tmp_path):
    put(tmp_path, {'calc.py': 'def add(a, b):\n    return a + b\n',
                   'test_calc.py': 'from calc import add\ndef test_add(): assert add(1, 2) == 3\n'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(lambda b: openai_reply(calls=[('read_file', {'path': 'calc.py', 'offset': 1})]))),
                    box, 'Do something.', stream=False, validator=Validator('auto', sandbox_mode='process'),
                    limits=RunLimits(max_turns=3)).run()
    assert 'verified_after_termination' not in result.completion
    assert result.completion['verified'] is False


def test_v2_can_be_disabled(tmp_path):
    put(tmp_path, {'calc.py': 'def add(a, b):\n    return a - b\n'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(lambda b: openai_reply(text='ok'))), box, 'Fix add.', stream=False,
                    limits=RunLimits(max_turns=3, completion_v2=False)).run()
    assert result.stats.prefetched_files == 0 and 'verified_artifact_after_stop' not in result.completion
