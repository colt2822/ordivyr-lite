"""Actual Runner journal/resume regressions; no model or external integrations."""
import copy
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.recovery import capture_intent
from ordivyr_lite.session import Session
from ordivyr_lite.tools import ToolBox


class Interrupted(BaseException):
    pass


def state(box, name, args):
    return {'messages': [{'role': 'user', 'content': 'task'}, {'role': 'assistant', 'content': None,
             'tool_calls': [{'id': 'c', 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}],
            'stats': {'model_calls': 1, 'tool_calls': 0}, 'limits': {'max_turns': 10},
            'effective_permissions': box.permission_manifest()}


def make_interrupted(root, name, args, pre, *, legacy=False, completed=False):
    target = root / args.get('path', 'target.txt')
    if pre is not None:
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(pre)
    box = ToolBox(root, profile='workspace')
    journal = Session.create(root / 'session.jsonl', run_id='recovery-test')
    journal.snapshot(state(box, name, args))
    runner = Runner(None, box, 'task', session=journal, limits=RunLimits(delegate=False))
    runner.steps = 1
    call = SimpleNamespace(id='c', name=name, args=args)
    if legacy:
        journal.append({'type': 'tool_started', 'tool_id': '1:c', 'tool': name, 'args': args})
        journal.flush()
    elif completed:
        assert not runner._run_tool(call).startswith('ERROR')
    else:
        with patch.object(box, 'call', side_effect=Interrupted), pytest.raises(Interrupted):
            runner._run_tool(call)
    return box, journal, target


def resume(box, journal, **kwargs):
    primary = SimpleNamespace(context_window=0, label=lambda: 'no-model')
    runner = Runner(SimpleNamespace(primary=primary, local_only=True), box, 'task',
                    session=journal, resume_state=journal.latest_snapshot(),
                    limits=RunLimits(delegate=False), **kwargs)
    with patch.object(box, 'call', side_effect=AssertionError('recovery must not dispatch file tools')):
        runner._prepare()
    return runner, [m['content'] for m in runner.messages if m.get('role') == 'tool']


CASES = [
    ('write_file', {'path': 'target.txt', 'content': 'agent-v1\n'}, b'base\n', b'human-edit\n', False),
    ('edit_file', {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'alpha omega'}, b'alpha\n', b'alpha omega\n', True),
    ('edit_file', {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'omega'}, b'alpha\n', b'human omega\n', False),
]


@pytest.mark.parametrize('legacy', [False, True])
@pytest.mark.parametrize('name,args,pre,current,expected_success', CASES)
def test_three_documented_failures(tmp_path, legacy, name, args, pre, current, expected_success):
    box, journal, target = make_interrupted(tmp_path, name, dict(args), pre, legacy=legacy)
    target.write_bytes(current)
    runner, messages = resume(box, journal)
    assert target.read_bytes() == current
    assert len(messages) == 1
    assert messages[0].startswith('OK') == (expected_success and not legacy)
    assert runner.stats.interrupted_calls == 1
    events = journal.events()
    completion_index = next(i for i, e in enumerate(events) if e.get('reconciled'))
    snapshot_index = max(i for i, e in enumerate(events) if e['type'] == 'snapshot')
    assert events[completion_index]['type'] == 'tool_completed'
    assert completion_index < snapshot_index
    again, repeated = resume(box, journal)
    assert target.read_bytes() == current and repeated == messages
    assert again.stats.interrupted_calls == 1
    journal.close()


@pytest.mark.parametrize('name,args,pre,post', [
    ('write_file', {'path': 'target.txt', 'content': 'new\n'}, None, b'new\n'),
    ('write_file', {'path': 'target.txt', 'content': ''}, b'old', b''),
    ('edit_file', {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'alpha omega', 'replace_all': True}, b'alpha alpha\n', b'alpha omega alpha omega\n'),
    ('edit_file', {'path': 'target.txt', 'old_string': 'alpha', 'new_string': ''}, b'alpha\n', b'\n'),
    ('edit_file', {'path': 'target.txt', 'old_string': 'a\nb', 'new_string': 'c\nd'}, b'a\r\nb\r\n', b'c\r\nd\r\n'),
    ('edit_file', {'path': 'target.txt', 'old_string': '  value = 1', 'new_string': '  value = 2'}, b'    value = 1\n', b'    value = 2\n'),
    ('edit_file', {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'omega'}, 'alpha 😀\n'.encode(), 'omega 😀\n'.encode()),
])
def test_exact_postimage_or_preimage_without_replay(tmp_path, name, args, pre, post):
    box, journal, target = make_interrupted(tmp_path, name, dict(args), pre)
    if pre is not None:
        _, messages = resume(box, journal)
        assert messages[-1].startswith('ERROR')
        assert target.read_bytes() == pre
        # Use the original interrupted snapshot for a second independent check,
        # since the first reconciliation result is intentionally durable.
        journal.close()
        tmp_path = tmp_path / 'post'; tmp_path.mkdir()
        box, journal, target = make_interrupted(tmp_path, name, dict(args), pre)
    target.write_bytes(post)
    _, messages = resume(box, journal)
    assert messages[-1].startswith('OK') and 'exact expected' in messages[-1]
    assert target.read_bytes() == post
    journal.close()


@pytest.mark.parametrize('change', ['deleted', 'unrelated_same_length', 'directory', 'symlink'])
def test_existence_and_content_conflicts_preserved(tmp_path, change):
    args = {'path': 'target.txt', 'content': 'new'}
    box, journal, target = make_interrupted(tmp_path, 'write_file', args, b'old')
    if change == 'deleted': target.unlink()
    elif change == 'directory': target.unlink(); target.mkdir()
    elif change == 'symlink':
        outside = tmp_path.parent / (tmp_path.name + '-private'); outside.write_bytes(b'human')
        target.unlink(); target.symlink_to(outside)
    else: target.write_bytes(b'xyz')
    _, messages = resume(box, journal)
    assert messages[-1].startswith('ERROR')
    if change == 'deleted': assert not target.exists()
    elif change == 'directory': assert target.is_dir()
    elif change == 'symlink': assert target.is_symlink() and outside.read_bytes() == b'human'
    else: assert target.read_bytes() == b'xyz'
    journal.close()


@pytest.mark.parametrize('field', ['run_id', 'tool_id', 'tool', 'root_sha256', 'target_sha256', 'args_sha256', 'algorithm', 'format', 'semantics', 'pre', 'post'])
def test_mismatched_or_malformed_identity_fails_closed(tmp_path, field):
    args = {'path': 'target.txt', 'content': 'new'}
    box, journal, target = make_interrupted(tmp_path, 'write_file', args, b'old')
    target.write_bytes(b'new')
    events = journal.events()
    identity = copy.deepcopy(events[-1]['file_identity']); assert identity
    identity[field] = 'invalid'
    # Exercise the actual verifier with a corrupted/misbound recovered record.
    primary = SimpleNamespace(context_window=0, label=lambda: 'no-model')
    runner = Runner(SimpleNamespace(primary=primary, local_only=True), box, '', session=journal)
    runner.steps = 1
    assert runner._verify_recovered_file('write_file', args, '1:c', identity).startswith('ERROR')
    assert target.read_bytes() == b'new'
    journal.close()


@pytest.mark.parametrize('completed', [False, True])
def test_current_permissions_and_scope_apply_to_acknowledgement(tmp_path, completed):
    args = {'path': 'target.txt', 'content': 'new'}
    box, journal, target = make_interrupted(tmp_path, 'write_file', args, b'old', completed=completed)
    target.write_bytes(b'new')
    narrowed = ToolBox(tmp_path, profile='read-only')
    _, messages = resume(narrowed, journal)
    assert messages[-1].startswith('ERROR') and target.read_bytes() == b'new'
    journal.close()


def test_strict_mode_stops_even_with_exact_postimage(tmp_path):
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'new')
    runner, messages = resume(box, journal, resume_policy='strict')
    assert runner._recovery_error and not messages and target.read_bytes() == b'new'
    journal.close()


def test_durable_result_is_not_current_content_proof(tmp_path):
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old', completed=True)
    target.write_bytes(b'human')
    _, messages = resume(box, journal)
    assert messages[-1].startswith('ERROR') and target.read_bytes() == b'human'
    journal.close()


def test_result_verification_rejects_false_success(tmp_path):
    box = ToolBox(tmp_path); journal = Session.create(tmp_path / 's.jsonl')
    runner = Runner(None, box, '', session=journal); runner.steps = 1
    with patch.object(box, 'call', return_value='wrote file'):
        output = runner._run_tool(SimpleNamespace(id='c', name='write_file', args={'path': 'target.txt', 'content': 'new'}))
    assert output.startswith('ERROR') and not journal.events()[-1]['ok']
    journal.close()


def test_intervening_writer_during_recovery_never_gets_overwritten(tmp_path):
    import ordivyr_lite.recovery as rec
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    actual = rec.read_image
    def human_write_before_verification(box, path):
        target.write_bytes(b'human-racing')
        return actual(box, path)
    with patch.object(rec, 'read_image', human_write_before_verification):
        _, messages = resume(box, journal)
    assert target.read_bytes() == b'human-racing' and messages[-1].startswith('ERROR')
    journal.close()


def test_required_reconciliation_result_barrier_stops_prepare(tmp_path):
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'new')
    with patch.object(journal, 'flush', side_effect=OSError('forced fsync error')), pytest.raises(OSError):
        resume(box, journal)
    assert target.read_bytes() == b'new'
    journal.close()


def test_unknown_external_operations_never_replayed(tmp_path):
    for i, name in enumerate(['run_command', 'delegate', 'mcp_unknown']):
        work = tmp_path / str(i); work.mkdir()
        box, journal, target = make_interrupted(work, name, {}, None, legacy=True)
        _, messages = resume(box, journal)
        assert messages[-1].startswith('ERROR') and 'NOT run' in messages[-1]
        journal.close()


def test_sigkill_after_file_effect_before_result(tmp_path):
    """Real child death leaves only a flushed intent; recovery never calls tools."""
    code = r'''
import os, signal, sys
from pathlib import Path
from types import SimpleNamespace
from ordivyr_lite import filetools
from ordivyr_lite.loop import Runner
from ordivyr_lite.session import Session
from ordivyr_lite.tools import ToolBox
root = Path(sys.argv[1]); box = ToolBox(root)
args = {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'alpha omega'}
session = Session.create(root / 'session.jsonl', run_id='kill-test')
session.snapshot({'messages': [{'role': 'assistant', 'content': None, 'tool_calls': [
    {'id': 'c', 'type': 'function', 'function': {'name': 'edit_file', 'arguments': __import__('json').dumps(args)}}]}],
    'stats': {'model_calls': 1, 'tool_calls': 0}, 'effective_permissions': box.permission_manifest()})
runner = Runner(None, box, '', session=session); runner.steps = 1
original = filetools.atomic_write
def write_then_kill(path, content, **kwargs):
    original(path, content, **kwargs)
    os.kill(os.getpid(), signal.SIGKILL)
filetools.atomic_write = write_then_kill
runner._run_tool(SimpleNamespace(id='c', name='edit_file', args=args))
'''
    target = tmp_path / 'target.txt'; target.write_bytes(b'alpha\n')
    child = subprocess.run([sys.executable, '-c', code, str(tmp_path)], capture_output=True, timeout=10)
    assert child.returncode == -signal.SIGKILL
    journal = Session.open(tmp_path / 'session.jsonl')
    assert any(e['type'] == 'tool_started' and e['file_identity'] for e in journal.events())
    assert not any(e['type'] == 'tool_completed' for e in journal.events())
    assert target.read_bytes() == b'alpha omega\n'
    _, messages = resume(ToolBox(tmp_path), journal)
    assert messages[-1].startswith('OK') and target.read_bytes() == b'alpha omega\n'
    journal.close()


def test_duplicate_intents_and_ambiguous_edit_are_unknown(tmp_path):
    args = {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'omega'}
    box, journal, target = make_interrupted(tmp_path, 'edit_file', args, b'alpha alpha')
    intent = [e for e in journal.events() if e['type'] == 'tool_started'][0]
    assert intent['file_identity'] is None  # repeated matches without replace_all
    journal.append(intent); journal.flush()
    target.write_bytes(b'omega')
    _, messages = resume(box, journal)
    assert messages[-1].startswith('ERROR') and target.read_bytes() == b'omega'
    journal.close()


def test_scope_narrowing_prevents_even_exact_postimage_ack(tmp_path):
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'new')
    _, messages = resume(box, journal, scope=['other/'])
    assert messages[-1].startswith('ERROR') and 'scope' in messages[-1]
    assert target.read_bytes() == b'new'
    journal.close()


def test_multiple_overlapping_operations_keep_call_order_without_replay(tmp_path):
    first = {'path': 'target.txt', 'old_string': 'alpha', 'new_string': 'alpha omega'}
    box, journal, target = make_interrupted(tmp_path, 'edit_file', first, b'alpha\n')
    target.write_bytes(b'alpha omega\n')
    second = {'path': 'target.txt', 'old_string': 'alpha omega', 'new_string': 'done'}
    updated = state(box, 'edit_file', first)
    updated['messages'][-1]['tool_calls'].append({'id': 'd', 'type': 'function',
        'function': {'name': 'edit_file', 'arguments': json.dumps(second)}})
    journal.snapshot(updated)
    runner = Runner(None, box, '', session=journal); runner.steps = 1
    with patch.object(box, 'call', side_effect=Interrupted), pytest.raises(Interrupted):
        runner._run_tool(SimpleNamespace(id='d', name='edit_file', args=second))
    target.write_bytes(b'done\n')
    runner, messages = resume(box, journal)
    assert messages[0].startswith('ERROR') and messages[1].startswith('OK')
    assert target.read_bytes() == b'done\n' and runner.stats.interrupted_calls == 2
    journal.close()


def test_duplicate_valid_intents_cannot_be_accepted(tmp_path):
    box, journal, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    start = [e for e in journal.events() if e['type'] == 'tool_started'][0]
    assert start['file_identity']
    journal.append(start); journal.flush(); target.write_bytes(b'new')
    _, messages = resume(box, journal)
    assert messages[-1].startswith('ERROR') and target.read_bytes() == b'new'
    journal.close()


def test_intent_metadata_has_digests_not_new_plaintext_content(tmp_path):
    target = tmp_path / 'private-target.txt'; target.write_bytes(b'private-preimage')
    box = ToolBox(tmp_path)
    identity = capture_intent(box, 'write_file', {'path': target.name, 'content': 'private-postimage'}, 'run', '1:c')
    assert identity
    serialized = json.dumps(identity)
    for private in [str(tmp_path), target.name, 'private-preimage', 'private-postimage']:
        assert private not in serialized
