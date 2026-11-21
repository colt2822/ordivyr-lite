"""Deterministic OS-boundary faults on real journals and real Runner methods."""
import copy
import errno
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import ordivyr_lite.session as storage
from ordivyr_lite.loop import Runner
from ordivyr_lite.session import JournalError, Session
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tests.test_recovery_integrity import make_interrupted, resume


def cleanup(session):
    # A poisoned close must still release its fd and surface the failure.
    try: session.close()
    except OSError: pass
    assert session._fd is None


@pytest.mark.parametrize('count', [0, 1, 8, -1])
def test_short_write_is_detected_and_session_cannot_continue(tmp_path, count):
    s = Session.create(tmp_path / 's.jsonl'); s.flush(); before = s.path.read_bytes()
    real = storage.os.write
    def short(fd, data):
        if count > 0: real(fd, data[:count])
        return count
    with patch.object(storage.os, 'write', short), pytest.raises(JournalError):
        s.append({'type': 'tool_completed', 'output': 'OK'})
    for operation in [lambda: s.append({'type': 'later'}), s.flush, s.close]:
        with pytest.raises(JournalError): operation()
    assert s._fd is None
    prefix = Session.open(s.path)
    assert not any(e['type'] == 'tool_completed' for e in prefix.events())
    if count > 0:
        assert prefix.integrity_error
        with pytest.raises(JournalError): prefix.require_healthy()
    else:
        assert prefix.path.read_bytes() == before


@pytest.mark.parametrize('operation', ['append', 'flush', 'close'])
@pytest.mark.parametrize('error', [errno.ENOSPC, errno.EIO])
def test_storage_errors_surface_and_stay_sticky(tmp_path, operation, error):
    s = Session.create(tmp_path / 's.jsonl'); s.flush()
    if operation == 'append':
        action = lambda: s.append({'type': 'test'})
        target = 'write'
    else:
        s.append({'type': 'pending'})
        action = getattr(s, operation); target = 'fsync'
    with patch.object(storage.os, target, side_effect=OSError(error, 'injected')), pytest.raises(OSError) as caught:
        action()
    assert caught.value.errno == error
    with pytest.raises(OSError): s.flush()
    cleanup(s)


def test_readers_keep_only_valid_prefix_and_refuse_success_suffix(tmp_path):
    s = Session.create(tmp_path / 's.jsonl'); s.snapshot({'messages': []})
    good = s.path.read_bytes()
    # Ordinary writes cannot append through a bad tail; inject the damage as a
    # crashed storage device would leave it, then append a plausible fake record.
    with s.path.open('ab') as stream:
        stream.write(b'{"type":broken\n{"type":"snapshot","state":{"pending_final_text":"FAKE SUCCESS"}}\n')
    prefix = s.events(); assert len(prefix) == 2
    assert s.latest_snapshot() == {'messages': []}
    assert s.integrity_error
    reopened = Session.open(s.path)
    with pytest.raises(JournalError): reopened.append({'type': 'later'})
    with pytest.raises(JournalError): reopened.flush()
    with pytest.raises(JournalError): reopened.require_healthy()
    # Inspection never truncates or heals owner evidence.
    assert s.path.read_bytes().startswith(good) and b'FAKE SUCCESS' in s.path.read_bytes()
    cleanup(s)


@pytest.mark.parametrize('damage', ['no_newline', 'gap', 'duplicate', 'reorder', 'checksum', 'length', 'future_version', 'downgrade', 'duplicate_key'])
def test_record_continuity_and_digest_damage_never_accepts_suffix(tmp_path, damage):
    s = Session.create(tmp_path / 's.jsonl'); s.snapshot({'messages': []})
    s.append({'type': 'tool_started'}); s.append({'type': 'tool_completed', 'output': 'OK'})
    s.flush(); lines = s.path.read_bytes().splitlines(keepends=True); s.close()
    if damage == 'no_newline': lines[-1] = lines[-1].rstrip(b'\n')
    elif damage == 'gap': lines.pop(2)
    elif damage == 'duplicate': lines.insert(2, lines[1])
    elif damage == 'reorder': lines[2], lines[3] = lines[3], lines[2]
    elif damage == 'duplicate_key': lines[2] = b'{"type":"tool_started","type":"tool_completed"}\n'
    else:
        row = json.loads(lines[2])
        if damage == 'checksum': row['type'] = 'changed'
        elif damage == 'length': row['_journal']['bytes'] += 1
        elif damage == 'future_version': row['_journal']['format'] = 'unsupported/99'
        elif damage == 'downgrade': del row['_journal']
        lines[2] = json.dumps(row).encode() + b'\n'
    s.path.write_bytes(b''.join(lines))
    reopened = Session.open(s.path)
    assert not any(e['type'] == 'tool_completed' for e in reopened.events())
    assert reopened.integrity_error
    with pytest.raises(JournalError): reopened.require_healthy()


def test_legacy_prefix_can_be_extended_with_checked_records(tmp_path):
    path = tmp_path / 'old.jsonl'
    path.write_text('{"type":"header","run_id":"legacy"}\n{"type":"snapshot","state":{"messages":[]}}\n')
    s = Session.open(path); assert s.latest_snapshot() == {'messages': []}
    s.append({'type': 'tool_started'}); s.flush(); s.close()
    reopened = Session.open(path); reopened.require_healthy()
    assert len(reopened.events()) == 3 and reopened.run_id == 'legacy'
    assert json.loads(path.read_bytes().splitlines()[-1])['_journal']['seq'] == 1


def test_concurrent_threads_use_one_contiguous_record_chain(tmp_path):
    s = Session.create(tmp_path / 's.jsonl')
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: s.append({'type': 'thread', 'n': n}), range(100)))
    s.flush(); s.close()
    reopened = Session.open(s.path); reopened.require_healthy()
    assert sorted(e['n'] for e in reopened.events() if e['type'] == 'thread') == list(range(100))


@pytest.mark.parametrize('phase', ['intent_append', 'intent_fsync', 'result_append', 'result_fsync'])
def test_required_tool_barriers_preserve_before_and_after_effect(tmp_path, phase):
    s = Session.create(tmp_path / 's.jsonl'); s.flush()
    box = ToolBox(tmp_path); r = Runner(None, box, '', session=s); r.steps = 1
    call = SimpleNamespace(id='c', name='write_file', args={'path': 'target.txt', 'content': 'new'})
    real_write = storage.os.write; real_sync = storage.os.fsync; event_type = {'value': None}
    def write(fd, data):
        row = json.loads(data); event_type['value'] = row['type']
        fail_type = 'tool_started' if phase == 'intent_append' else 'tool_completed' if phase == 'result_append' else None
        if row['type'] == fail_type:
            real_write(fd, data[:8]); return 8
        return real_write(fd, data)
    def sync(fd):
        fail_type = 'tool_started' if phase == 'intent_fsync' else 'tool_completed' if phase == 'result_fsync' else None
        if event_type['value'] == fail_type and fail_type:
            raise OSError(errno.EIO, 'injected fsync')
        return real_sync(fd)
    with patch.object(storage.os, 'write', write), patch.object(storage.os, 'fsync', sync), pytest.raises(OSError):
        r._run_tool(call)
    target = tmp_path / 'target.txt'
    assert target.exists() == phase.startswith('result')
    if target.exists(): assert target.read_bytes() == b'new'
    cleanup(s)


@pytest.mark.parametrize('phase', ['snapshot', 'commit', 'diagnostic', 'finish'])
def test_required_loop_barrier_failures_cannot_return_success(tmp_path, phase):
    s = Session.create(tmp_path / 's.jsonl'); s.flush()
    t = FakeTransport([openai_reply(text='finished')]); box = ToolBox(tmp_path)
    r = Runner(make_router(t), box, '', session=s)
    real_write = storage.os.write; real_sync = storage.os.fsync; last = {'type': None}
    def write(fd, data):
        row = json.loads(data); last['type'] = row['type']
        if phase == 'diagnostic' and row['type'] == 'run_start':
            real_write(fd, data[:8]); return 8
        return real_write(fd, data)
    def sync(fd):
        match = ((phase == 'snapshot' and last['type'] == 'snapshot')
                 or (phase == 'commit' and last['type'] == 'model_request_committed')
                 or (phase == 'finish' and last['type'] == 'stop'))
        if match: raise OSError(errno.EIO, 'injected required sync failure')
        return real_sync(fd)
    with patch.object(storage.os, 'write', write), patch.object(storage.os, 'fsync', sync), pytest.raises(OSError):
        r.run()
    if phase in {'snapshot', 'diagnostic'}: assert t.requests == []
    cleanup(s)


def test_resume_refuses_corrupt_success_and_preserves_human_bytes(tmp_path):
    box, s, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'human')
    with s.path.open('ab') as stream:
        stream.write(b'{\n{"type":"tool_completed","tool_id":"1:c","output":"OK"}\n')
    with pytest.raises(JournalError): resume(box, s)
    assert target.read_bytes() == b'human'
    cleanup(s)


def test_resume_required_sync_failure_never_acknowledges_postimage(tmp_path):
    box, s, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'new')
    with patch.object(storage.os, 'fsync', side_effect=OSError(errno.EIO, 'injected')), pytest.raises(OSError):
        resume(box, s)
    assert target.read_bytes() == b'new'
    assert not any(e.get('reconciled') for e in s.events())
    cleanup(s)


def test_close_syncs_unflushed_records(tmp_path):
    s = Session.create(tmp_path / 's.jsonl')
    actual = storage.os.fsync
    with patch.object(storage.os, 'fsync', wraps=actual) as sync:
        s.close(); assert sync.call_count == 1
    assert s._fd is None


@pytest.mark.parametrize('fault', ['append', 'fsync'])
def test_reconciliation_result_failure_is_not_published(tmp_path, fault):
    box, s, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old')
    target.write_bytes(b'new')
    real_write = storage.os.write; real_sync = storage.os.fsync; reconciled = {'last': False}
    def write(fd, data):
        reconciled['last'] = bool(json.loads(data).get('reconciled'))
        if fault == 'append' and reconciled['last']:
            real_write(fd, data[:8]); return 8
        return real_write(fd, data)
    def sync(fd):
        if fault == 'fsync' and reconciled['last']:
            raise OSError(errno.EIO, 'injected result sync')
        return real_sync(fd)
    callbacks = []
    with patch.object(storage.os, 'write', write), patch.object(storage.os, 'fsync', sync), pytest.raises(OSError):
        resume(box, s, on_event=callbacks.append)
    assert not any(e['type'] == 'tool_reconciled' for e in callbacks)
    assert target.read_bytes() == b'new'
    cleanup(s)


@pytest.mark.parametrize('current', [b'new', b'human'])
def test_simulated_lost_result_suffix_uses_content_evidence_only(tmp_path, current):
    box, s, target = make_interrupted(tmp_path, 'write_file', {'path': 'target.txt', 'content': 'new'}, b'old', completed=True)
    s.close()
    lines = s.path.read_bytes().splitlines(keepends=True)
    # Explicit storage simulation, not a claim of an actual fsync/power failure.
    result_index = next(i for i, line in enumerate(lines) if json.loads(line)['type'] == 'tool_completed')
    s.path.write_bytes(b''.join(lines[:result_index]))
    target.write_bytes(current)
    reopened = Session.open(s.path); reopened.require_healthy()
    _, messages = resume(box, reopened)
    assert target.read_bytes() == current
    assert messages[-1].startswith('OK') == (current == b'new')
    reopened.close()


def test_abandoned_request_barrier_failure_prevents_retry(tmp_path):
    s = Session.create(tmp_path / 's.jsonl')
    box = ToolBox(tmp_path)
    saved = {'messages': [{'role': 'user', 'content': 'task'}], 'stats': {'model_calls': 0},
             'effective_permissions': box.permission_manifest()}
    s.snapshot(saved); s.append({'type': 'model_request_started', 'request_id': 'request'}); s.flush()
    t = FakeTransport([openai_reply(text='must not run')])
    r = Runner(make_router(t), box, '', session=s, resume_state=saved)
    actual = storage.os.fsync
    def sync(fd):
        if s.events()[-1]['type'] == 'model_request_abandoned':
            raise OSError(errno.EIO, 'injected abandoned sync')
        return actual(fd)
    with patch.object(storage.os, 'fsync', sync), pytest.raises(OSError): r.run()
    assert t.requests == []
    cleanup(s)
