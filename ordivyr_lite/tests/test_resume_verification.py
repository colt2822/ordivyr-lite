"""Interruption boundaries use actual local check effects and durable journals."""
from contextlib import closing
from pathlib import Path
import shlex
import sys

import pytest

from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.session import Session
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator
from ordivyr_lite import validators as VD
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply


class Interrupted(BaseException): pass


def router(replies):
    transport = FakeTransport(replies)
    return make_router(transport), transport


def test_resume_preserves_original_task_prompt(tmp_path):
    (tmp_path / 'work').mkdir()
    with closing(ToolBox(tmp_path / 'work', sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        first, _ = router([])
        original = Runner(first, box, 'Implement `parse_options` in api.py.', session=session, stream=False)
        original._prepare(); original._snapshot()
        again, _ = router([])
        resumed = Runner(again, box, '', session=session, resume_state=session.latest_snapshot(), stream=False)
        resumed._prepare()
        assert resumed.prompt == original.prompt
        assert session.latest_snapshot()['prompt'] == original.prompt


def test_resume_instruction_invalidates_pending_final_and_reaches_model(tmp_path):
    (tmp_path / 'work').mkdir()
    with closing(ToolBox(tmp_path / 'work', sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        first, _ = router([])
        original = Runner(first, box, 'Summarize the work.', session=session, stream=False)
        original._prepare(); original.pending_final_text = 'old completion'; original._snapshot()
        again, transport = router([openai_reply(calls=[('write_file', {'path':'marker.txt','content':'new requirement'})]), openai_reply(text='done')])
        resumed = Runner(again, box, 'Add `marker.txt` containing the new requirement.', session=session,
                         resume_state=session.latest_snapshot(), stream=False, validator=Validator('syntax'))
        result = resumed.run()
        assert result.stop_reason == 'done'
        assert (box.root / 'marker.txt').read_text() == 'new requirement'
        sent = transport.requests[0]['messages']
        assert any(m.get('role') == 'user' and 'Add `marker.txt`' in str(m.get('content')) for m in sent)
        assert any('marker.txt' in i['text'] for i in resumed.ledger.requirements)
        assert 'Summarize the work.' in resumed.prompt


@pytest.mark.parametrize('resumed_mode', ['command', 'auto', 'off'])
def test_interrupted_custom_check_never_runs_again_or_silently_completes(tmp_path, monkeypatch, resumed_mode):
    root = tmp_path / 'work'; root.mkdir()
    code = "from pathlib import Path; p=Path('check-effects.txt'); p.write_text((p.read_text() if p.exists() else '')+'once\\n')"
    command = shlex.join([sys.executable, '-c', code])
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        original_run_command = VD._run_command
        def interrupted(*args, **kwargs):
            result = original_run_command(*args, **kwargs)
            assert result[0] is True
            raise Interrupted()
        monkeypatch.setattr(VD, '_run_command', interrupted)
        first, _ = router([openai_reply(text='done')])
        original = Runner(first, box, 'Check the existing work.', validator=Validator('command', command, sandbox_mode='process'),
                          session=session, stream=False)
        with pytest.raises(Interrupted): original.run()
        assert (root / 'check-effects.txt').read_text() == 'once\n'
        monkeypatch.setattr(VD, '_run_command', original_run_command)
        for _ in range(2):
            again, transport = router([])
            resumed = Runner(again, box, '', validator=Validator(resumed_mode, command if resumed_mode == 'command' else None, sandbox_mode='process'),
                             session=session, resume_state=session.latest_snapshot(), stream=False)
            result = resumed.run()
            assert result.stop_reason == 'unverified'
            assert 'check' in str(result.error).lower()
            assert transport.requests == []
            assert (root / 'check-effects.txt').read_text() == 'once\n'
            assert not result.completion['verified']


def test_failed_custom_check_is_not_retried_by_finalization(tmp_path, monkeypatch):
    root = tmp_path / 'work'; root.mkdir(); (root / 'app.py').write_text('X=1\n')
    attempts = []
    def uncertain(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError('lost checker response')
    monkeypatch.setattr(VD, '_run_command', uncertain)
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        model, _ = router([openai_reply(text='done')])
        runner = Runner(model, box, 'Check app.py.', validator=Validator('command', 'operator-check'), session=session, stream=False)
        runner.wrote = ['app.py']  # ensure after-stop verification would otherwise run
        result = runner.run()
        assert result.stop_reason == 'unverified' and len(attempts) == 1
        assert not result.completion['verified_artifact_after_stop']


def test_resume_new_preservation_rule_keeps_old_protected_bytes(tmp_path):
    root = tmp_path / 'work'; root.mkdir()
    (root / 'old.txt').write_text('old owner bytes'); (root / 'new.txt').write_text('new owner bytes')
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        model, _ = router([])
        original = Runner(model, box, 'Keep `old.txt` unchanged.', session=session, stream=False)
        original._prepare(); original._snapshot()
        model, _ = router([])
        resumed = Runner(model, box, 'Keep `new.txt` unchanged.', session=session, resume_state=session.latest_snapshot(), stream=False)
        resumed._prepare(); resumed._snapshot()
        assert set(resumed.task_constraints.files) == {'old.txt', 'new.txt'}
        assert set(session.latest_snapshot()['task_constraints']['files']) == {'old.txt', 'new.txt'}
        assert resumed.task_constraints.refusal(box, 'write_file', {'path':'new.txt','content':'overwrite'})


def test_resume_discards_historical_post_stop_verification(tmp_path):
    root = tmp_path / 'work'; root.mkdir()
    (root / 'app.py').write_text('VALUE=1\n')
    (root / 'test_app.py').write_text('from app import VALUE\ndef test_value(): assert VALUE == 1\n')
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        model, _ = router([]); validator = Validator('tests', sandbox_mode='process')
        original = Runner(model, box, 'Check the application.', validator=validator, session=session, stream=False)
        original._prepare(); validator.prepare(box)
        original.pending_final_text = 'done'
        original.completion = {'verified_after_termination': {'ok':True,'tests_run':True}, 'tests_passed':True}
        original._snapshot(); (root / 'app.py').write_text('VALUE=2\n')
        model, transport = router([])
        resumed = Runner(model, box, '', validator=Validator('tests', sandbox_mode='process'), session=session,
                         resume_state=session.latest_snapshot(), stream=False)
        result = resumed.run()
        assert result.stop_reason == 'unverified' and transport.requests == []
        assert not result.completion['verified_artifact_after_stop']
        assert result.completion['tests_passed'] is False


def test_cli_restores_check_policy_and_respects_explicit_override(tmp_path):
    from types import SimpleNamespace
    from ordivyr_lite.cli import build_parser, _validator
    saved = {'verification': Validator('command', 'original-check', timeout=37).snapshot()}
    args = build_parser().parse_args(['resume', 'session'])
    box = SimpleNamespace(sandbox_mode='process')
    restored = _validator(args, {'validate':'auto'}, box, saved)
    assert (restored.mode, restored.command, restored.timeout, restored.sandbox_mode) == ('command','original-check',37,'process')
    args = build_parser().parse_args(['resume','session','--validate','syntax','--validate-timeout','18'])
    override = _validator(args, {'validate':'auto'}, box, saved)
    assert (override.mode, override.timeout) == ('syntax',18)


def test_sigkill_after_custom_check_effect_never_repeats_on_cli_resume(tmp_path):
    import json
    import os
    import subprocess
    import time
    from ordivyr_lite.tests.support import ScriptedServer
    from ordivyr_lite.tests.test_cli import cli
    checkout = Path(__file__).resolve().parents[2]
    work = tmp_path / 'work'; work.mkdir()
    home = tmp_path / 'home'; journal = tmp_path / 'killed.jsonl'; ready = tmp_path / 'ready'
    command = shlex.join([sys.executable, '-c', "from pathlib import Path; p=Path('counter.txt'); p.write_text((p.read_text() if p.exists() else '')+'once\\n')"])
    server = ScriptedServer(lambda body: openai_reply(text='complete')).start()
    code = f'''
from pathlib import Path
import threading
from ordivyr_lite import validators as V
original=V._run_command
def pause(*args,**kwargs):
    result=original(*args,**kwargs)
    assert result[0] is True
    Path({str(ready)!r}).write_text('effect complete')
    threading.Event().wait(60)
    return result
V._run_command=pause
from ordivyr_lite.cli import main
raise SystemExit(main(['run','Check the work.','--workdir',{str(work)!r},'--session',{str(journal)!r},
    '--provider','custom','--base-url',{server.base!r},'--model','test-model',
    '--sandbox','process','--check',{command!r},'--json']))
'''
    child = None
    try:
        env = {**os.environ, 'PYTHONPATH':str(checkout), 'ORDIVYR_LITE_HOME':str(home), 'PYTHONDONTWRITEBYTECODE':'1'}
        child = subprocess.Popen([sys.executable,'-c',code], cwd=checkout, env=env,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        end = time.monotonic() + 15
        while not ready.exists() and child.poll() is None and time.monotonic() < end: time.sleep(.02)
        assert ready.exists(), 'custom checker never reached the post-effect boundary'
        child.kill(); child.communicate(timeout=10)
        assert (work/'counter.txt').read_text() == 'once\n'
        with closing(Session.open(journal)) as saved:
            events = saved.events()
            assert sum(e.get('type') == 'validation_started' for e in events) == 1
            assert not any(e.get('type') == 'validation_completed' for e in events)
        before = len(server.requests)
        for extra in ([], ['--validate','off']):
            result = cli(home,'resume',str(journal),'--workdir',str(work),'--provider','custom',
                         '--base-url',server.base,'--model','test-model','--sandbox','process','--json',*extra)
            value = json.loads(result.stdout)
            assert value['stop_reason'] == 'unverified', (result.stdout,result.stderr)
            assert 'check' in value['error'].lower()
            assert not value['completion']['verified']
            assert (work/'counter.txt').read_text() == 'once\n'
            assert len(server.requests) == before
    finally:
        if child is not None:
            if child.poll() is None: child.kill(); child.communicate(timeout=10)
            if child.stderr: child.stderr.close()
        server.stop()


@pytest.mark.parametrize('boundary, expected_calls', [('validation_started',0), ('validation_completed',1)])
def test_custom_check_requires_durable_start_and_completion(tmp_path, monkeypatch, boundary, expected_calls):
    import errno
    from ordivyr_lite import session as storage
    root = tmp_path / 'work'; root.mkdir()
    box = ToolBox(root, sandbox_mode='process'); session = Session.create(tmp_path / 'run.jsonl')
    real_append = session.append; real_sync = storage.os.fsync; last = {'type':None}; effects = []
    def append(event):
        real_append(event); last['type'] = event.get('type')
    def sync(fd):
        if last['type'] == boundary: raise OSError(errno.EIO,'injected validation receipt sync failure')
        return real_sync(fd)
    monkeypatch.setattr(session,'append',append); monkeypatch.setattr(storage.os,'fsync',sync)
    monkeypatch.setattr(VD,'_run_command',lambda *a,**k: (effects.append('effect') or True,'ok',False))
    try:
        model, _ = router([openai_reply(text='done')])
        runner = Runner(model, box, 'Check existing work.', validator=Validator('command','operator-check'), session=session, stream=False)
        with pytest.raises(OSError): runner.run()
        assert len(effects) == expected_calls
        assert not runner.completion.get('verified',False)
    finally:
        monkeypatch.undo(); box.close()
        with pytest.raises(OSError): session.close()


def test_existing_session_cannot_bypass_uncertain_check_as_fresh_chat_turn(tmp_path):
    root = tmp_path / 'work'; root.mkdir()
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        session.append({'type':'validation_started','check_id':'uncertain','mode':'command','command':'old-check'}); session.flush()
        model, transport = router([openai_reply(text='done')])
        result = Runner(model,box,'Continue.',session=session,validator=Validator('off'),stream=False).run()
        assert result.stop_reason == 'unverified' and transport.requests == []


def test_new_instruction_is_durable_if_interrupted_during_tool_reconciliation(tmp_path, monkeypatch):
    import json
    from ordivyr_lite.ledger import Ledger
    root = tmp_path / 'work'; root.mkdir()
    with closing(ToolBox(root, sandbox_mode='process')) as box, closing(Session.create(tmp_path / 'run.jsonl')) as session:
        original_prompt='Read the workspace.'
        session.snapshot({'prompt':original_prompt,'ledger':Ledger.from_prompt(original_prompt).to_dict(),
            'messages':[{'role':'assistant','content':None,'tool_calls':[{'id':'read','type':'function','function':{'name':'read_file','arguments':json.dumps({'path':'missing.txt'})}}]}],
            'stats':{'model_calls':1},'effective_permissions':box.permission_manifest()})
        model, _=router([])
        interrupted=Runner(model,box,'Add `marker.txt`.',session=session,resume_state=session.latest_snapshot(),stream=False)
        original_snapshot=interrupted._snapshot
        def crash_after_snapshot():
            original_snapshot(); raise Interrupted()
        monkeypatch.setattr(interrupted,'_snapshot',crash_after_snapshot)
        with pytest.raises(Interrupted): interrupted._prepare()
        assert session.latest_snapshot()['pending_resume_instruction'] == 'Add `marker.txt`.'
        model, _=router([])
        resumed=Runner(model,box,'',session=session,resume_state=session.latest_snapshot(),stream=False)
        resumed._prepare(); resumed._snapshot()
        assert any(m.get('role')=='user' and m.get('content')=='Add `marker.txt`.' for m in resumed.messages)
        assert len([r for r in resumed.ledger.requirements if 'marker.txt' in r['text']]) == 1
        assert session.latest_snapshot()['pending_resume_instruction'] == ''


def test_second_interruption_does_not_erase_original_validation_baseline(tmp_path, monkeypatch):
    root = tmp_path / 'work'; root.mkdir()
    (root/'test_original.py').write_text('def test_original(): assert False\n')
    with closing(ToolBox(root,sandbox_mode='process')) as box, closing(Session.create(tmp_path/'run.jsonl')) as session:
        model,_=router([]); validator=Validator('auto',sandbox_mode='process')
        original=Runner(model,box,'Check existing work.',session=session,validator=validator,stream=False)
        original._prepare(); validator.prepare(box); original.pending_final_text='done'; original._snapshot()
        initial=session.latest_snapshot()['verification']['initial_files']
        model,_=router([])
        interrupted=Runner(model,box,'',session=session,resume_state=session.latest_snapshot(),
                           validator=Validator('auto',sandbox_mode='process'),stream=False)
        original_snapshot=interrupted._snapshot
        def crash_after_snapshot():
            original_snapshot(); raise Interrupted()
        monkeypatch.setattr(interrupted,'_snapshot',crash_after_snapshot)
        with pytest.raises(Interrupted): interrupted.run()
        saved=session.latest_snapshot()['verification']
        assert saved['initial_files'] == initial and saved['had_tests'] is True
        (root/'.ordivyr-lite-no-tests').touch()
        model,transport=router([])
        resumed=Runner(model,box,'',session=session,resume_state=session.latest_snapshot(),
                       validator=Validator('auto',sandbox_mode='process'),stream=False)
        result=resumed.run()
        assert result.stop_reason == 'unverified' and transport.requests == []
        assert 'disabled' in str(result.error)


def test_tool_reconciliation_snapshot_keeps_original_verifier_baseline(tmp_path,monkeypatch):
    import json
    root=tmp_path/'work'; root.mkdir(); (root/'test_original.py').write_text('def test_original(): pass\n')
    with closing(ToolBox(root,sandbox_mode='process')) as box, closing(Session.create(tmp_path/'run.jsonl')) as session:
        model,_=router([]); validator=Validator('auto',sandbox_mode='process')
        original=Runner(model,box,'Read the workspace.',session=session,validator=validator,stream=False)
        original._prepare(); validator.prepare(box)
        original.messages.append({'role':'assistant','content':None,'tool_calls':[{'id':'read','type':'function',
            'function':{'name':'read_file','arguments':json.dumps({'path':'test_original.py'})}}]})
        original.steps=1; original.stats.model_calls=1; original._snapshot()
        initial=session.latest_snapshot()['verification']['initial_files']
        model,_=router([])
        resumed=Runner(model,box,'',session=session,resume_state=session.latest_snapshot(),
                       validator=Validator('auto',sandbox_mode='process'),stream=False)
        save=resumed._snapshot
        def crash_after_snapshot(): save(); raise Interrupted()
        monkeypatch.setattr(resumed,'_snapshot',crash_after_snapshot)
        with pytest.raises(Interrupted): resumed._prepare()
        assert session.latest_snapshot()['verification']['initial_files'] == initial
        assert session.latest_snapshot()['verification']['had_tests'] is True
