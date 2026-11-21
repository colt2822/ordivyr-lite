import os
import shlex
import shutil

import pytest

from ordivyr_lite import sandbox as SB
from ordivyr_lite.constraints import TaskConstraints
from ordivyr_lite.errors import SandboxError
from ordivyr_lite.loop import Runner,RunLimits
from ordivyr_lite.tests.support import FakeTransport,make_router,openai_reply
from ordivyr_lite.tools import ToolBox


@pytest.fixture
def protected(tmp_path):
    if not SB.bwrap_path():pytest.skip('bubblewrap required')
    try:
        probe=SB.run('true',tmp_path,mode='bwrap',readonly_workspace=True)
        if not probe.ok:pytest.skip('user namespace unavailable: '+probe.stderr)
    except SandboxError as exc:pytest.skip(str(exc))
    (tmp_path/'protected.txt').write_text('original bytes\n')
    box=ToolBox(tmp_path,profile='trusted',sandbox_mode='process',network=True)
    box.task_constraints=TaskConstraints.capture(tmp_path,'Leave protected.txt intact.')
    yield box
    assert (tmp_path/'protected.txt').read_text()=='original bytes\n'
    box.close()


def test_readonly_command_can_run_diagnostics_and_never_uses_writable_session(protected):
    class ForbiddenSession:
        def run(self,*args,**kwargs):raise AssertionError('writable session reused')
        def close(self):pass
    protected._sandbox=ForbiddenSession()
    out=protected.call('run_command',{'command':'python3 -c "print(2 + 3)"'})
    assert not out.startswith('ERROR') and '5' in out and 'workspace=read-only' in out
    assert protected.task_constraints.violation() is None


@pytest.mark.parametrize('command',[
    'echo overwritten > protected.txt',
    'rm protected.txt',
    'mv protected.txt renamed.txt',
    'mkdir generated',
    'mount -o remount,rw /workspace && echo overwritten > protected.txt',
])
def test_writes_renames_deletes_and_remounts_cannot_change_workspace(protected,command):
    assert protected.call('run_command',{'command':command}).startswith('ERROR')
    assert not (protected.root/'renamed.txt').exists()
    assert not (protected.root/'generated').exists()


def test_hardlink_and_symlink_aliases_are_readonly_too(protected):
    os.link(protected.root/'protected.txt',protected.root/'hardlink.txt')
    (protected.root/'symlink.txt').symlink_to('protected.txt')
    for path in ('hardlink.txt','symlink.txt'):
        assert protected.call('run_command',{'command':'echo overwritten > '+path}).startswith('ERROR')
    assert protected.task_constraints.violation() is None


def test_original_absolute_path_is_not_a_writable_workspace_alias(protected):
    root=shlex.quote(str(protected.root))
    target=shlex.quote(str(protected.root/'protected.txt'))
    # Host test directories are under /tmp; the command's /tmp is private.
    out=protected.call('run_command',{'command':f'mkdir -p {root}; echo private > {target}'})
    assert not out.startswith('ERROR'),out
    assert protected.task_constraints.violation() is None


def test_readonly_commands_have_no_external_network(protected):
    code="import socket; s=socket.socket(); s.settimeout(1); code=s.connect_ex(('1.1.1.1',443)); assert code != 0; print('network blocked')"
    out=protected.call('run_command',{'command':'python3 -c '+shlex.quote(code)})
    assert not out.startswith('ERROR') and 'network blocked' in out


@pytest.mark.skipif(not shutil.which('node'),reason='Node required')
def test_node_diagnostics_run_with_workspace_readonly(protected):
    out=protected.call('run_command',{'command':"node -e 'console.log(3 + 4)'"})
    assert not out.startswith('ERROR'),out
    assert '7' in out and 'workspace=read-only' in out


def test_readonly_api_never_falls_back_to_process_or_network(tmp_path,monkeypatch):
    monkeypatch.setattr(SB,'_run_popen',lambda *args,**kwargs:pytest.fail('unsafe fallback executed'))
    for kwargs in ({'mode':'process'},{'mode':'none'},{'mode':'bwrap','network':True}):
        with pytest.raises(SandboxError,match='require bubblewrap'):
            SB.run('echo unsafe',tmp_path,readonly_workspace=True,**kwargs)


def test_missing_bwrap_still_refuses_and_explicit_shell_denial_is_preserved(tmp_path,monkeypatch):
    (tmp_path/'protected.txt').write_text('original')
    box=ToolBox(tmp_path,profile='trusted',sandbox_mode='process')
    box.task_constraints=TaskConstraints.capture(tmp_path,'Leave protected.txt intact.')
    monkeypatch.setattr(SB,'bwrap_path',lambda:None)
    try:
        assert 'run_command' not in box.available_names()
        assert box.call('run_command',{'command':'echo bad > protected.txt'}).startswith('ERROR: task constraints')
        assert (tmp_path/'protected.txt').read_text()=='original'
    finally:box.close()
    monkeypatch.setattr(SB,'bwrap_path',lambda:'/usr/bin/bwrap')
    for kwargs in ({'profile':'read-only'},{'profile':'trusted','allow_shell':False},
                   {'profile':'trusted','allow_tools':['read_file']}):
        denied=ToolBox(tmp_path,**kwargs)
        try:assert not denied.readonly_commands_available()
        finally:denied.close()


def test_protected_runner_advertises_real_readonly_execution(protected):
    transport=FakeTransport([openai_reply(calls=[('run_command',{'command':'python3 -c "print(7)"'})]),openai_reply(text='Inspected.')])
    result=Runner(make_router(transport),protected,'Inspect the project. Leave protected.txt intact.',
                  stream=False,limits=RunLimits(max_turns=3)).run()
    assert result.stop_reason=='done' and result.stats.tool_errors==0
    schema=next(t['function'] for t in transport.requests[0]['tools'] if t['function']['name']=='run_command')
    assert 'READ-ONLY' in schema['description']
    assert 'workspace read-only' in transport.requests[0]['messages'][0]['content']
