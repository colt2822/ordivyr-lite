"""Native CLI launch is explicit, credential-free and never an API fallback."""
import hashlib
import json
import os
from pathlib import Path
import pty
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ordivyr_lite import cli,native
from ordivyr_lite.errors import ConfigError


def test_check_does_not_load_credentials_or_start_client(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(cli,'load_config',lambda:pytest.fail('native check must not load Lite configuration'))
    monkeypatch.setattr(native.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(native.os,'execv',lambda *args:pytest.fail('check must not start any CLI'))
    assert cli.main(['native','claude','--check','--workdir',str(tmp_path)])==0
    state=json.loads(capsys.readouterr().out)
    assert state['installed'] and state['application']=='Claude Code'
    assert not (tmp_path/'config.json').exists()


def test_missing_cli_is_actionable_and_never_installs(tmp_path,monkeypatch,capsys):
    monkeypatch.setattr(native.shutil,'which',lambda name:None)
    assert native.launch('claude',tmp_path,check=True)==2
    assert json.loads(capsys.readouterr().out)['installed'] is False
    with pytest.raises(ConfigError,match='not on PATH'):native.launch('claude',tmp_path)


def test_noninteractive_handoff_refuses_before_exec(tmp_path,monkeypatch):
    monkeypatch.setattr(native.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(native.sys,'stdin',SimpleNamespace(isatty=lambda:False))
    monkeypatch.setattr(native.os,'execv',lambda *args:pytest.fail('noninteractive execution is not a handoff'))
    with pytest.raises(ConfigError,match='interactive terminal'):native.launch('claude',tmp_path)


def test_workdir_and_client_are_validated(tmp_path):
    with pytest.raises(ConfigError):native.inspect('claude;echo bad',tmp_path)
    with pytest.raises(ConfigError,match='existing project'):native.inspect('claude',tmp_path/'missing')


def test_handoff_inherits_auth_environment_without_loading_lite_keys(tmp_path,monkeypatch):
    project=tmp_path/'project';project.mkdir();before=Path.cwd();seen=[]
    monkeypatch.setattr(native.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(native.sys,'stdin',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(native.sys,'stdout',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setenv('ANTHROPIC_API_KEY','existing-shell-key')
    environment=dict(os.environ)
    def fake_exec(path,argv):seen.append((path,argv,Path.cwd(),dict(os.environ)))
    monkeypatch.setattr(native.os,'execv',fake_exec)
    assert native.launch('claude',project)==0
    assert seen==[('/usr/bin/claude',['/usr/bin/claude'],project,environment)]
    assert Path.cwd()==before and dict(os.environ)==environment


def test_exec_failure_restores_working_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(native.shutil,'which',lambda name:'/usr/bin/'+name)
    monkeypatch.setattr(native.sys,'stdin',SimpleNamespace(isatty=lambda:True))
    monkeypatch.setattr(native.sys,'stdout',SimpleNamespace(isatty=lambda:True))
    def fail(*args):raise OSError('cannot exec')
    monkeypatch.setattr(native.os,'execv',fail);before=Path.cwd()
    with pytest.raises(ConfigError,match='Could not start'):native.launch('claude',tmp_path)
    assert Path.cwd()==before


@pytest.mark.parametrize('client',['claude','codex','gemini'])
def test_real_terminal_exec_preserves_client_status_config_and_auth_source(tmp_path,client):
    binary=tmp_path/'bin';binary.mkdir();project=tmp_path/'project with space';project.mkdir()
    report=tmp_path/'native-report.json';home=tmp_path/'lite-home';home.mkdir()
    config=home/'config.json';config.write_text('{"providers":{"anthropic":{"api_key":"LITE_KEY_MUST_NOT_BE_EXPORTED"}}}')
    before=hashlib.sha256(config.read_bytes()).hexdigest()
    fake=binary/client
    fake.write_text('#!/usr/bin/python3\nimport json,os,sys\nfrom pathlib import Path\nPath(os.environ["NATIVE_TEST_REPORT"]).write_text(json.dumps({"argv":sys.argv,"cwd":os.getcwd(),"auth":os.environ.get("ANTHROPIC_API_KEY")}))\nraise SystemExit(17)\n')
    fake.chmod(0o755)
    env={**os.environ,'PATH':str(binary)+':/usr/bin:/bin','PYTHONPATH':str(Path(__file__).resolve().parents[2]),'NATIVE_TEST_REPORT':str(report),'ANTHROPIC_API_KEY':'SHELL_KEY_REMAINS_NATIVE'}
    master,slave=pty.openpty()
    try:
        child=subprocess.Popen([sys.executable,'-m','ordivyr_lite','--home',str(home),'native',client,'--workdir',str(project)],stdin=slave,stdout=slave,stderr=slave,env=env)
        os.close(slave);slave=None
        assert child.wait(timeout=10)==17
    finally:
        if slave is not None:os.close(slave)
        os.close(master)
        if 'child' in locals() and child.poll() is None:child.kill();child.wait()
    data=json.loads(report.read_text())
    assert data=={'argv':[str(fake)],'cwd':str(project),'auth':'SHELL_KEY_REMAINS_NATIVE'}
    assert hashlib.sha256(config.read_bytes()).hexdigest()==before
    assert sorted(p.name for p in home.iterdir())==['config.json']
