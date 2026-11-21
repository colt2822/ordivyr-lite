"""Real namespace checks: project PYTHONPATH must not discard active venv deps."""
import json
from pathlib import Path
import shutil
import sys
import venv

import pytest

from ordivyr_lite import sandbox


@pytest.mark.skipif(not shutil.which('bwrap'),reason='bubblewrap required')
def test_active_venv_survives_src_path_override_and_stays_readonly(tmp_path,monkeypatch):
    environment=tmp_path/'active environment';venv.EnvBuilder(with_pip=False).create(environment)
    packages=environment/f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    dependency=packages/'only_in_active_venv.py';dependency.write_text('VALUE=41\n')
    work=tmp_path/'project';(work/'src').mkdir(parents=True)
    (work/'src'/'project_value.py').write_text('VALUE=1\n')
    monkeypatch.setattr(sys,'prefix',str(environment))
    code=('import sys,json;from only_in_active_venv import VALUE as a;'
          'from project_value import VALUE as b;print(json.dumps([sys.prefix,a+b]));'
          'from pathlib import Path;'
          f'Path({str(dependency)!r}).write_text("bad")')
    import shlex
    result=sandbox.run('PYTHONPATH=/workspace/src python3 -c '+shlex.quote(code),work,
                       mode='bwrap',network=False,readonly_workspace=True)
    assert result.isolated and result.readonly_workspace and not result.network
    assert json.loads(result.stdout.splitlines()[0])==[str(environment),42]
    assert result.exit_code!=0 and ('Read-only file system' in result.stderr or 'Permission denied' in result.stderr)
    assert dependency.read_text()=='VALUE=41\n'
    assert (work/'src'/'project_value.py').read_text()=='VALUE=1\n'
