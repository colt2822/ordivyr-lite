from pathlib import Path
import shlex
import sys

import pytest

from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator
from ordivyr_lite.verification import inventory, select_tests, failure_kind
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply


def put(root, files):
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_transitive_relative_imports_and_src_layout(tmp_path):
    put(tmp_path, {'src/pkg/__init__.py': '', 'src/pkg/model.py': 'VALUE = 1',
                   'src/pkg/api.py': 'from .model import VALUE',
                   'test_api.py': 'from pkg.api import VALUE',
                   'test_other.py': 'def test_other(): pass'})
    s = select_tests(tmp_path, ['src/pkg/model.py'], inventory(tmp_path))
    assert s.tests == ['test_api.py'] and not s.broad


@pytest.mark.parametrize('changed', [['config.json'], ['gone.py'], []])
def test_uncertain_changes_require_full_suite(tmp_path, changed):
    put(tmp_path, {'test_a.py': 'def test_a(): pass', 'config.json': '{}'})
    assert select_tests(tmp_path, changed, inventory(tmp_path)).broad


def test_dynamic_import_falls_back(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=1', 'test_a.py': "m=__import__('lib')"})
    assert select_tests(tmp_path, ['lib.py'], inventory(tmp_path)).broad


def test_focused_green_still_requires_full_suite(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=1',
                   'test_a.py': 'from lib import VALUE\ndef test_a(): assert VALUE == 1',
                   'test_b.py': 'def test_b(): assert False'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Validator('auto')(box, ['lib.py'])
    assert result.ok is False
    assert [s['stage'] for s in result.evidence['stages']] == ['full']
    assert result.evidence['outside_selected_failure_nodes'] == ['test_b.py::test_b']
    assert not result.evidence['preexisting_failure_nodes']
    box.close()


def test_opt_in_baseline_labels_but_does_not_ignore_failure(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=1', 'test_a.py': 'def test_a(): assert False'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    v = Validator('auto', baseline_tests=True)
    v.prepare(box)
    result = v(box, ['lib.py'])
    assert result.ok is False
    assert result.evidence['preexisting_failure_nodes'] == ['test_a.py::test_a']
    box.close()


def test_same_size_same_mtime_content_change_invalidates_identity(tmp_path):
    import os
    put(tmp_path, {'test_a.py': 'assert True '})
    box = ToolBox(tmp_path)
    v = Validator()
    before = v.fingerprint(box)
    stat = (tmp_path / 'test_a.py').stat()
    (tmp_path / 'test_a.py').write_text('assert False')
    os.utime(tmp_path / 'test_a.py', ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert v.fingerprint(box) != before


def test_shell_only_edit_is_verified_and_repaired_preserving_user_file(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=0', 'test_lib.py':
                   'from lib import VALUE\ndef test_value(): assert VALUE == 2',
                   'user.txt': 'uncommitted user text\n'})
    def command(value):
        program = f"from pathlib import Path; Path('lib.py').write_text('VALUE={value}')"
        return ('run_command', {'command': shlex.join([sys.executable, '-c', program])})
    fake = FakeTransport([openai_reply(calls=[command(1)]), openai_reply(text='done'),
                          openai_reply(calls=[command(2)]), openai_reply(text='done')])
    box = ToolBox(tmp_path, profile='trusted', sandbox_mode='process')
    result = Runner(make_router(fake), box, 'Fix the value', stream=False,
                    validator=Validator('auto'), limits=RunLimits(max_turns=6)).run()
    assert result.stop_reason == 'done' and result.stats.repairs == 1
    assert result.completion['tests_passed'] is True
    assert result.completion['validation']['evidence']['changed'] == ['lib.py']
    assert (tmp_path / 'user.txt').read_text() == 'uncommitted user text\n'
    box.close()


def test_inventory_survives_validator_restart_and_tracks_deletions(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=0', 'test_lib.py': 'import lib'})
    box = ToolBox(tmp_path, sandbox_mode='process')
    v = Validator()
    v.prepare(box)
    restored = Validator()
    restored.restore(v.snapshot())
    (tmp_path / 'lib.py').unlink()
    result = restored(box, [])
    assert result.ok is False and result.evidence['selection']['broad']
    assert result.evidence['failure_kind'] == 'import_or_dependency'
    box.close()


def test_inventory_bounds_fail_closed(tmp_path, monkeypatch):
    import ordivyr_lite.verification as verification
    put(tmp_path, {'a.py': 'x=1', 'b.py': 'x=2'})
    monkeypatch.setattr(verification, 'MAX_FILES', 1)
    with pytest.raises(ValueError, match='limit'):
        inventory(tmp_path)


def test_new_test_disable_marker_cannot_turn_suite_green(tmp_path):
    put(tmp_path, {'lib.py': 'VALUE=0', 'test_a.py': 'def test_a(): assert False'})
    box = ToolBox(tmp_path)
    v = Validator()
    v.prepare(box)
    (tmp_path / '.ordivyr-lite-no-tests').touch()
    result = v(box, ['lib.py'])
    assert result.ok is False and 'disabled' in result.report


def test_symlink_directory_identity_changes_without_following_it(tmp_path):
    (tmp_path / 'link').symlink_to('/nonexistent-one', target_is_directory=True)
    first = inventory(tmp_path)
    (tmp_path / 'link').unlink()
    (tmp_path / 'link').symlink_to('/nonexistent-two', target_is_directory=True)
    assert inventory(tmp_path) != first


def test_failure_categories_do_not_treat_import_error_as_environment_proof():
    assert failure_kind('ModuleNotFoundError: service') == 'import_or_dependency'
    assert failure_kind('timed_out=True') == 'environment'
    assert failure_kind('AssertionError') == 'code_or_test'


def test_missing_python_hint_does_not_replay_prior_shell_mutation(tmp_path):
    put(tmp_path, {'counter.txt': ''})
    command = 'echo once >> counter.txt; python -c "print(1)"'
    fake = FakeTransport([openai_reply(calls=[('run_command', {'command': command})]),
                          openai_reply(text='blocked')])
    box = ToolBox(tmp_path, profile='trusted', sandbox_mode='process')
    result = Runner(make_router(fake), box, 'Run the check', stream=False,
                    limits=RunLimits(max_turns=3)).run()
    outputs = [m.get('content','') for m in result.messages if m['role']=='tool']
    assert any('python3' in o and 'already have run' in o for o in outputs)
    assert (tmp_path/'counter.txt').read_text() == 'once\n'
    box.close()


def test_build_guidance_and_budget_reminder_reuse_existing_validator(tmp_path):
    fake = FakeTransport([openai_reply(calls=[('write_file', {'path':'a.py','content':'X=1'})]),
                          openai_reply(text='done')])
    box = ToolBox(tmp_path)
    result = Runner(make_router(fake), box, 'Create a module', stream=False,
                    validator=Validator('syntax'), limits=RunLimits(max_turns=3)).run()
    assert not any('Two model turns remain' in str(m.get('content')) for m in result.messages)
    fake = FakeTransport([openai_reply(calls=[('write_file', {'path':'a.py','content':'X=2'})]),
                          openai_reply(text='done')])
    result = Runner(make_router(fake), box, 'Fix module', stream=False,
                    validator=Validator('auto'), limits=RunLimits(max_turns=3)).run()
    assert 'python3' in result.messages[0]['content']
    assert any('Two model turns remain' in str(m.get('content')) for m in result.messages)


def test_checkpoint_diagnoses_unchanged_defect_with_time_left_to_repair(tmp_path):
    put(tmp_path, {'lib.py':'VALUE=1', 'test_lib.py':
                   'from lib import VALUE\ndef test_value(): assert VALUE==2', 'owner.txt':'draft'})
    fixed = False
    def policy(body):
        nonlocal fixed
        if fixed:
            return openai_reply(text='done')
        if any('Verification checkpoint before' in str(m.get('content')) for m in body['messages']):
            fixed = True
            return openai_reply(calls=[('write_file', {'path':'lib.py','content':'VALUE=2'})])
        return openai_reply(calls=[('read_file', {'path':'lib.py'})])
    events=[]
    box=ToolBox(tmp_path,sandbox_mode='process')
    result=Runner(make_router(FakeTransport(policy)),box,'Fix the value',stream=False,
                  validator=Validator(),limits=RunLimits(max_turns=8),on_event=events.append).run()
    assert result.stop_reason=='done' and result.stats.model_calls==6
    assert result.completion['tests_passed'] is True
    checkpoints=[e for e in events if e['type']=='verification_checkpoint']
    assert len(checkpoints)==1 and checkpoints[0]['ok'] is False
    assert (tmp_path/'owner.txt').read_text()=='draft'
    box.close()


def test_green_checkpoint_is_reused_by_final_validation(tmp_path):
    put(tmp_path, {'lib.py':'VALUE=2', 'test_lib.py':
                   'from lib import VALUE\ndef test_value(): assert VALUE==2'})
    responses=[openai_reply(calls=[('read_file',{'path':'lib.py'})]) for _ in range(4)]
    responses.append(openai_reply(text='done'))
    box=ToolBox(tmp_path,sandbox_mode='process')
    result=Runner(make_router(FakeTransport(responses)),box,'Check the value',stream=False,
                  validator=Validator(),limits=RunLimits(max_turns=8)).run()
    assert result.stop_reason=='done' and result.stats.skipped_validations==1
    assert result.completion['tests_passed'] is True
    box.close()


def test_change_after_green_checkpoint_cannot_reuse_it(tmp_path):
    put(tmp_path, {'lib.py':'VALUE=2', 'test_lib.py':
                   'from lib import VALUE\ndef test_value(): assert VALUE==2'})
    calls=0
    def policy(body):
        nonlocal calls
        calls+=1
        if calls==5:
            (tmp_path/'lib.py').write_text('VALUE=3')
            return openai_reply(text='done')
        return openai_reply(calls=[('read_file',{'path':'lib.py'})])
    box=ToolBox(tmp_path,sandbox_mode='process')
    result=Runner(make_router(FakeTransport(policy)),box,'Check the value',stream=False,
                  validator=Validator(),limits=RunLimits(max_turns=8,max_repairs=0)).run()
    assert result.stop_reason=='unverified' and result.completion['tests_passed'] is False
    assert result.stats.skipped_validations==0
    box.close()


@pytest.mark.parametrize('profile,allowed', [('read-only',None), ('workspace',['read_file'])])
def test_auto_checkpoint_does_not_execute_code_with_only_read_permissions(tmp_path,profile,allowed):
    put(tmp_path, {'lib.py':'VALUE=2', 'test_effect.py':
                   "from pathlib import Path\ndef test_effect(): Path('effect.txt').write_text('ran')"})
    replies=[openai_reply(calls=[('read_file',{'path':'lib.py'})]) for _ in range(8)]
    box=ToolBox(tmp_path,profile=profile,allow_tools=allowed,sandbox_mode='process')
    result=Runner(make_router(FakeTransport(replies)),box,'Inspect the repository',stream=False,
                  validator=Validator(),limits=RunLimits(max_turns=8)).run()
    assert result.stop_reason=='steps'
    assert not (tmp_path/'effect.txt').exists()
    box.close()


def test_inventory_rejects_fifo_without_blocking(tmp_path):
    import os
    import subprocess
    os.mkfifo(tmp_path/'input.pipe')
    root=Path(__file__).resolve().parents[2]
    program=('import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);'
             'from ordivyr_lite.verification import inventory;inventory(Path(sys.argv[2]))')
    proc=subprocess.run([sys.executable,'-I','-c',program,str(root),str(tmp_path)],
                        capture_output=True,text=True,timeout=2)
    assert proc.returncode!=0 and 'non-regular' in proc.stderr


def test_turn_limited_result_keeps_required_completion_schema_fields(tmp_path):
    put(tmp_path,{'lib.py':'VALUE=2'})
    fake=FakeTransport([openai_reply(calls=[('read_file',{'path':'lib.py'})])])
    box=ToolBox(tmp_path,profile='read-only')
    result=Runner(make_router(fake),box,'Inspect the module',stream=False,
                  limits=RunLimits(max_turns=1)).run()
    required={'model_said_done','code_written','syntax_ok','tests_passed','independent_review'}
    assert required <= result.completion.keys()
    assert result.completion['model_said_done'] is False
    assert result.completion['tests_passed'] is None
    box.close()
