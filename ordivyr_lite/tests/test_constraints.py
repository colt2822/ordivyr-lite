import json
from pathlib import Path

import pytest

from ordivyr_lite.constraints import TaskConstraints, directives
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.session import Session
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply


def fixture(root):
    (root/'src').mkdir(); (root/'tests').mkdir()
    for path, text in {'src/value.py':'VALUE=0\n','tests/test_value.py':'def test_old(): assert True\n',
                       'package.json':'{"scripts":{"test":"node --test"}}\n','README.md':'owner\n'}.items():
        (root/path).write_text(text)


@pytest.mark.parametrize('prompt', [
    'Keep existing tests unchanged.', 'Never modify original test files.',
    'Existing tests must remain byte-for-byte unchanged.',
    'Fix src/value.py and preserve existing tests unchanged.',
])
def test_explicit_test_preservation_blocks_existing_bytes_but_allows_new_coverage(tmp_path, prompt):
    fixture(tmp_path)
    policy = TaskConstraints.capture(tmp_path, prompt)
    box = ToolBox(tmp_path); box.task_constraints = policy
    before = (tmp_path/'tests/test_value.py').read_bytes()
    out = box.call('write_file', {'path':'tests/test_value.py','content':'replaced'})
    assert out.startswith('ERROR: task constraint') and (tmp_path/'tests/test_value.py').read_bytes() == before
    assert not box.call('write_file', {'path':'tests/test_more.py','content':'def test_more(): assert True\n'}).startswith('ERROR')
    assert not box.call('write_file', {'path':'src/value.py','content':'VALUE=1\n'}).startswith('ERROR')
    assert policy.violation() is None
    box.close()


def test_config_and_explicit_path_protection_is_opt_in(tmp_path):
    fixture(tmp_path)
    plain = TaskConstraints.capture(tmp_path, 'Update tests and package.json.')
    assert not plain.files
    policy = TaskConstraints.capture(tmp_path, 'Keep original tests and configuration files unchanged. Leave `README.md` intact.')
    assert set(policy.files) == {'tests/test_value.py','package.json','README.md'}
    assert directives('Do not preserve existing tests.')[0] is False


def test_preservation_subject_does_not_capture_requested_source_edits(tmp_path):
    fixture(tmp_path)
    policy = TaskConstraints.capture(tmp_path,
        'Modify src/value.py; keep existing tests and config files unchanged.')
    assert set(policy.files) == {'tests/test_value.py','package.json'}


def test_protected_identity_survives_resume_and_external_changes_are_preserved(tmp_path):
    fixture(tmp_path)
    policy = TaskConstraints.capture(tmp_path, 'Keep existing tests unchanged.')
    saved = json.loads(json.dumps(policy.snapshot()))
    restored = TaskConstraints(tmp_path, saved['files'])
    target = tmp_path/'tests/test_value.py'; target.write_text('owner newer bytes')
    assert 'changed' in restored.violation() and target.read_text() == 'owner newer bytes'


def test_unknown_mutation_and_shell_do_not_bypass_protected_files(tmp_path):
    fixture(tmp_path)
    box = ToolBox(tmp_path, profile='trusted', sandbox_mode='process')
    box.task_constraints = TaskConstraints.capture(tmp_path, 'Keep existing tests unchanged.')
    out = box.call('run_command', {'command':'echo bad > tests/test_value.py'})
    assert out.startswith('ERROR:')
    assert (tmp_path/'tests/test_value.py').read_text() == 'def test_old(): assert True\n'
    box.close()


def test_runner_snapshots_constraints_and_never_edits_original_test(tmp_path):
    fixture(tmp_path)
    (tmp_path/'package.json').unlink()
    session = Session.create(tmp_path.parent/(tmp_path.name+'.jsonl'))
    transport = FakeTransport([
        openai_reply(calls=[('write_file',{'path':'tests/test_value.py','content':'bad'})]),
        openai_reply(calls=[('write_file',{'path':'src/value.py','content':'VALUE=1\n'})]),
        openai_reply(text='done')])
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(transport), box,
        'Fix the value in src/value.py. Keep existing tests unchanged.', session=session,
        stream=False, validator=Validator('syntax'), limits=RunLimits(max_turns=5)).run()
    assert result.stop_reason == 'done'
    assert result.completion['task_constraints']['protected_count'] == 1
    saved = session.latest_snapshot()
    assert 'tests/test_value.py' in saved['task_constraints']['files']
    assert (tmp_path/'tests/test_value.py').read_text() == 'def test_old(): assert True\n'
    assert (tmp_path/'src/value.py').read_text() == 'VALUE=1\n'
    box.close(); session.close()


def test_protected_fifos_fail_without_blocking(tmp_path):
    import os
    (tmp_path/'test_fifo.py').write_text('value')
    policy = TaskConstraints.capture(tmp_path, 'Keep existing tests unchanged.')
    (tmp_path/'test_fifo.py').unlink();os.mkfifo(tmp_path/'test_fifo.py')
    assert 'unsupported' in policy.violation()


def test_resumed_preservation_without_original_baseline_fails_closed(tmp_path):
    with pytest.raises(ValueError,match='baseline missing'):
        TaskConstraints.capture(tmp_path,'Keep existing tests unchanged.',resumed=True)


def test_common_preservation_subjects_and_non_file_semantics(tmp_path):
    fixture(tmp_path); (tmp_path/'LICENSE').write_text('owner');(tmp_path/'.env').write_text('placeholder')
    policy = TaskConstraints.capture(tmp_path, 'Existing tests and configuration are protected: keep every byte unchanged. '
        'Leave ./README.md unchanged. Keep `LICENSE` intact. Keep `.env` unchanged.')
    assert set(policy.files) == {'tests/test_value.py','package.json','README.md','LICENSE','.env'}
    assert not TaskConstraints.capture(tmp_path,'Keep test coverage unchanged. Keep configuration defaults unchanged.').files


def test_post_stop_validator_cannot_claim_acceptance_after_changing_protected_files(tmp_path):
    fixture(tmp_path)
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('command', command="python3 -c \"from pathlib import Path; Path('tests/test_value.py').write_text('changed by validator')\"",
                          sandbox_mode='process')
    runner = Runner(make_router(FakeTransport([])), box, 'Keep existing tests unchanged.', validator=validator, stream=False)
    runner._prepare();runner.wrote=['src/value.py']
    result = runner._finish('steps','')
    assert result.completion['task_constraints']['unchanged'] is False
    assert result.completion['verified_artifact_after_stop'] is False
    assert result.completion['verified_after_termination']['ok'] is False
    assert (tmp_path/'tests/test_value.py').read_text() == 'changed by validator'
    box.close()


def test_resume_stops_before_model_when_protected_original_changed(tmp_path):
    fixture(tmp_path)
    session = Session.create(tmp_path.parent/(tmp_path.name+'.jsonl'))
    box = ToolBox(tmp_path)
    router = make_router(FakeTransport([]))
    first = Runner(router,box,'Keep existing tests unchanged.',session=session,stream=False)
    first._prepare();first._snapshot();state=session.latest_snapshot()
    (tmp_path/'tests/test_value.py').write_text('owner after crash')
    second_box=ToolBox(tmp_path)
    result=Runner(router,second_box,'',session=session,resume_state=state,stream=False).run()
    assert result.stop_reason == 'unverified' and result.stats.model_calls == 0
    assert (tmp_path/'tests/test_value.py').read_text() == 'owner after crash'
    box.close();second_box.close();session.close()


def test_ambiguous_preservation_exception_is_reported_before_editing(tmp_path):
    fixture(tmp_path)
    with pytest.raises(ValueError,match='preservation exceptions'):
        TaskConstraints.capture(tmp_path,'Keep existing tests unchanged except tests/test_value.py.')
