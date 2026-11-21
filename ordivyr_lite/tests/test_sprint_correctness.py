import hashlib
import json
from pathlib import Path

import pytest

from ordivyr_lite import requirements as REQ
from ordivyr_lite import filetools as FT
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator, detect_test_command
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply


def test_long_requirements_are_not_silently_lost():
    tail = 'preserve this full trailing behavior ' * 12
    prompt = '\n'.join('Must implement behavior %d with %s.' % (n, tail) for n in range(22))
    items = REQ.extract(prompt)
    assert len(items) == 22 and all(tail.strip() in i['text'] for i in items)


@pytest.mark.parametrize('extension', ['js', 'go'])
def test_nonpython_tests_directory_does_not_advertise_pytest(tmp_path, extension):
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests' / ('sample_test.' + extension)).write_text('')
    assert detect_test_command(tmp_path) is None


def test_legitimate_conftest_is_bound_to_initial_bytes(tmp_path):
    (tmp_path / 'conftest.py').write_text('import pytest\n@pytest.fixture\ndef number(): return 7\n')
    (tmp_path / 'test_sample.py').write_text('def test_sample(number): assert number == 7\n')
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('auto', sandbox_mode='process')
    validator.prepare(box)
    assert validator.checkpoint(box, []).ok is True
    (tmp_path / 'conftest.py').write_text('raise Exception("changed")\n')
    result = validator.checkpoint(box, [])
    assert result.ok is False and 'agent_pytest_shadow' in result.report
    box.close()


@pytest.mark.parametrize('tool', ['edit_file', 'write_file'])
def test_stale_read_blocks_whole_and_partial_edits(tmp_path, tool):
    target = tmp_path / 'value.txt'
    target.write_text('original\n')
    box = ToolBox(tmp_path)
    box.call('read_file', {'path': 'value.txt'})
    target.write_text('owner changed\n')
    args = {'path': 'value.txt', 'content': 'overwrite\n'} if tool == 'write_file' else {
        'path': 'value.txt', 'old_string': 'owner changed', 'new_string': 'overwrite'}
    assert box.call(tool, args).startswith('ERROR: stale file identity')
    assert target.read_text() == 'owner changed\n'
    box.call('read_file', {'path': 'value.txt'})
    assert not box.call(tool, args).startswith('ERROR:')
    box.close()


def test_change_during_patch_preparation_is_preserved(tmp_path, monkeypatch):
    target = tmp_path / 'value.txt'; target.write_text('old\n')
    box = ToolBox(tmp_path)
    original = FT.plan_edit
    def race(*args, **kwargs):
        plan = original(*args, **kwargs)
        target.write_text('owner racing\n')
        return plan
    monkeypatch.setattr(FT, 'plan_edit', race)
    output = box.call('edit_file', {'path': 'value.txt', 'old_string': 'old', 'new_string': 'new'})
    assert output.startswith('ERROR: stale file identity')
    assert target.read_text() == 'owner racing\n'
    assert not list(tmp_path.glob('.ordivyr-lite-*'))
    box.close()


def test_green_sparse_tests_trigger_bounded_behavior_review(tmp_path):
    (tmp_path / 'number.py').write_text('def value(x): return x\n')
    (tmp_path / 'test_number.py').write_text('from number import value\ndef test_old(): assert value(1) == 1\n')
    seen = []
    def policy(request):
        seen.append(request)
        if len(seen) == 1:
            return openai_reply(calls=[('edit_file', {'path': 'number.py', 'old_string': 'return x', 'new_string': 'return x or 0'})])
        return openai_reply(text='Complete.')
    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box,
        'Fix number.py. Must return zero for None and preserve positive numbers.', stream=False,
        validator=Validator('auto', sandbox_mode='process'), limits=RunLimits(max_turns=8)).run()
    assert len(seen) == 3 and result.completion['requirements_reviewed']
    notes = '\n'.join(str(m.get('content')) for m in result.messages)
    assert 'Before finishing, review the complete request' in notes
    assert 'reply now with one short sentence' not in notes
    assert result.completion['independent_review'] is False
    box.close()


def test_deleted_original_fixture_cannot_disable_test_environment(tmp_path):
    (tmp_path / 'conftest.py').write_text('')
    (tmp_path / 'test_x.py').write_text('def test_x(): assert True\n')
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('auto', sandbox_mode='process'); validator.prepare(box)
    (tmp_path / 'conftest.py').unlink()
    result = validator.checkpoint(box, [])
    assert result.ok is False and 'disappeared' in result.report
    box.close()


def test_resume_keeps_observed_content_identity(tmp_path):
    from ordivyr_lite.session import Session
    (tmp_path / 'value.txt').write_text('before')
    box = ToolBox(tmp_path)
    session = Session.create(tmp_path.parent / (tmp_path.name + '.jsonl'))
    router = make_router(FakeTransport([]))
    runner = Runner(router, box, 'Fix value.txt', session=session, stream=False)
    runner._prepare()
    box.call('read_file', {'path': 'value.txt'})
    runner._snapshot()
    saved = session.latest_snapshot()
    (tmp_path / 'value.txt').write_text('owner after interruption')
    resumed_box = ToolBox(tmp_path)
    resumed = Runner(router, resumed_box, 'Fix value.txt', session=session, resume_state=saved, stream=False)
    resumed._prepare()
    output = resumed_box.call('write_file', {'path': 'value.txt', 'content': 'stale overwrite'})
    assert output.startswith('ERROR: stale file identity')
    assert (tmp_path / 'value.txt').read_text() == 'owner after interruption'
    box.close(); resumed_box.close(); session.close()


def test_init_preserves_existing_configuration(tmp_path, monkeypatch, capsys):
    from ordivyr_lite.cli import main
    monkeypatch.setenv('ORDIVYR_LITE_HOME', str(tmp_path))
    args = ['init', '--model', 'test', '--base-url', 'http://127.0.0.1:1234/v1']
    assert main(args) == 0
    before = (tmp_path / 'config.json').read_bytes()
    assert main(args) != 0
    assert (tmp_path / 'config.json').read_bytes() == before


def test_cancel_during_generation_refuses_returned_mutation(tmp_path):
    import threading
    stop = threading.Event()
    (tmp_path / 'keep.txt').write_text('owner')
    def policy(request):
        stop.set()
        return openai_reply(calls=[('write_file', {'path': 'keep.txt', 'content': 'bad'})])
    box = ToolBox(tmp_path)
    result = Runner(make_router(FakeTransport(policy)), box, 'Edit keep.txt',
        cancel_event=stop, stream=False).run()
    assert result.stop_reason == 'cancelled' and result.stats.model_calls == 1
    assert result.stats.tool_calls == 0 and (tmp_path / 'keep.txt').read_text() == 'owner'
    box.close()


def test_init_key_reference_resolves_without_exposing_credential(tmp_path, monkeypatch, capsys):
    from ordivyr_lite.cli import main
    from ordivyr_lite.config import load_config, resolve_route
    monkeypatch.setenv('ORDIVYR_LITE_HOME', str(tmp_path / 'home'))
    key = tmp_path / 'key'; key.write_text('local-test-credential')
    assert main(['init', '--model', 'test', '--base-url', 'http://127.0.0.1:1234/v1',
                 '--api-key-file', str(key)]) == 0
    assert resolve_route(load_config()).has_key
    assert 'local-test-credential' not in capsys.readouterr().out


def test_removed_pytest_section_cannot_change_trusted_test_definition(tmp_path):
    config = tmp_path / 'pyproject.toml'
    config.write_text('[tool.pytest.ini_options]\npythonpath=["."]\n')
    (tmp_path / 'test_x.py').write_text('def test_x(): assert True\n')
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('auto'); validator.prepare(box)
    restored = Validator('auto'); restored.restore(validator.snapshot())
    config.write_text('[project]\nname="changed"\n')
    result = restored.checkpoint(box, [])
    assert result.ok is False and 'test_definition_changed' in result.report
    box.close()


def test_explicit_zero_repair_and_tool_budgets_survive_configuration():
    limits = RunLimits.from_config({'limits': {'max_tool_calls': 0, 'max_repairs': 0}})
    assert limits.max_tool_calls == 0 and limits.max_repairs == 0


def test_housekeeping_rows_do_not_force_repeated_checks_after_valid_completion(tmp_path):
    (tmp_path/'number.py').write_text('def square(x):\n    """Return x squared."""\n    raise NotImplementedError\n')
    (tmp_path/'test_number.py').write_text('from number import square\ndef test_value(): assert square(3)==9\n')
    transport=FakeTransport([
        openai_reply(calls=[('edit_file',{'path':'number.py','old_string':'raise NotImplementedError','new_string':'return x*x'})]),
        openai_reply(text='Complete.')])
    box=ToolBox(tmp_path,sandbox_mode='process')
    try:
        result=Runner(make_router(transport),box,
            'Implement `square` in number.py according to its existing documentation. Preserve the function signature. Do not modify tests.',
            stream=False,validator=Validator('auto',sandbox_mode='process'),limits=RunLimits(max_turns=8)).run()
        assert result.stop_reason=='done' and result.completion['verified']
        assert result.stats.model_calls==2 and not result.completion.get('requirements_reviewed')
        assert result.completion['independent_review'] is False
        import runpy
        assert runpy.run_path(str(tmp_path/'number.py'))['square'](-2)==4
    finally:box.close()
