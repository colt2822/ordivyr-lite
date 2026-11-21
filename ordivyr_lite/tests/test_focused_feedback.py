"""Focused feedback saves suite launches without certifying a partial pass."""
from types import SimpleNamespace

from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator
from ordivyr_lite import requirements as REQ


def workspace(root):
    for name, text in {
        'calc.py': 'def add(a, b):\n    return a - b\n',
        'other.py': 'VALUE = 0\n',
        'test_calc.py': 'from calc import add\ndef test_add(): assert add(2, 3) == 5\n',
        'test_other.py': 'from other import VALUE\ndef test_other(): assert VALUE == 1\n',
    }.items():
        (root / name).write_text(text)


def test_selected_pass_cannot_reuse_cache_for_full_completion(tmp_path):
    workspace(tmp_path)
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('auto', sandbox_mode='process')
    runner = Runner(make_router(FakeTransport(lambda b: openai_reply(text='done'))), box,
                    'Fix addition in calc.py.', stream=False, validator=validator)
    validator.prepare(box)
    (tmp_path / 'calc.py').write_text('def add(a, b):\n    return a + b\n')
    runner.wrote = ['calc.py']
    observed_key, observed = runner._verify_current_state(observation=True)
    assert observed.ok is True and observed.evidence['full_suite'] is False
    assert [s['stage'] for s in observed.evidence['stages']] == ['focused']
    final_key, final = runner._verify_current_state()
    assert observed_key != final_key
    assert final.ok is False and 'test_other' in final.report
    assert [s['stage'] for s in final.evidence['stages']] == ['full']
    assert runner._verify_current_state()[1] is final


def test_green_edit_feedback_does_not_finish_with_unrelated_red_suite(tmp_path):
    workspace(tmp_path)
    calls = 0

    def policy(body):
        nonlocal calls
        calls += 1
        if calls == 1:
            return openai_reply(calls=[('edit_file', {'path': 'calc.py', 'old_string': 'a - b', 'new_string': 'a + b'})])
        return openai_reply(text='done')

    box = ToolBox(tmp_path, sandbox_mode='process')
    result = Runner(make_router(FakeTransport(policy)), box, 'Fix addition in calc.py.', stream=False,
                    validator=Validator('auto', sandbox_mode='process'),
                    limits=RunLimits(max_turns=5, max_repairs=0)).run()
    assert result.stop_reason == 'unverified'
    assert result.completion['verified'] is False
    assert any('full-suite verification is still pending' in m.get('content', '')
               for m in result.messages if m.get('role') == 'tool')


def test_broad_selection_still_runs_full_suite_during_observation(tmp_path):
    workspace(tmp_path)
    box = ToolBox(tmp_path, sandbox_mode='process')
    validator = Validator('auto', sandbox_mode='process')
    validator.prepare(box)
    (tmp_path / 'unmapped.py').write_text('X = 1\n')
    result = validator.observe(box, ['unmapped.py'])
    assert result.ok is False
    assert result.evidence['full_suite'] is True
    assert [s['stage'] for s in result.evidence['stages']] == ['full']


def test_prefetch_racing_change_does_not_mark_old_snapshot_reusable(tmp_path, monkeypatch):
    (tmp_path / 'calc.py').write_text('VALUE = 1\n')
    box = ToolBox(tmp_path, sandbox_mode='process')
    runner = Runner(make_router(FakeTransport(lambda b: openai_reply(text='done'))), box,
                    'Fix VALUE in calc.py.', stream=False)
    original_call = box.call

    def raced(name, args):
        out = original_call(name, args)
        if name == 'read_file': (tmp_path / 'calc.py').write_text('VALUE = 2\n')
        return out

    monkeypatch.setattr(box, 'call', raced)
    shown = runner._prefetch_block()
    assert 'VALUE = 1' in shown and 'calc.py' not in runner._prefetched
    monkeypatch.setattr(box, 'call', original_call)
    current = runner._run_tool(SimpleNamespace(name='read_file', args={'path': 'calc.py'}, id='r'))
    assert 'VALUE = 2' in current and 'unchanged since' not in current


def test_named_function_in_wrong_module_cannot_satisfy_requirement(tmp_path):
    (tmp_path / 'api.py').write_text('VALUE = 1\n')
    (tmp_path / 'test_api.py').write_text('def parse_options(): pass\n')
    items = REQ.extract('Add a function `parse_options` to `api.py`.')
    REQ.evaluate(items, tmp_path)
    assert items[0]['status'] == 'open'
    (tmp_path / 'api.py').write_text('def parse_options(): return []\n')
    REQ.evaluate(items, tmp_path)
    assert items[0]['status'] == 'satisfied'


def test_missing_named_target_is_open_even_without_other_python_files(tmp_path):
    items = REQ.extract('Implement `parse_options` in api.py.')
    REQ.evaluate(items, tmp_path)
    assert items[0]['status'] == 'open'
