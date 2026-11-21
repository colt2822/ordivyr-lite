from ordivyr_lite.loop import Runner
from ordivyr_lite.tests.support import FakeTransport, make_router
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator, ValidationResult


def runner(tmp_path):
    (tmp_path/'test_original.py').write_text('def test_original(): assert True\n')
    box=ToolBox(tmp_path)
    validator=Validator('auto',sandbox_mode='process')
    validator.prepare(box)
    return Runner(make_router(FakeTransport([])),box,'Implement requested behavior.',validator=validator,stream=False),box


def test_new_assertions_get_spec_review_but_original_failures_do_not(tmp_path):
    run,box=runner(tmp_path)
    try:
        run.wrote=['test_extra.py']
        failure=ValidationResult(False,'FAILED test_extra.py::test_boundary - AssertionError',kind='tests',tests_run=True)
        note=run._new_test_failure_note(failure)
        assert 'ADDED DURING THIS TASK' in note and 'original request' in note
        assert 'Do not weaken provided tests' in note
        assert run._new_test_failure_note(failure)==''
        original=ValidationResult(False,'FAILED test_original.py::test_original - AssertionError',kind='tests',tests_run=True)
        run.wrote.append('test_original.py')
        assert run._new_test_failure_note(original)==''
    finally:box.close()


def test_unknown_or_external_failures_never_get_agent_authorship_attribution(tmp_path):
    run,box=runner(tmp_path)
    try:
        run.wrote=['test_extra.py']
        for report in ('FAILED test_human.py::test_boundary', 'FAILED /outside/test_extra.py::test_boundary'):
            assert run._new_test_failure_note(ValidationResult(False,report,kind='tests'))==''
        assert run._new_test_failure_note(ValidationResult(True,'FAILED test_extra.py::test_boundary',kind='tests'))==''
    finally:box.close()


def test_node_test_failure_is_eligible_without_changing_the_test_policy(tmp_path):
    run,box=runner(tmp_path)
    try:
        run.wrote=['extra.test.cjs']
        assert 'extra.test.cjs' in run._new_test_failure_note(ValidationResult(False,'not ok 1 - extra.test.cjs',kind='tests'))
        assert box.task_constraints is None
    finally:box.close()


def test_isolated_validator_workspace_paths_map_back_to_written_files(tmp_path):
    run,box=runner(tmp_path)
    try:
        run.wrote=['extra.test.cjs']
        failure=ValidationResult(False,'not ok 1 - /workspace/extra.test.cjs',kind='tests',
                                 evidence={'stages':[{'isolated':True}]})
        assert 'extra.test.cjs' in run._new_test_failure_note(failure)
    finally:box.close()
