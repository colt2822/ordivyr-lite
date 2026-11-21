import shutil

import pytest

from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator, detect_project_tests

pytestmark=pytest.mark.skipif(not shutil.which('node'),reason='Node required')


@pytest.mark.parametrize('suffix',['test.cjs','test.mjs','spec.js'])
def test_bare_node_suite_runs_without_package_json(tmp_path,suffix):
    (tmp_path/'value.cjs').write_text('exports.VALUE=0;\n')
    check=("import assert from 'node:assert/strict';import value from './value.cjs';assert.equal(value.VALUE,1);\n"
           if suffix.endswith('mjs') else "const a=require('node:assert/strict');a.equal(require('./value.cjs').VALUE,1);\n")
    (tmp_path/('value.'+suffix)).write_text(check)
    box=ToolBox(tmp_path,sandbox_mode='process');validator=Validator('auto',sandbox_mode='process')
    try:
        validator.prepare(box)
        assert validator.had_tests and validator.project_def
        assert not validator(box,['value.cjs']).ok
        (tmp_path/'value.cjs').write_text('exports.VALUE=1;\n')
        result=validator(box,['value.cjs'])
        assert result.ok and result.tests_run and result.kind=='tests'
    finally:box.close()


def test_bare_node_discovery_skips_symlinks_and_dependencies(tmp_path):
    (tmp_path/'node_modules').mkdir();(tmp_path/'node_modules/vendor.test.cjs').write_text('throw Error("bad")')
    (tmp_path/'linked.test.cjs').symlink_to('/outside/missing')
    assert detect_project_tests(tmp_path) is None
    (tmp_path/'real.test.cjs').write_text("require('node:assert/strict').equal(1,1);\n")
    project=detect_project_tests(tmp_path)
    assert project and project[1][-1]=='./real.test.cjs' and len(project[1])==4


def test_new_node_regressions_do_not_redefine_existing_test_command(tmp_path):
    (tmp_path/'value.test.cjs').write_text("require('node:assert/strict').equal(1,1);\n")
    box=ToolBox(tmp_path,sandbox_mode='process');validator=Validator('auto',sandbox_mode='process')
    try:
        validator.prepare(box)
        (tmp_path/'more.test.cjs').write_text("require('node:assert/strict').equal(2,2);\n")
        result=validator(box,['more.test.cjs'])
        assert result.ok and result.tests_run
    finally:box.close()
