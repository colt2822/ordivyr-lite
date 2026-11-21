from __future__ import annotations

import shutil
import sys
import os

import pytest

from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator, detect_test_command


@pytest.fixture(autouse=True)
def close_sandbox_session():
    """Keep validator tests from leaking the process-wide sandbox lease."""
    yield
    from ordivyr_lite import sandbox
    active = sandbox._ACTIVE
    if active is not None:
        active.close()


@pytest.fixture
def work(tmp_path):
    (tmp_path / "lib.py").write_text("def add(a, b):\n    return a + b\n")
    (tmp_path / "test_lib.py").write_text("from lib import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    return tmp_path


def test_syntax_validator_passes_and_fails(work):
    box = ToolBox(work, profile="workspace")
    v = Validator("syntax")
    assert v(box, ["lib.py"]).ok is True
    (work / "bad.py").write_text("def broken(:\n")
    result = v(box, ["bad.py"])
    assert result.ok is False
    assert "SyntaxError" in result.report


def test_auto_validator_runs_detected_tests(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    box = ToolBox(work, profile="workspace")
    v = Validator("auto", sandbox_mode="bwrap")
    result = v(box, ["lib.py"])
    assert result.tests_run is True
    assert result.ok is True
    (work / "lib.py").write_text("def add(a, b):\n    return a - b\n")
    result = v(box, ["lib.py"])
    assert result.ok is False
    assert "1 failed" in result.report or "assert" in result.report
    box.close()


def test_command_validator(work):
    if not shutil.which("bwrap"):
        pytest.skip("bwrap unavailable")
    box = ToolBox(work, profile="workspace")
    v = Validator("command", "python3 -c 'import lib; assert lib.add(2,3)==5'",
                  sandbox_mode="bwrap")
    result = v(box, [])
    assert result.tests_run is True
    assert result.ok is True
    box.close()


def test_off_does_nothing(work):
    box = ToolBox(work, profile="workspace")
    assert Validator("off")(box, []).ok is None


def test_detect_test_command_none(work, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert detect_test_command(empty) is None


@pytest.mark.parametrize("name", ["pytest.py", "pytest", "conftest.py"])
def test_auto_refuses_workspace_pytest_shadows(work, name):
    target = work / name
    if name == "pytest":
        target.mkdir()
    else:
        target.write_text("raise SystemExit(0)\n")
    result = Validator("auto")(ToolBox(work, profile="workspace"), ["lib.py"])
    assert result.ok is False
    assert result.tests_run is False
    assert "agent_pytest_shadow" in result.report


def test_auto_ignores_project_addopts(work):
    command = detect_test_command(work)
    assert command is not None
    assert command[0] == "env"
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in command
    assert "-o" in command and "addopts=" in command
    assert "-p" in command and "no:cacheprovider" in command
    assert "-c" in command and os.devnull in command
    assert "--rootdir" in command and "." in command
    assert sys.executable in command


def test_workspace_path_pytest_and_fake_success_output_cannot_validate(work):
    shadow_dir = work / "bin"
    shadow_dir.mkdir()
    (shadow_dir / "pytest").write_text("#!/bin/sh\nprintf '1 passed\\n'\nexit 0\n")
    (shadow_dir / "pytest").chmod(0o755)
    (work / "test_lib.py").write_text(
        "def test_fake_green_output_is_not_success():\n"
        "    print('1 passed')\n"
        "    assert False, 'real assertion must fail'\n")
    # The sandbox PATH excludes workspace/bin; captured text cannot override
    # the trusted pytest exit status.
    result = Validator("auto", sandbox_mode="process")(ToolBox(work, profile="workspace"),
                                                            ["lib.py"])
    assert result.tests_run is True
    assert result.ok is False
    assert "real assertion must fail" in result.report


def test_pytest_project_config_is_not_loaded(work):
    (work / "pytest.ini").write_text("[pytest]\naddopts = -p fake_success\n")
    (work / "fake_success.py").write_text("def pytest_sessionfinish(session):\n    session.exitstatus = 0\n")
    command = detect_test_command(work)
    assert command is not None
    assert "-c" in command and os.devnull in command


def test_missing_pytest_dependency_fails_explicitly(work, monkeypatch):
    import ordivyr_lite.validators as validators
    real_find_spec = validators.importlib.util.find_spec
    monkeypatch.setattr(validators.importlib.util, "find_spec",
                        lambda name, *args, **kwargs: None if name == "pytest"
                        else real_find_spec(name, *args, **kwargs))
    result = Validator("auto")(ToolBox(work, profile="workspace"), ["lib.py"])
    assert result.ok is False
    assert "validator_dependency_missing" in result.report
