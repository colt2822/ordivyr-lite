from __future__ import annotations

import json

import pytest

from ordivyr_lite import JOB_PROTOCOL, RESULT_PROTOCOL
from ordivyr_lite.config import load_config
from ordivyr_lite.errors import ConfigError, JobError
from ordivyr_lite.job import example_job, job_schema, run_job, validate_job
from ordivyr_lite.tests.support import ScriptedServer, file_tool_reply
from ordivyr_lite.tools import ToolBox


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("ORDIVYR_LITE_HOME", str(tmp_path / "home"))
    return tmp_path


def test_example_job_validates(home):
    job = validate_job(example_job(str(home)))
    assert job["protocol"] == JOB_PROTOCOL
    assert job["version"] == 1
    assert job["task"]["workdir"] == str(home)
    assert job["permissions"]["profile"] == "read-only"


def test_job_rejects_wrong_version(home):
    job = example_job(str(home))
    job["version"] = 2
    with pytest.raises(JobError, match="unsupported protocol version"):
        validate_job(job)


def test_job_rejects_inline_secret(home):
    job = example_job(str(home))
    job["model"]["api_key"] = "sk-nope"
    with pytest.raises(JobError, match="inline secret"):
        validate_job(job)


def test_job_rejects_unknown_field(home):
    job = example_job(str(home))
    job["mystery"] = True
    with pytest.raises(JobError, match="unknown job field"):
        validate_job(job)


def test_job_rejects_missing_prompt(home):
    job = example_job(str(home))
    job["task"]["prompt"] = ""
    with pytest.raises(JobError, match="task.prompt"):
        validate_job(job)


def test_job_permissions_keep_explicit_shell_grant_and_deny(home):
    job = example_job(str(home))
    job["permissions"] = {"profile": "workspace", "allow_shell": True}
    granted = validate_job(job)
    box = ToolBox(home, profile=granted["permissions"]["profile"],
                  allow_shell=granted["permissions"]["allow_shell"])
    assert box.has("run_command")

    job["permissions"] = {"profile": "trusted", "allow_shell": False}
    denied = validate_job(job)
    box = ToolBox(home, profile=denied["permissions"]["profile"],
                  allow_shell=denied["permissions"]["allow_shell"])
    assert not box.has("run_command")


def test_job_rejects_read_only_shell_grant(home):
    job = example_job(str(home))
    job["permissions"] = {"profile": "read-only", "allow_shell": True}
    with pytest.raises(JobError, match="read-only profile cannot enable"):
        validate_job(job)


def test_job_schema_is_versioned():
    schema = job_schema()
    assert schema["properties"]["protocol"]["const"] == JOB_PROTOCOL
    assert schema["properties"]["version"]["const"] == 1


def test_run_job_returns_versioned_result(home):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = home / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        job = example_job(str(work))
        job["model"] = {"provider": "custom", "model": "test-model", "base_url": srv.base}
        job["permissions"] = {"profile": "workspace", "sandbox": "auto"}
        job["validation"] = {"mode": "syntax"}
        job["session"] = {"enabled": False}
        result = run_job(job)
        assert result["protocol"] == RESULT_PROTOCOL
        assert result["version"] == 1
        assert result["status"] == "succeeded"
        assert result["stop_reason"] == "done"
        assert result["text"] == "done"
        assert result["usage"]["tool_calls"] == 2
        assert "b.txt" in result["artifacts"]
        assert (work / "b.txt").read_text() == "world\n"
    finally:
        srv.stop()


def test_job_cannot_authorise_paid_spending(home, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    cfg = load_config()
    cfg["prices"]["gpt-4o-mini"] = [0.1, 0.2]
    job = example_job(str(home))
    job["model"] = {"provider": "openai", "model": "gpt-4o-mini"}
    job["spend"] = {"allow_paid": True, "max_cost_usd": 1.0}
    with pytest.raises(ConfigError, match="not authorised"):
        run_job(job, cfg=cfg, allow_paid=False)


def test_job_resume_state_reuses_session(home):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = home / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        job = example_job(str(work))
        job["model"] = {"provider": "custom", "model": "test-model", "base_url": srv.base}
        job["permissions"] = {"profile": "workspace"}
        job["validation"] = {"mode": "syntax"}
        job["session"] = {"enabled": True, "path": str(home / "s.jsonl")}
        first = run_job(job)
        assert first["status"] == "succeeded"
        job2 = json.loads(json.dumps(job))
        job2["session"]["resume"] = True
        second = run_job(job2)
        assert second["session"]["path"].endswith("s.jsonl")
    finally:
        srv.stop()
