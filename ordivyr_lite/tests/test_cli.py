from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ordivyr_lite.tests.support import ScriptedServer, file_tool_reply, openai_reply

ROOT = Path(__file__).resolve().parents[2]


def cli(home: Path, *args: str, input_text: str | None = None, env_extra: dict | None = None):
    env = {**os.environ, "ORDIVYR_LITE_HOME": str(home),
           "PYTHONPATH": str(ROOT), "PYTHONDONTWRITEBYTECODE": "1", **(env_extra or {})}
    return subprocess.run([sys.executable, "-m", "ordivyr_lite", *args],
                          input=input_text, capture_output=True, text=True, timeout=120,
                          cwd=str(ROOT), env=env)


@pytest.fixture
def home(tmp_path):
    return tmp_path / "home"


def test_run_json_and_session(home, tmp_path):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        session = tmp_path / "s.jsonl"
        p = cli(home, "run", "read then write", "--workdir", str(work),
                "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                "--permission", "workspace", "--validate", "syntax",
                "--session", str(session), "--json")
        assert p.returncode == 0, p.stderr
        result = json.loads(p.stdout)
        assert result["status"] == "succeeded"
        assert result["usage"]["tool_calls"] == 2
        assert (work / "b.txt").read_text() == "world\n"
        assert session.exists()
        assert result["session_path"] == str(session)
        info = cli(home, "sessions", "show", str(session))
        assert json.loads(info.stdout)["path"] == str(session)
    finally:
        srv.stop()


def test_usage_report_after_run(home, tmp_path):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        p = cli(home, "run", "read then write", "--workdir", str(work),
                "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                "--no-session", "--json")
        assert p.returncode == 0, p.stderr
        u = cli(home, "usage", "--json")
        assert u.returncode == 0, u.stderr
        report = json.loads(u.stdout)
        assert report["totals"]["calls"] == 3
        assert report["by_model"]["test-model"]["calls"] == 3
    finally:
        srv.stop()


def test_doctor_json(home):
    p = cli(home, "doctor", "--json")
    assert p.returncode == 0, p.stderr
    report = json.loads(p.stdout)
    assert report["python"]["ok"] is True
    assert "bubblewrap" in report


def test_config_show_redacts(home):
    p = cli(home, "config", "show")
    assert p.returncode == 0, p.stderr
    cfg = json.loads(p.stdout)
    assert "providers" in cfg and "limits" in cfg
    assert "secret" not in p.stdout.lower()


def test_job_validate_and_run(home, tmp_path):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        job = {
            "protocol": "ordivyr-lite/job", "version": 1, "job_id": "j1",
            "task": {"prompt": "read then write", "workdir": str(work)},
            "model": {"provider": "custom", "model": "test-model", "base_url": srv.base},
            "permissions": {"profile": "workspace"},
            "validation": {"mode": "syntax"},
            "session": {"enabled": False},
        }
        valid = cli(home, "job", "validate", "-f", "-", input_text=json.dumps(job))
        assert valid.returncode == 0, valid.stderr
        run = cli(home, "job", "run", "-f", "-", input_text=json.dumps(job))
        assert run.returncode == 0, run.stderr
        result = json.loads(run.stdout)
        assert result["protocol"] == "ordivyr-lite/result"
        assert result["status"] == "succeeded"
        assert result["job_id"] == "j1"
        assert (work / "b.txt").read_text() == "world\n"
    finally:
        srv.stop()


def test_job_invalid_version(home):
    bad = {"protocol": "ordivyr-lite/job", "version": 99, "task": {"prompt": "x"},
           "model": {"provider": "ollama", "model": "m"}}
    p = cli(home, "job", "validate", "-f", "-", input_text=json.dumps(bad))
    assert p.returncode == 2
    assert "unsupported protocol version" in p.stderr


def test_dry_run_makes_no_model_call(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "hello", "--workdir", str(work), "--provider", "llamacpp",
            "--model", "m", "--dry-run")
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert plan["route"]["provider"] == "llamacpp"
    assert plan["tools"] == []


def test_dry_run_keeps_tools_for_a_file_task(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "read NOTE.txt and change the total", "--workdir", str(work),
            "--provider", "llamacpp", "--model", "m", "--dry-run")
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert "read_file" in plan["tools"]
    assert "edit_file" in plan["tools"]


def test_explicit_permission_keeps_tools_on_a_reply(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "Reply with the single word pong.", "--workdir", str(work),
            "--provider", "llamacpp", "--model", "m", "--permission", "workspace", "--dry-run")
    assert p.returncode == 0, p.stderr
    assert "read_file" in json.loads(p.stdout)["tools"]


def test_paid_without_max_cost_is_refused(home, tmp_path):
    p = cli(home, "run", "hello", "--workdir", str(tmp_path), "--provider", "openai",
            "--model", "gpt-4o-mini", "--allow-paid", "--dry-run")
    assert p.returncode == 2
    assert "not authorised" in p.stderr or "no credential" in p.stderr


def test_chat_one_shot_input(home, tmp_path):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        p = cli(home, "chat", "--workdir", str(work), "--provider", "custom",
                "--base-url", srv.base, "--model", "test-model", "--no-session",
                input_text="read then write\n/exit\n")
        assert p.returncode == 0, p.stderr
        assert (work / "b.txt").read_text() == "world\n"
        assert "done" in p.stdout
    finally:
        srv.stop()


def test_resume_is_promptless(home, tmp_path):
    srv = ScriptedServer(file_tool_reply).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "a.txt").write_text("hello\n")
        session = tmp_path / "s.jsonl"
        first = cli(home, "run", "read then write", "--workdir", str(work),
                    "--provider", "custom", "--base-url", srv.base,
                    "--model", "test-model", "--session", str(session),
                    "--max-turns", "1", "--validate", "off", "--json")
        assert first.returncode == 6, first.stderr
        assert session.exists()
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--validate", "off", "--json")
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout)
        assert result["status"] == "succeeded"
        assert (work / "b.txt").read_text() == "world\n"
    finally:
        srv.stop()


def test_process_kill_during_model_response_fails_closed_without_duplicate_request(home, tmp_path):
    srv = ScriptedServer(lambda _body: openai_reply(text="finished"), delay=10).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        session = tmp_path / "response-crash.jsonl"
        env = {**os.environ, "ORDIVYR_LITE_HOME": str(home), "PYTHONPATH": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1"}
        code = (
            "from ordivyr_lite.cli import main; import sys; "
            "sys.exit(main(['run','recover response','--workdir'," + repr(str(work))
            + ",'--provider','custom','--base-url'," + repr(srv.base)
            + ",'--model','test-model','--validate','off','--session',"
            + repr(str(session)) + ",'--json']))")
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        while not srv.requests and time.monotonic() < deadline:
            time.sleep(0.02)
        assert srv.requests, "child never reached model response"
        proc.kill()
        proc.communicate(timeout=10)
        from ordivyr_lite.session import Session
        assert Session.open(session).latest_snapshot() is not None
        srv.delay = 0
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--validate", "off", "--json",
                      env_extra={"ORDIVYR_LITE_RESUME": "strict"})
        assert resumed.returncode != 0
        result = json.loads(resumed.stdout)
        assert result["stop_reason"] == "unverified"
        assert "duplicate billing" in result["error"]
        assert len(srv.requests) == 1
        # Default policy: the interrupted turn is asked again and the run finishes;
        # the possibly-billed request is counted, not hidden.
        again = cli(home, "resume", str(session), "--workdir", str(work),
                    "--provider", "custom", "--base-url", srv.base,
                    "--model", "test-model", "--validate", "off", "--json")
        retried = json.loads(again.stdout)
        assert retried["stop_reason"] == "done", again.stdout + again.stderr
        assert len(srv.requests) == 2
        assert retried["usage"]["uncertain_requests"] == 1
    finally:
        srv.stop()


def test_process_kill_after_tool_effect_is_never_replayed(home, tmp_path):
    srv = ScriptedServer(lambda _body: openai_reply(
        calls=[("write_file", {"path": "effect.txt", "content": "once"})])).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        session = tmp_path / "tool-crash.jsonl"
        signal = tmp_path / "tool-started"
        env = {**os.environ, "ORDIVYR_LITE_HOME": str(home), "PYTHONPATH": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1"}
        code = f"""
import time
from pathlib import Path
from ordivyr_lite.tools import ToolBox
old = ToolBox.call
def slow(self, name, args):
    out = old(self, name, args)
    if name == 'write_file':
        Path({str(signal)!r}).write_text('ready')
        time.sleep(60)
    return out
ToolBox.call = slow
from ordivyr_lite.cli import main
import sys
sys.exit(main(['run', 'write exactly once', '--workdir', {str(work)!r},
 '--provider', 'custom', '--base-url', {srv.base!r}, '--model', 'test-model',
 '--permission', 'workspace', '--validate', 'off', '--session', {str(session)!r}, '--json']))
"""
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        while not signal.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert signal.exists(), "child never reached the injected in-tool interruption point"
        proc.kill()
        proc.communicate(timeout=10)
        assert (work / "effect.txt").read_text() == "once"
        before = len(srv.requests)
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--permission", "workspace",
                      "--validate", "off", "--json",
                      env_extra={"ORDIVYR_LITE_RESUME": "strict"})
        assert resumed.returncode != 0
        result = json.loads(resumed.stdout)
        assert result["stop_reason"] == "unverified"
        assert "refusing to replay" in result["error"]
        assert len(srv.requests) == before
        assert (work / "effect.txt").read_text() == "once"
    finally:
        srv.stop()


def test_process_kill_after_durable_tool_result_reconciles_checkpoint(home, tmp_path):
    calls = {"n": 0}
    def responder(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return openai_reply(calls=[("write_file", {"path": "effect.txt", "content": "once"})])
        return openai_reply(text="finished")
    srv = ScriptedServer(responder).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        session = tmp_path / "tool-result-crash.jsonl"
        signal = tmp_path / "tool-result-durable"
        env = {**os.environ, "ORDIVYR_LITE_HOME": str(home), "PYTHONPATH": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1"}
        code = f"""
import time
from pathlib import Path
from ordivyr_lite.session import Session
old = Session.append
def slow(self, event):
    old(self, event)
    if event.get('type') == 'tool_completed':
        self.flush()
        Path({str(signal)!r}).write_text('ready')
        time.sleep(60)
Session.append = slow
from ordivyr_lite.cli import main
import sys
sys.exit(main(['run', 'write and continue', '--workdir', {str(work)!r},
 '--provider', 'custom', '--base-url', {srv.base!r}, '--model', 'test-model',
 '--permission', 'workspace', '--validate', 'off', '--session', {str(session)!r}, '--json']))
"""
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        while not signal.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert signal.exists(), "child never reached the durable result checkpoint"
        proc.kill()
        proc.communicate(timeout=10)
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--permission", "workspace",
                      "--validate", "off", "--json")
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout)
        assert result["status"] == "succeeded"
        assert result["usage"]["tool_calls"] == 1
        assert len(srv.requests) == 2
        assert sum(1 for message in srv.requests[-1]["messages"]
                   if message.get("role") == "tool") == 1
        assert (work / "effect.txt").read_text() == "once"
    finally:
        srv.stop()


def test_process_kill_during_validation_resumes_and_revalidates(home, tmp_path):
    calls = {"n": 0}
    def responder(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return openai_reply(calls=[("write_file", {"path": "application.py",
                                                           "content": "VALUE = 1\n"})])
        return openai_reply(text="task complete")
    srv = ScriptedServer(responder).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "test_wait.py").write_text(
            "from pathlib import Path\nimport time\n"
            "def test_wait_for_recovery():\n"
            "    Path('validation.active').write_text('running')\n"
            "    time.sleep(60)\n")
        session = tmp_path / "validation-crash.jsonl"
        env = {**os.environ, "ORDIVYR_LITE_HOME": str(home), "PYTHONPATH": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1",
               # V2 also validates after each edit; this test targets the final-answer path.
               "ORDIVYR_LITE_NO_HARNESS_CHECK": "1"}
        code = (
            "from ordivyr_lite.cli import main; import sys; "
            "sys.exit(main(['run','complete the task','--workdir'," + repr(str(work))
            + ",'--provider','custom','--base-url'," + repr(srv.base)
            + ",'--model','test-model','--permission','workspace','--session',"
            + repr(str(session)) + ",'--json']))")
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        active = work / "validation.active"
        while not active.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert active.exists(), "child never entered validation"
        proc.kill()
        proc.communicate(timeout=10)
        (work / "test_wait.py").write_text("def test_recovered_validation():\n    assert True\n")
        before = len(srv.requests)
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--permission", "workspace", "--json",
                      env_extra={"ORDIVYR_LITE_NO_HARNESS_CHECK": "1"})
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout)
        assert result["status"] == "succeeded"
        assert result["completion"]["tests_passed"] is True
        assert result["completion"]["independent_review"] is False
        assert len(srv.requests) == before
    finally:
        srv.stop()


def test_process_kill_during_harness_check_resumes_and_finishes(home, tmp_path):
    calls = {"n": 0}
    def responder(_body):
        calls["n"] += 1
        if calls["n"] == 1:
            return openai_reply(calls=[("write_file", {"path": "application.py",
                                                           "content": "VALUE = 1\n"})])
        return openai_reply(text="task complete")
    srv = ScriptedServer(responder).start()
    try:
        work = tmp_path / "work"
        work.mkdir()
        (work / "test_wait.py").write_text(
            "from pathlib import Path\nimport time\n"
            "def test_wait_for_recovery():\n"
            "    Path('validation.active').write_text('running')\n"
            "    time.sleep(60)\n")
        session = tmp_path / "validation-crash.jsonl"
        env = {**os.environ, "ORDIVYR_LITE_HOME": str(home), "PYTHONPATH": str(ROOT),
               "PYTHONDONTWRITEBYTECODE": "1"}
        code = (
            "from ordivyr_lite.cli import main; import sys; "
            "sys.exit(main(['run','complete the task','--workdir'," + repr(str(work))
            + ",'--provider','custom','--base-url'," + repr(srv.base)
            + ",'--model','test-model','--permission','workspace','--session',"
            + repr(str(session)) + ",'--json']))")
        proc = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        deadline = time.monotonic() + 15
        active = work / "validation.active"
        while not active.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert active.exists(), "child never entered validation"
        proc.kill()
        proc.communicate(timeout=10)
        (work / "test_wait.py").write_text("def test_recovered_validation():\n    assert True\n")
        before = len(srv.requests)
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base,
                      "--model", "test-model", "--permission", "workspace", "--json")
        assert resumed.returncode == 0, resumed.stderr
        result = json.loads(resumed.stdout)
        assert result["status"] == "succeeded"
        assert result["completion"]["tests_passed"] is True
        assert result["completion"]["independent_review"] is False
        # The edit's results were not yet delivered to the model: exactly one new request.
        assert len(srv.requests) == before + 1
    finally:
        srv.stop()


def test_trusted_omitted_shell_flag_registers_and_executes_command(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    observed = []
    step = {"n": 0}

    def respond(request):
        names = {t["function"]["name"] for t in request.get("tools", [])}
        observed.append(names)
        if step["n"] == 0:
            step["n"] += 1
            assert "run_command" in names
            return openai_reply(calls=[("run_command", {"command": "printf ran > marker.txt"})])
        return openai_reply(text="command executed")

    srv = ScriptedServer(respond).start()
    try:
        p = cli(home, "run", "run the required check", "--workdir", str(work),
                "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                "--permission", "trusted", "--sandbox", "process", "--validate", "off",
                "--no-session", "--json")
        assert p.returncode == 0, p.stderr
        assert (work / "marker.txt").read_text() == "ran"
        assert all("run_command" in names for names in observed)
        assert "shell=enabled" in p.stderr
    finally:
        srv.stop()


def test_explicit_shell_deny_and_tool_allow_list_are_enforced(home, tmp_path):
    for extra, expected in [(["--no-shell"], False), (["--tools", "read_file"], False)]:
        work = tmp_path / ("work" + str(len(extra)))
        work.mkdir()
        seen = []

        def respond(request):
            names = {t["function"]["name"] for t in request.get("tools", [])}
            seen.append(names)
            if "run_command" not in names:
                return openai_reply(text="I cannot run the command under the current permissions.")
            return openai_reply(calls=[("run_command", {"command": "touch should-not-exist.txt"})])

        srv = ScriptedServer(respond).start()
        try:
            p = cli(home, "run", "run a command", "--workdir", str(work),
                    "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                    "--permission", "trusted", "--sandbox", "process", "--validate", "off",
                    "--no-session", "--json", *extra)
            assert p.returncode == 0, p.stderr
            assert seen and all(("run_command" in names) is expected for names in seen)
            assert not (work / "should-not-exist.txt").exists()
        finally:
            srv.stop()


def test_conflicting_shell_flags_are_actionable_cli_error(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "x", "--workdir", str(work), "--allow-shell", "--no-shell")
    assert p.returncode == 2
    assert "not allowed with argument" in p.stderr


def test_read_only_profile_shell_grant_is_actionable_error(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "x", "--workdir", str(work), "--provider", "custom",
            "--base-url", "http://127.0.0.1:9/v1", "--model", "test-model",
            "--permission", "read-only", "--allow-shell")
    assert p.returncode == 2
    assert "read-only profile cannot enable shell" in p.stderr


def test_dry_run_shows_effective_permissions_and_denial_reason(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "x", "--workdir", str(work), "--provider", "custom",
            "--base-url", "http://127.0.0.1:9/v1", "--model", "test-model",
            "--permission", "trusted", "--no-shell", "--dry-run")
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert "run_command" not in plan["tools"]
    assert "explicitly disabled" in plan["tool_permissions"]["unavailable"]["run_command"]


def test_explicit_empty_tool_list_stays_empty_in_cli(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    p = cli(home, "run", "x", "--workdir", str(work), "--provider", "custom",
            "--base-url", "http://127.0.0.1:9/v1", "--model", "test-model",
            "--permission", "trusted", "--tools", "", "--dry-run")
    assert p.returncode == 0, p.stderr
    plan = json.loads(p.stdout)
    assert plan["tools"] == []


def test_config_shell_grant_is_not_overridden_by_omitted_cli_flag(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    home.mkdir()
    (home / "config.json").write_text(json.dumps({"permissions": {
        "profile": "workspace", "allow_shell": True, "sandbox": "process"}}))
    state = {"called": False}

    def respond(request):
        has_shell = any(t["function"]["name"] == "run_command"
                        for t in request.get("tools", []))
        if has_shell and not state["called"]:
            state["called"] = True
            return openai_reply(calls=[("run_command", {"command": "printf config > config-grant.txt"})])
        return openai_reply(text="done" if has_shell else "shell unavailable")

    srv = ScriptedServer(respond).start()
    try:
        p = cli(home, "run", "create a marker", "--workdir", str(work), "--provider", "custom",
                "--base-url", srv.base, "--model", "test-model", "--no-session", "--json")
        assert p.returncode == 0, p.stderr
        assert (work / "config-grant.txt").read_text() == "config"
    finally:
        srv.stop()


def test_resume_inherits_omitted_policy_fields_when_one_field_is_overridden(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    session = tmp_path / "resume.jsonl"
    seen = []
    step = {"n": 0}

    def respond(request):
        names = {t["function"]["name"] for t in request.get("tools", [])}
        seen.append(names)
        if step["n"] == 0:
            step["n"] += 1
            return openai_reply(calls=[("run_command", {"command": "echo resumed-ok"})])
        return openai_reply(text="finished")

    srv = ScriptedServer(respond).start()
    try:
        first = cli(home, "run", "perform a check", "--workdir", str(work),
                    "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                    "--permission", "trusted", "--sandbox", "process", "--max-turns", "1",
                    "--validate", "off", "--session", str(session), "--json")
        assert first.returncode == 6, first.stderr
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                      "--validate", "off", "--network", "--json")
        assert resumed.returncode == 0, resumed.stderr
        assert len(seen) >= 2 and all("run_command" in names for names in seen)
        assert "network=enabled" in resumed.stderr and "shell=enabled" in resumed.stderr
    finally:
        srv.stop()


def test_resume_explicit_narrower_profile_cannot_reuse_saved_shell(home, tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (work / "a.txt").write_text("input\n")
    session = tmp_path / "narrow.jsonl"
    seen = []
    step = {"n": 0}

    def respond(request):
        names = {t["function"]["name"] for t in request.get("tools", [])}
        seen.append((names, request.get("messages", [])))
        if step["n"] == 0:
            step["n"] += 1
            return openai_reply(calls=[("read_file", {"path": "a.txt"})])
        if step["n"] == 1:
            step["n"] += 1
            return openai_reply(calls=[("run_command", {"command": "touch must-not-exist"})])
        return openai_reply(text="Continuing without shell access.")

    srv = ScriptedServer(respond).start()
    try:
        first = cli(home, "run", "inspect the input", "--workdir", str(work),
                    "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                    "--permission", "trusted", "--sandbox", "process", "--max-turns", "1",
                    "--validate", "off", "--session", str(session), "--json")
        assert first.returncode == 6, first.stderr
        resumed = cli(home, "resume", str(session), "--workdir", str(work),
                      "--provider", "custom", "--base-url", srv.base, "--model", "test-model",
                      "--permission", "workspace", "--validate", "off", "--json")
        assert resumed.returncode == 0, resumed.stderr
        assert seen[0][0] and "run_command" in seen[0][0]
        assert seen[1][0] and "run_command" not in seen[1][0]
        assert any("Session resumed" in str(m.get("content", ""))
                   for m in seen[1][1] if m.get("role") == "user")
        assert not (work / "must-not-exist").exists()
    finally:
        srv.stop()


def test_fallback_uses_its_own_preset_base(home, tmp_path):
    from types import SimpleNamespace
    from ordivyr_lite.cli import _build_router
    from ordivyr_lite.config import load_config

    args = SimpleNamespace(provider="custom", model="primary", base_url="http://127.0.0.1:9999/v1",
                           protocol=None, api_key_env=None, api_key_file=None,
                           context_window=None, timeout=None, allow_paid=False,
                           max_cost=0.0, fallback=["llamacpp:local-model"],
                           allow_paid_fallback=False)
    router = _build_router(load_config(), args, None)
    assert router.routes[0].base_url == "http://127.0.0.1:9999/v1"
    assert router.routes[1].base_url == "http://127.0.0.1:8080/v1"
    assert router.routes[1].model == "local-model"


def test_fast_version_path(home):
    p = cli(home, "--version")
    assert p.returncode == 0
    assert "ordivyr-lite 0.2.0" in p.stdout
