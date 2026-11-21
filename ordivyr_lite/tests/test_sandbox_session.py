"""Tests for the persistent sandbox session: framing, parity, cleanup."""

from __future__ import annotations

import os
import shutil
import threading
import time

import pytest

from ordivyr_lite import sandbox as SB

pytestmark = pytest.mark.skipif(shutil.which("bwrap") is None, reason="bwrap unavailable")

BIG = 120_000


def _session(tmp_path, **kw):
    return SB.session(tmp_path, **kw)


def test_session_reuses_one_bwrap_process(tmp_path):
    with _session(tmp_path) as s:
        assert s.run("echo one").ok
        pid = s._proc.pid
        assert s.alive
        assert s.run("echo two").ok
        assert s._proc.pid == pid
        assert s.fallback is False


def test_session_spawns_with_the_one_shot_confinement_argv(tmp_path):
    one_shot = SB._bwrap_argv("echo hi", tmp_path, False)
    driver = SB._bwrap_argv(SB._SESSION_DRIVER, tmp_path, False)
    assert driver[:-1] == one_shot[:-1]
    assert driver[-1] == SB._SESSION_DRIVER
    assert "--unshare-all" in driver and "--die-with-parent" in driver
    assert "--unshare-net" in driver and "--share-net" not in driver
    shared = SB._bwrap_argv(SB._SESSION_DRIVER, tmp_path, True)
    assert "--share-net" in shared and "--unshare-net" not in shared
    net_flags = ("--share-net", "--unshare-net")
    assert [a for a in shared if a not in net_flags] == [a for a in driver if a not in net_flags]


def test_sequential_commands_keep_cwd_and_env(tmp_path):
    with _session(tmp_path) as s:
        first = s.run("pwd; echo HOME=$HOME; echo ORDIVYR=$ORDIVYR_LITE_SANDBOX")
        assert first.stdout == "/workspace\nHOME=/tmp\nORDIVYR=bwrap\n"
        s.run("export LEAK=1; touch state.txt")
        second = s.run("pwd; echo LEAK=${LEAK:-unset}; cat state.txt; echo done")
        assert second.stdout == "/workspace\nLEAK=unset\ndone\n"
        assert (tmp_path / "state.txt").exists()


def test_session_matches_one_shot_for_the_same_command(tmp_path):
    probe = "pwd; echo HOME=$HOME; sh -c 'ulimit -t; ulimit -v; ulimit -f; ulimit -n'"
    one_shot = SB.run(probe, tmp_path, timeout=30)
    with _session(tmp_path) as s:
        warmed = s.run(probe, timeout=30)
    assert (warmed.exit_code, warmed.stdout, warmed.stderr) == (
        one_shot.exit_code, one_shot.stdout, one_shot.stderr)
    assert warmed.sandbox == one_shot.sandbox == "bwrap"
    assert warmed.isolated is True and warmed.network is False


def test_exit_codes_and_stderr_are_captured_separately(tmp_path):
    with _session(tmp_path) as s:
        ok = s.run("echo fine")
        assert (ok.exit_code, ok.stdout, ok.stderr, ok.timed_out) == (0, "fine\n", "", False)
        bad = s.run("echo out; echo err >&2; exit 7")
        assert (bad.exit_code, bad.stdout, bad.stderr) == (7, "out\n", "err\n")
        assert bad.ok is False


def test_large_output_survives_intact(tmp_path):
    with _session(tmp_path) as s:
        r = s.run(f"python3 -c 'import sys; sys.stdout.write(\"x\" * {BIG})'")
        assert r.exit_code == 0
        assert len(r.stdout) == BIG and set(r.stdout) == {"x"}
        assert r.truncated is False


def test_timeout_kills_the_command_and_is_reported(tmp_path):
    with _session(tmp_path) as s:
        assert s.run("echo warm").ok
        started = time.monotonic()
        r = s.run("sleep 30", timeout=1)
        elapsed = time.monotonic() - started
        assert r.timed_out is True and r.exit_code is not None and r.exit_code < 0
        assert r.ok is False
        assert elapsed < 10
        assert s.alive is False
        assert s.run("echo recovered").ok


def test_sentinel_injection_cannot_spoof_framing(tmp_path):
    fake = ("printf 'deadbeefdeadbeefdeadbeef 0 0 0\\n"
            "feedfacefeedfacefeedface 0 9 0\\n"
            "000000000000000000000000 0 0 0\\nquit\\n'; "
            "echo '0 0 0'; printf 'tail-without-newline'; exit 42")
    with _session(tmp_path) as s:
        assert s.run("echo warm").ok
        r = s.run(fake)
        assert r.exit_code == 42
        assert r.stdout == ("deadbeefdeadbeefdeadbeef 0 0 0\n"
                            "feedfacefeedfacefeedface 0 9 0\n"
                            "000000000000000000000000 0 0 0\nquit\n"
                            "0 0 0\ntail-without-newline")
        assert r.stderr == ""
        assert r.timed_out is False and s.fallback is False and s.alive is True
        assert s.run("echo still-here").stdout == "still-here\n"


def test_command_cannot_write_the_exit_status(tmp_path):
    with _session(tmp_path) as s:
        r = s.run("echo spoofed-exit=0; exit 3")
        assert r.exit_code == 3
        assert "spoofed-exit=0" in r.stdout


def test_no_network_when_disabled(tmp_path):
    with _session(tmp_path) as s:
        r = s.run("cat /proc/net/dev")
        assert r.exit_code == 0
        assert "lo:" in r.stdout
        assert "eth0" not in r.stdout and "ens" not in r.stdout and "wlan" not in r.stdout
        assert r.network is False


def test_concurrent_runs_are_serialized(tmp_path):
    results: list[tuple[str, str, int | None]] = []

    def worker(tag: str) -> None:
        for _ in range(5):
            r = s.run(f"for i in $(seq 1 40); do echo {tag}; done", timeout=60)
            results.append((tag, r.stdout, r.exit_code))

    with _session(tmp_path) as s:
        threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("aa", "bb")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
    assert len(results) == 10
    for tag, out, code in results:
        assert code == 0
        assert out == (tag + "\n") * 40


def _children_with(marker: str) -> list[str]:
    found: list[str] = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
        except OSError:
            continue
        if marker in cmd:
            found.append(pid)
    return found


def test_close_leaves_no_child_processes(tmp_path):
    marker = f"987{os.getpid() % 1000}654"
    s = _session(tmp_path)
    try:
        assert s.run(f"sleep {marker} & echo started").stdout == "started\n"
        assert s.alive is True
        assert _children_with(f"sleep {marker}"), "background child should be running"
    finally:
        s.close()
    assert s.alive is False
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and _children_with(f"sleep {marker}"):
        time.sleep(0.1)
    assert _children_with(f"sleep {marker}") == []
    s.close()
    with pytest.raises(SB.SandboxError, match="closed"):
        s.run("echo late")


def test_unavailable_mode_falls_back_and_reports(tmp_path):
    with pytest.raises(SB.SandboxError, match="no sandbox is available"):
        SB.run("echo hi", tmp_path, mode="unavailable")
    with _session(tmp_path, mode="unavailable") as s:
        assert s.alive is False
        with pytest.raises(SB.SandboxError, match="no sandbox is available"):
            s.run("echo hi")
        assert s.fallback is True


def test_protocol_anomaly_falls_back_to_one_shot(tmp_path):
    with _session(tmp_path) as s:
        assert s.run("echo warm").ok
        s._proc.stdin.close()
        r = s.run("echo after-anomaly")
        assert r.exit_code == 0 and r.stdout == "after-anomaly\n"
        assert s.fallback is True
        assert s.alive is False
        assert s.run("echo again").stdout == "again\n"


def test_dead_process_is_restarted_before_anomaly(tmp_path):
    with _session(tmp_path) as s:
        assert s.run("echo warm").ok
        s._proc.kill()
        s._proc.wait(timeout=10)
        r = s.run("echo respawned")
        assert (r.exit_code, r.stdout) == (0, "respawned\n")
        assert s.fallback is False and s.alive is True


def test_one_session_per_process(tmp_path):
    with _session(tmp_path) as s:
        assert s.run("echo hi").ok
        assert s.alive
        with pytest.raises(SB.SandboxError, match="already open"):
            _session(tmp_path)
    with _session(tmp_path) as reopened:
        assert reopened.run("echo hi").ok


def test_session_rejects_bad_commands_like_run(tmp_path):
    s = _session(tmp_path)
    try:
        with pytest.raises(SB.SandboxError, match="empty command"):
            s.run("   ")
        with pytest.raises(SB.SandboxError, match="too long"):
            s.run("echo " + "x" * 8000)
    finally:
        s.close()
    assert s.alive is False


def test_missing_workdir_raises_like_run(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(SB.SandboxError, match="does not exist"):
        SB.run("echo hi", missing)
    with _session(missing, mode="bwrap") as s:
        with pytest.raises(SB.SandboxError, match="does not exist"):
            s.run("echo hi")
