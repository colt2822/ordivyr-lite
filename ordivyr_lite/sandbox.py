"""Sandboxed command execution.

Default is bubblewrap (bwrap): a new mount/pid/net namespace, the working folder
mounted at /workspace, the toolchain mounted read-only, a tmpfs HOME, no network
unless explicitly enabled, and capabilities dropped.  If bwrap is unavailable,
`run_command` is disabled unless the operator explicitly chooses the
non-isolated ``process`` mode (clearly reported in the result).

``run`` pays one namespace setup per command.  ``SandboxSession`` starts that
same bwrap argv once and frames every command with a per-command nonce plus
byte counts, so a command costs its own execution instead of a cold start.
Any protocol anomaly falls back to ``run`` for the rest of the session.
"""

from __future__ import annotations

import atexit
import base64
import os
try:
    import resource
except ImportError:  # Windows: rlimits and bubblewrap are POSIX-only
    resource = None  # type: ignore[assignment]
import secrets
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .errors import SandboxError

MAX_OUTPUT = 200_000
DEFAULT_TIMEOUT = 60
MAX_TIMEOUT = 300


@dataclass
class SandboxResult:
    exit_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    sandbox: str
    isolated: bool
    duration_ms: float
    network: bool = False
    truncated: bool = False
    readonly_workspace: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def text(self) -> str:
        parts = [f"exit={self.exit_code} timed_out={self.timed_out} sandbox={self.sandbox}"]
        if self.readonly_workspace:
            parts.append('workspace=read-only; changes require file tools')
        if self.stdout:
            parts.append("stdout:\n" + self.stdout)
        if self.stderr:
            parts.append("stderr:\n" + self.stderr)
        return "\n".join(parts)


def bwrap_path() -> str | None:
    return shutil.which("bwrap")


def _limits(cpu_seconds: int | None, memory_mb: int):
    def apply() -> None:
        if resource is None:
            return
        if cpu_seconds is not None:
            # A session counts CPU across many commands, so it leaves RLIMIT_CPU
            # alone here and applies the per-command value in its driver.
            try:
                resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 2))
            except (ValueError, OSError):
                pass
        try:
            mem = memory_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024, 100 * 1024 * 1024))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (512, 512))
        except (ValueError, OSError):
            pass

    return apply


def _bwrap_argv(command: str, workdir: Path, network: bool, readonly_workspace: bool = False) -> list[str]:
    argv = [
        bwrap_path() or "bwrap",
        "--die-with-parent", "--new-session", "--unshare-all",
        "--cap-drop", "ALL",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ]
    # Toolchain and system configuration are read-only.
    for path in ("/usr", "/usr/local", "/bin", "/sbin", "/lib", "/lib64", "/lib32",
                 "/libx32", "/etc", "/opt", "/run/systemd/resolve", "/run/nscd"):
        if readonly_workspace and path.startswith('/run/'):
            continue  # do not expose host resolver/cache IPC to read-only diagnostics
        if os.path.exists(path):
            argv += ["--ro-bind", path, path]
    # Installed virtual environments and explicitly configured Go toolchains
    # are trusted execution inputs, not the surrounding home/config directory.
    roots = [sys.prefix]
    go_binary = os.environ.get('ORDIVYR_LITE_GO_BINARY')
    if go_binary:
        roots.append(str(Path(go_binary).resolve().parent.parent))
    for root in dict.fromkeys(os.path.realpath(p) for p in roots):
        if root not in ('/', '/home', str(Path.home())) and not root.startswith(('/usr', '/bin', '/lib', '/etc', '/opt')):
            if os.path.isdir(root):
                argv += ['--ro-bind', root, root]
    # Python packages installed in the user's site-packages (common on hosts
    # where pytest/pip --user is used) are needed for test validation.  Bind
    # only those directories read-only and add them to PYTHONPATH; never bind
    # the whole home directory.
    py_paths: list[str] = []
    try:
        import site
        candidates = list(site.getsitepackages()) + [site.getusersitepackages()]
        for path in candidates:
            if not path or not os.path.isdir(path):
                continue
            real = os.path.realpath(path)
            if real.startswith(("/usr", "/bin", "/lib", "/etc", "/opt")):
                continue
            if real not in py_paths:
                py_paths.append(real)
                argv += ["--ro-bind", real, real]
    except Exception:  # noqa: BLE001 - site discovery is best effort
        pass
    # Honor the interpreter environment that launched Lite. Using only system
    # PATH loses installed test dependencies whenever a src-layout command
    # legitimately replaces PYTHONPATH. The active venv is already mounted RO.
    command_path = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    if sys.prefix != sys.base_prefix and os.path.isdir(os.path.join(sys.prefix, 'bin')):
        command_path = os.path.join(sys.prefix, 'bin') + ':' + command_path
    argv += ["--ro-bind" if readonly_workspace else "--bind", str(workdir), "/workspace", "--chdir", "/workspace",
             "--clearenv",
             "--setenv", "PATH", command_path,
             "--setenv", "HOME", "/tmp",
             "--setenv", "PWD", "/workspace",
             "--setenv", "LANG", "C.UTF-8",
             "--setenv", "LC_ALL", "C.UTF-8",
             "--setenv", "TERM", "dumb",
             "--setenv", "PYTHONNOUSERSITE", "1",
             "--setenv", "PYTHONDONTWRITEBYTECODE", "1",
             "--setenv", "ORDIVYR_LITE_SANDBOX", "bwrap"]
    if py_paths:
        argv += ["--setenv", "PYTHONPATH", ":".join(py_paths)]
    if network:
        argv.append("--share-net")       # --unshare-all unshared it; opt back in explicitly
    else:
        argv.append("--unshare-net")
    argv += ["--", "/bin/sh", "-c", command]
    return argv


def _run_popen(argv: list[str], *, cwd: Path | None, env: dict, timeout: int,
               cpu_seconds: int, memory_mb: int) -> tuple[int | None, bool, str, str, float]:
    t0 = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
            start_new_session=True, preexec_fn=_limits(cpu_seconds, memory_mb))
    except (OSError, ValueError) as exc:
        raise SandboxError(f"cannot start sandboxed command: {exc}") from exc
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            proc.kill()
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
    return proc.returncode, timed_out, out or "", err or "", round((time.monotonic() - t0) * 1000, 1)


def _clip(text: str) -> tuple[str, bool]:
    if len(text) <= MAX_OUTPUT:
        return text, False
    head = int(MAX_OUTPUT * 0.7)
    tail = MAX_OUTPUT - head
    return text[:head] + "\n[... output truncated ...]\n" + text[-tail:], True


def _prepare(command: str, workdir: str | os.PathLike,
             timeout: int) -> tuple[str, int, Path]:
    command = str(command)
    if not command.strip():
        raise SandboxError("empty command")
    if len(command) > 8000:
        raise SandboxError("command is too long (max 8000 chars)")
    timeout = max(1, min(int(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
    cwd = Path(workdir).resolve()
    if not cwd.is_dir():
        raise SandboxError(f"working directory does not exist: {cwd}")
    return command, timeout, cwd


def run(command: str, workdir: str | os.PathLike, *, timeout: int = DEFAULT_TIMEOUT,
        network: bool = False, mode: str = "auto", memory_mb: int = 4096,
        readonly_workspace: bool = False) -> SandboxResult:
    """Run one shell command.  Raises SandboxError when isolation is unavailable.

    mode: ``auto`` (bwrap if present, else refuse), ``bwrap`` (require it),
    ``process`` (explicitly non-isolated, for environments without user
    namespaces), ``none`` (unsafe; caller must opt in).
    """
    command, timeout, cwd = _prepare(command, workdir, timeout)
    mode = (mode or "auto").lower()
    has_bwrap = bwrap_path() is not None
    if mode == "auto":
        mode = "bwrap" if has_bwrap else "unavailable"
    if readonly_workspace and (mode != 'bwrap' or network):
        raise SandboxError('read-only workspace commands require bubblewrap and disabled network; no process fallback is permitted')
    if mode == "bwrap":
        if not has_bwrap:
            raise SandboxError("bubblewrap (bwrap) is not installed; run_command is disabled. "
                               "Install bubblewrap, or opt into the non-isolated process sandbox "
                               "with --sandbox process.")
        argv = (_bwrap_argv(command, cwd, network, readonly_workspace=True) if readonly_workspace else
                _bwrap_argv(command, cwd, network))
        code, timed_out, out, err, dur = _run_popen(
            argv, cwd=None, env={"PATH": "/usr/bin:/bin"}, timeout=timeout,
            cpu_seconds=timeout, memory_mb=memory_mb)
        out, trunc1 = _clip(out)
        err, trunc2 = _clip(err)
        return SandboxResult(code, timed_out, out, err, "bwrap", True, dur, network,
                             trunc1 or trunc2, readonly_workspace)
    if mode == "process":
        home = tempfile.mkdtemp(prefix="ordivyr-lite-home-")
        env = {"PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
               "HOME": home, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "dumb",
               "PYTHONDONTWRITEBYTECODE": "1", "ORDIVYR_LITE_SANDBOX": "process"}
        try:
            code, timed_out, out, err, dur = _run_popen(
                ["/bin/sh", "-c", command], cwd=cwd, env=env, timeout=timeout,
                cpu_seconds=timeout, memory_mb=memory_mb)
        finally:
            shutil.rmtree(home, ignore_errors=True)
        out, trunc1 = _clip(out)
        err, trunc2 = _clip(err)
        return SandboxResult(code, timed_out, out, err, "process", False, dur, network,
                             trunc1 or trunc2)
    if mode == "none":
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(cwd),
               "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "dumb",
               "ORDIVYR_LITE_SANDBOX": "none"}
        code, timed_out, out, err, dur = _run_popen(
            ["/bin/sh", "-c", command], cwd=cwd, env=env, timeout=timeout,
            cpu_seconds=timeout, memory_mb=memory_mb)
        out, trunc1 = _clip(out)
        err, trunc2 = _clip(err)
        return SandboxResult(code, timed_out, out, err, "none", False, dur, network,
                             trunc1 or trunc2)
    raise SandboxError(
        "no sandbox is available: install bubblewrap, or opt into --sandbox process "
        "(non-isolated) explicitly")

_SESSION_DRIVER = r"""set -u
command -v base64 >/dev/null 2>&1 || exit 71
command -v wc >/dev/null 2>&1 || exit 71
command -v head >/dev/null 2>&1 || exit 71
while IFS=' ' read -r nonce cpu mem payload; do
    [ -n "$nonce" ] || continue
    [ "$nonce" = quit ] && exit 0
    out="/tmp/.ordivyr-$nonce.out"
    err="/tmp/.ordivyr-$nonce.err"
    cmd=$(printf '%s' "$payload" | base64 -d) || exit 70
    ( ulimit -t "$cpu" 2>/dev/null; ulimit -v "$mem" 2>/dev/null; exec /bin/sh -c "$cmd" ) </dev/null >"$out" 2>"$err"
    rc=$?
    olen=$(wc -c < "$out")
    elen=$(wc -c < "$err")
    [ -n "$olen" ] && [ -n "$elen" ] || exit 72
    printf '%s %s %s %s\n' "$nonce" "$rc" "$olen" "$elen"
    head -c "$olen" "$out"
    head -c "$elen" "$err"
    rm -f "$out" "$err"
done
exit 0
"""

_CHUNK = 65536
_MAX_FRAME_LINE = 8 * _CHUNK
_READ_CAP = MAX_OUTPUT * 4
_FRAME_GRACE = 30.0
_SALVAGE_CAP = 20_000


class _SessionTimeout(Exception):
    """The command did not report back before its deadline."""


class _SessionBroken(Exception):
    """The session protocol could not complete a frame."""


class SandboxSession:
    """One bwrap process reused by many commands.

    The argv, the confinement flags, the environment and the rlimits are the
    ones ``run`` uses; each command is still a fresh ``/bin/sh -c`` inside the
    sandboxed namespace with /workspace as its directory.  Every frame carries
    the exit status behind a per-command nonce that the command never sees, and
    payloads are byte counted, so no command output can spoof or interleave a
    result.  Any protocol anomaly degrades the session to ``run`` for good.
    """

    def __init__(self, workdir: str | os.PathLike, *, network: bool = False,
                 mode: str = "auto", memory_mb: int = 4096) -> None:
        mode = (mode or "auto").lower()
        has_bwrap = bwrap_path() is not None
        if mode == "auto":
            mode = "bwrap" if has_bwrap else "unavailable"
        self.workdir = Path(workdir).resolve()
        self.network = bool(network)
        self.mode = mode
        self.memory_mb = int(memory_mb)
        self.fallback = False
        self._persistent = mode == "bwrap" and has_bwrap
        self._proc: subprocess.Popen | None = None
        self._buffer = b""
        self._lock = threading.Lock()
        self._closed = False
        _claim(self)

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    def __enter__(self) -> "SandboxSession":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._discard()
            _release(self)

    def run(self, command: str, timeout: int = DEFAULT_TIMEOUT) -> SandboxResult:
        command, timeout, _ = _prepare(command, self.workdir, timeout)
        with self._lock:
            if self._closed:
                raise SandboxError("sandbox session is closed")
            if not self._persistent or self.fallback:
                return self._one_shot(command, timeout)
            try:
                self._ensure_started()
            except SandboxError:
                self._discard()
                return self._one_shot(command, timeout)
            return self._run_framed(command, timeout)

    # ---------------------------------------------------------------- protocol
    def _run_framed(self, command: str, timeout: int) -> SandboxResult:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return self._one_shot(command, timeout)
        t0 = time.monotonic()
        nonce = secrets.token_hex(12)
        payload = base64.b64encode(command.encode("utf-8", "replace")).decode("ascii")
        line = f"{nonce} {timeout} {self.memory_mb * 1024} {payload}\n".encode("ascii")
        try:
            proc.stdin.write(line)
            proc.stdin.flush()
        except (OSError, ValueError):
            self._discard()
            return self._one_shot(command, timeout)
        try:
            code, out, err = self._read_frame(nonce, t0 + timeout)
        except _SessionTimeout:
            duration = round((time.monotonic() - t0) * 1000, 1)
            salvaged, truncated = self._salvage(nonce)   # before _kill wipes the buffer
            return SandboxResult(self._kill(), True, salvaged, "", "bwrap", True, duration,
                                 self.network, truncated)
        except (_SessionBroken, OSError, ValueError):
            self._discard()
            return self._one_shot(command, timeout)
        duration = round((time.monotonic() - t0) * 1000, 1)
        out, trunc1 = _clip(out)
        err, trunc2 = _clip(err)
        return SandboxResult(code, False, out, err, "bwrap", True, duration, self.network,
                             trunc1 or trunc2)

    def _salvage(self, nonce: str) -> tuple[str, bool]:
        """Text the driver already streamed for the frame that timed out.

        The frame header is dropped when it is the one for this nonce; control
        bytes are removed and the result is capped like any other output.  A
        timed-out command is a failed command, but whatever it printed before
        the deadline is still the user's output.
        """
        raw = self._buffer
        if not raw:
            return "", False
        text = raw.decode("utf-8", "replace")
        head, sep, rest = text.partition("\n")
        if sep and head.split()[:1] == [nonce]:
            text = rest
        text = "".join(ch for ch in text if ch in "\n\t" or ch.isprintable()).strip("\n")
        if not text:
            return "", False
        if len(text) > _SALVAGE_CAP:
            return text[: _SALVAGE_CAP - 40] + "\n[... output truncated ...]", True
        return text, False

    def _read_frame(self, nonce: str, deadline: float) -> tuple[int, str, str]:
        while True:
            parts = self._read_line(deadline).split()
            if len(parts) != 4 or parts[0] != nonce:
                continue
            try:
                code, out_len, err_len = int(parts[1]), int(parts[2]), int(parts[3])
            except ValueError:
                continue
            if out_len < 0 or err_len < 0:
                continue
            out = self._payload(out_len, deadline + _FRAME_GRACE)
            err = self._payload(err_len, deadline + _FRAME_GRACE)
            return code, out.decode("utf-8", "replace"), err.decode("utf-8", "replace")

    def _payload(self, count: int, deadline: float) -> bytes:
        kept = bytearray()
        remaining = count
        while remaining:
            chunk = self._take(min(remaining, _CHUNK), deadline)
            remaining -= len(chunk)
            if len(kept) < _READ_CAP:
                kept += chunk[:_READ_CAP - len(kept)]
        return bytes(kept)

    def _read_line(self, deadline: float) -> str:
        while b"\n" not in self._buffer:
            self._fill(deadline)
        raw, self._buffer = self._buffer.split(b"\n", 1)
        return raw.decode("ascii", "replace")

    def _take(self, count: int, deadline: float) -> bytes:
        while not self._buffer:
            self._fill(deadline)
        chunk, self._buffer = self._buffer[:count], self._buffer[count:]
        return chunk

    def _fill(self, deadline: float) -> None:
        if len(self._buffer) > _MAX_FRAME_LINE:
            raise _SessionBroken("unterminated frame")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _SessionTimeout("deadline")
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise _SessionBroken("no session process")
        try:
            ready, _, _ = select.select([proc.stdout], [], [], remaining)
        except (OSError, ValueError) as exc:
            raise _SessionBroken(str(exc)) from exc
        if not ready:
            raise _SessionTimeout("deadline")
        chunk = os.read(proc.stdout.fileno(), _CHUNK)
        if not chunk:
            raise _SessionBroken("session exited")
        self._buffer += chunk

    # ---------------------------------------------------------------- process
    def _ensure_started(self) -> None:
        if self.alive:
            return
        self._discard()
        argv = _bwrap_argv(_SESSION_DRIVER, self.workdir, self.network)
        try:
            self._proc = subprocess.Popen(
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, env={"PATH": "/usr/bin:/bin"},
                start_new_session=True, bufsize=0,
                preexec_fn=_limits(None, self.memory_mb))
        except (OSError, ValueError) as exc:
            raise SandboxError(f"cannot start sandbox session: {exc}") from exc
        self._buffer = b""

    def _one_shot(self, command: str, timeout: int) -> SandboxResult:
        self.fallback = True
        return run(command, self.workdir, timeout=timeout, network=self.network,
                   mode=self.mode, memory_mb=self.memory_mb)

    def _kill(self) -> int | None:
        proc = self._proc
        if proc is None:
            self._buffer = b""
            return None
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self._close_pipes()
        self._proc = None
        self._buffer = b""
        return proc.returncode

    def _discard(self) -> None:
        proc = self._proc
        self._buffer = b""
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(b"quit\n")
                proc.stdin.flush()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._kill()
            return
        self._close_pipes()
        self._proc = None

    def _close_pipes(self) -> None:
        proc = self._proc
        if proc is None:
            return
        for pipe in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass


_ACTIVE: SandboxSession | None = None
_ACTIVE_LOCK = threading.Lock()
_REAPER = False


def _claim(new: SandboxSession) -> None:
    global _ACTIVE, _REAPER
    with _ACTIVE_LOCK:
        current = _ACTIVE
        if current is not None and current is not new and not current._closed:
            raise SandboxError("a sandbox session is already open in this process; "
                               "close it before starting another")
        _ACTIVE = new
        if not _REAPER:
            atexit.register(_reap_active)
            _REAPER = True


def _release(old: SandboxSession) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        if _ACTIVE is old:
            _ACTIVE = None


def _reap_active() -> None:
    with _ACTIVE_LOCK:
        current = _ACTIVE
    if current is not None:
        try:
            current.close()
        except Exception:  # noqa: BLE001 - the exit path must not raise
            pass


def session(workdir: str | os.PathLike, *, network: bool = False, mode: str = "auto",
            memory_mb: int = 4096) -> SandboxSession:
    """Open the process-wide sandbox session (bwrap when available)."""
    return SandboxSession(workdir, network=network, mode=mode, memory_mb=memory_mb)

