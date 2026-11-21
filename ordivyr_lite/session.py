"""Append-only, checked session journal with required explicit sync barriers.

New records have versioned length/digest/sequence framing. Readers preserve only
the valid prefix; automatic continuation refuses corruption. Legacy complete
JSONL prefixes remain readable but cannot gain retroactive integrity guarantees.
File-effect recovery follows the separate 021A content contract.
"""

from __future__ import annotations

import json
import errno
import hashlib
import os
import re
import time
import threading
from pathlib import Path

from . import PROTOCOL_VERSION, __version__
from .config import config_home
from .util import atomic_write_text, ensure_private_dir, new_run_id, utc_now

SESSION_FORMAT = "ordivyr-lite/session"
MAX_SNAPSHOT_BYTES = 32 * 1024 * 1024
FRAME = "ordivyr-lite/journal-record/1"
ZERO_DIGEST = "0" * 64


class JournalError(OSError):
    """Journal continuity or durability is uncertain; do not continue the run."""


def _encoded(value) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True,
                      default=str, allow_nan=False).encode("utf-8")


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate journal member")
        result[key] = value
    return result


class Session:
    def __init__(self, path: str | os.PathLike, run_id: str | None = None):
        self.path = Path(path).expanduser()
        ensure_private_dir(self.path.parent)
        self.run_id = run_id or new_run_id()
        self._fd: int | None = None
        self._lock = threading.RLock()
        self._failure: OSError | None = None
        self._dirty = False
        self._seq = 0
        self._previous = ZERO_DIGEST
        self.integrity_error: str | None = None

    @classmethod
    def create(cls, path: str | os.PathLike | None = None, run_id: str | None = None,
               home: str | os.PathLike | None = None) -> "Session":
        rid = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id or new_run_id()))[:120] or new_run_id()
        if path is None:
            root = ensure_private_dir(Path(home) if home else config_home() / "sessions")
            path = root / f"{rid}.jsonl"
        s = cls(path, rid)
        s.path.parent.mkdir(parents=True, exist_ok=True)
        s.path.touch(mode=0o600, exist_ok=True)
        try:
            os.chmod(s.path, 0o600)
        except OSError:
            pass
        try:
            s.append({"type": "header", "format": SESSION_FORMAT, "protocol_version": PROTOCOL_VERSION,
                      "lite_version": __version__, "run_id": rid, "created_at": utc_now(),
                      "cwd": os.getcwd()})
        except BaseException:
            if s._fd is not None:
                os.close(s._fd); s._fd = None
            raise
        return s

    @classmethod
    def open(cls, path: str | os.PathLike) -> "Session":
        p = Path(path).expanduser()
        if not p.exists():
            raise FileNotFoundError(f"session not found: {p}")
        session = cls(p)
        events = session.events()
        session.run_id = next((e["run_id"] for e in events if e.get("run_id")), session.run_id)
        return session

    # ------------------------------------------------------------------ writes
    def _open(self) -> int:
        self._check_failure()
        if self._fd is None:
            _events, self._seq, self._previous = self._scan(allow_missing=True)
            if self.integrity_error:
                raise JournalError(errno.EIO, "journal has an invalid suffix; preserve it for inspection")
            self._fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        return self._fd

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise JournalError(errno.EIO, "journal is unusable after an earlier storage failure") from self._failure

    def append(self, event: dict) -> None:
        with self._lock:
            self._check_failure()
            if "_journal" in event:
                raise ValueError("reserved journal framing member")
            event = {"ts": time.time(), **event}
            payload = _encoded(event)
            fd = self._open()
            frame = {"format": FRAME, "seq": self._seq + 1, "previous": self._previous,
                     "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
            data = _encoded({**event, "_journal": frame}) + b"\n"
            try:
                self._dirty = True
                written = os.write(fd, data)
                if type(written) is not int or written != len(data):
                    raise JournalError(errno.EIO, "incomplete journal append; no further writes permitted")
            except OSError as exc:
                self._failure = exc
                raise
            self._seq += 1
            self._previous = hashlib.sha256(data).hexdigest()

    def flush(self) -> None:
        """Durably commit appended events before an externally visible action."""
        with self._lock:
            self._check_failure()
            try:
                os.fsync(self._open())
            except OSError as exc:
                self._failure = exc
                raise
            self._dirty = False

    def snapshot(self, state: dict) -> None:
        payload = json.dumps(state, separators=(",", ":"), default=str)
        if len(payload) > MAX_SNAPSHOT_BYTES:
            state = {**state, "messages": state.get("messages", [])[-40:],
                     "_snapshot_truncated": True}
            payload = json.dumps(state, separators=(",", ":"), default=str)
        self.append({"type": "snapshot", "run_id": self.run_id, "state": state})
        self.flush()

    def close(self) -> None:
        with self._lock:
            if self._fd is None:
                self._check_failure()
                return
            failure = None
            try:
                self._check_failure()
                if self._dirty:
                    self.flush()
            except OSError as exc:
                failure = exc
            finally:
                try:
                    os.close(self._fd)
                except OSError as exc:
                    failure = failure or exc
                self._fd = None
            if failure is not None:
                self._failure = failure
                raise failure

    # ------------------------------------------------------------------ reads
    def _scan(self, *, allow_missing=False):
        out = []; seq = 0; previous = ZERO_DIGEST; framed = False
        self.integrity_error = None
        try:
            with open(self.path, "rb") as f:
                for line in f:
                    try:
                        if not line.endswith(b"\n"):
                            raise ValueError("torn journal line")
                        ev = json.loads(line, object_pairs_hook=_unique_pairs)
                        if not isinstance(ev, dict):
                            raise ValueError("non-object journal record")
                        meta = ev.pop("_journal", None)
                        if meta is None and framed:
                            raise ValueError("missing journal frame")
                        if meta is not None:
                            if not isinstance(meta, dict) or set(meta) != {"format", "seq", "previous", "bytes", "sha256"}:
                                raise ValueError("invalid journal frame")
                            payload = _encoded(ev)
                            if (meta["format"] != FRAME or type(meta["seq"]) is not int or meta["seq"] != seq + 1
                                    or meta["previous"] != previous or type(meta["bytes"]) is not int
                                    or meta["bytes"] != len(payload) or meta["sha256"] != hashlib.sha256(payload).hexdigest()):
                                raise ValueError("journal continuity check failed")
                            seq += 1; framed = True
                        out.append(ev)
                        previous = hashlib.sha256(line).hexdigest()
                    except (ValueError, TypeError, UnicodeError, RecursionError):
                        self.integrity_error = "invalid_or_incomplete_journal_suffix"
                        break
        except FileNotFoundError:
            if not allow_missing: raise
        return out, seq, previous

    def events(self, limit: int | None = None) -> list[dict]:
        with self._lock:
            out, _seq, _previous = self._scan()
        return out[-limit:] if limit else out

    def require_healthy(self) -> None:
        """Inspection may expose a valid prefix; automatic continuation may not
        ignore a broken suffix or a prior failure on this Session object.
        """
        with self._lock:
            self._check_failure()
            self._scan()
            if self.integrity_error:
                raise JournalError(errno.EIO, "journal continuity is uncertain; automatic resume refused")

    def latest_snapshot(self) -> dict | None:
        latest = None
        for ev in self.events():
            if ev.get("type") == "snapshot" and isinstance(ev.get("state"), dict):
                latest = ev["state"]
        return latest

    def metadata(self) -> dict:
        """Return the latest durable session metadata, including effective permissions."""
        metadata: dict = {}
        for ev in self.events():
            if ev.get("type") == "metadata" and isinstance(ev.get("metadata"), dict):
                metadata.update(ev["metadata"])
            state = ev.get("state")
            if ev.get("type") == "snapshot" and isinstance(state, dict):
                if isinstance(state.get("effective_permissions"), dict):
                    metadata["permissions"] = state["effective_permissions"]
        return metadata

    def set_metadata(self, **values) -> None:
        self.append({"type": "metadata", "metadata": values})

    def info(self) -> dict:
        events = self.events()
        snaps = [e for e in events if e.get("type") == "snapshot"]
        header = next((e for e in events if e.get("type") == "header"), {})
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "created_at": header.get("created_at"),
            "lite_version": header.get("lite_version"),
            "events": len(events),
            "snapshots": len(snaps),
            "last_snapshot_at": snaps[-1].get("ts") if snaps else None,
            "bytes": self.path.stat().st_size if self.path.exists() else 0,
        }

    def write_sidecar(self, name: str, text: str) -> Path:
        p = self.path.with_name(f"{self.path.stem}.{name}")
        atomic_write_text(p, text, mode=0o600)
        return p


def list_sessions(home: str | os.PathLike | None = None) -> list[dict]:
    root = Path(home) if home else config_home() / "sessions"
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*.jsonl"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            s = Session.open(p)
            out.append(s.info())
        except OSError:
            continue
    return out


def prune_sessions(keep: int = 20, home: str | os.PathLike | None = None) -> list[Path]:
    sessions = list_sessions(home)
    removed = []
    for info in sessions[max(0, keep):]:
        p = Path(info["path"])
        try:
            p.unlink()
            removed.append(p)
        except OSError:
            pass
    return removed
