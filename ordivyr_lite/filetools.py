"""File tools: paginated line-numbered read, surgical edit over a unique-match ladder, bounded
repository search, head+tail retention with task-local spill.

Every function receives the ToolBox (for its confined ``_path``, ``root`` and
spill dir).  Nothing follows a symlink out of the working folder and nothing
here writes without the box's write grant (checked by the caller).
"""

from __future__ import annotations

import difflib
import hashlib
import os
import re
import tempfile
import time
from pathlib import Path

READ_DEFAULT_LINES = 400
READ_MAX_LINES = 2000
READ_MAX_CHARS = 30_000
READ_MAX_FILE = 8_000_000
LINE_CAP = 2000
SEARCH_SKIP = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".tox", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build", ".next", "target", ".idea",
    ".gradle", ".ordivyr-spill", ".ordivyr-lite",
}
SEARCH_MAX_FILE = 1_000_000
GREP_MAX_MATCHES = 100
GREP_LINE_CAP = 240
FIND_MAX = 300
SEARCH_SECONDS = 10
# The echo is re-sent on every later turn of the run, so it stays small: enough
# to catch a mis-targeted tolerant match, cheap enough not to tax the context.
DIFF_MAX_LINES = 8
DIFF_MAX_CHARS = 600
DIFF_ROW_CHARS = 220
DIFF_MAX_PAIRS = 250_000
DIFF_MAX_REGION_LINES = 2000
RESULT_MAX_CHARS = 900
_AMBIG_CAP = 64
DEADLINE_NOTE = "deadline reached, results incomplete"


class FileToolError(ValueError):
    pass


def _text(p: Path, expected_sha256=None, on_read=None) -> tuple[str, str]:
    """-> (text, newline) with newlines untranslated; refuses binary/oversized."""
    if p.stat().st_size > READ_MAX_FILE:
        raise FileToolError(
            f"file is {p.stat().st_size} bytes; too large to read whole - use grep to find the part you need")
    b = p.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(b).hexdigest() != expected_sha256:
        raise FileToolError("stale file identity: expected_sha256 differs; read the file and retry the edit")
    if b"\0" in b[:8192]:
        raise FileToolError("binary file")
    if on_read:
        on_read(b)
    t = b.decode("utf-8", "replace")
    return t, ("\r\n" if t.count("\r\n") * 2 > t.count("\n") else "\n")


def resolve(box, rel) -> Path:
    """Read-side path: a workspace file, or ``spill:N`` (this box's saved output)."""
    rel = str(rel or "")
    if rel.startswith("spill:"):
        name = rel[6:]
        if not name.isdigit():
            raise FileToolError("bad spill reference")
        p = box.spill_dir() / name
        if not p.is_file():
            raise FileToolError("no such spill")
        return p
    p = box._path(rel)
    if not p.is_file():
        raise FileToolError("not a file")
    return p


def read_file(box, args) -> str:
    p = resolve(box, args.get("path"))
    def observed(data):
        if hasattr(box, '_observed_files'):
            box._observed_files[str(p)] = hashlib.sha256(data).hexdigest()
    t, _ = _text(p, on_read=observed)
    lines = t.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    total = len(lines)
    try:
        off = max(1, int(args.get("offset") or 1))
        lim = int(args.get("limit") or READ_DEFAULT_LINES)
    except (TypeError, ValueError):
        raise FileToolError("offset and limit must be integers") from None
    lim = max(1, min(lim, READ_MAX_LINES))
    if total == 0:
        return "(empty file)"
    if off > total:
        raise FileToolError(f"offset {off} is past the end ({total} lines)")
    out: list[str] = []
    used = 0
    end = off - 1
    for i in range(off - 1, min(total, off - 1 + lim)):
        ln = lines[i].rstrip("\r")
        cut = len(ln) > LINE_CAP
        row = f"{i + 1:>6}\t{ln[:LINE_CAP]}" + (f" [line cut at {LINE_CAP} of {len(ln)} chars]" if cut else "")
        if used + len(row) > READ_MAX_CHARS and out:
            break
        out.append(row)
        used += len(row) + 1
        end = i + 1
    body = "\n".join(out)
    if end < total:
        body += f"\n[showing lines {off}-{end} of {total}; call read_file again with offset={end + 1} to continue]"
    elif off > 1:
        body += f"\n[end of file: lines {off}-{end} of {total}]"
    return body


MATCH_STATS: dict[str, int] = {"exact": 0, "rstrip": 0, "indent": 0, "collapse": 0, "fail": 0}

_WS_RUN = re.compile(r"[ \t]+")


def _indent(s: str) -> str:
    return s[: len(s) - len(s.lstrip())]


def _tight(ln: str) -> str:
    """Trailing whitespace dropped, as an editor or a copy-paste may leave it."""
    return ln.rstrip()


def _stripped(ln: str) -> str:
    """Leading indentation dropped, so a dedented block still matches."""
    return ln.strip()


def _collapsed(ln: str) -> str:
    """Every run of spaces or tabs inside the line folded to one space."""
    return _WS_RUN.sub(" ", ln).strip()


_LADDER = (("rstrip", _tight), ("indent", _stripped), ("collapse", _collapsed))
_HOW = {
    "rstrip": "trailing-whitespace-tolerant unique match",
    "indent": "indentation-insensitive whitespace-tolerant unique match",
    "collapse": "collapsed-whitespace whitespace-tolerant unique match",
}


def _starts(hay: list[str], want: list[str], norm, limit: int | None = 2) -> list[int]:
    """0-based line indexes where want occurs line-aligned after norm, at most limit."""
    if not want or len(want) > len(hay):
        return []
    key = "\n".join(norm(ln) for ln in want)
    if not key.strip():
        return []
    body = "\n" + "\n".join(norm(ln) for ln in hay) + "\n"
    needle = "\n" + key + "\n"
    out: list[int] = []
    pos = body.find(needle)
    while pos != -1 and (limit is None or len(out) < limit):
        out.append(body.count("\n", 0, pos))
        pos = body.find(needle, pos + 1)
    return out


def _reindent(old: list[str], block: list[str], new: list[str]) -> list[str]:
    """Shift new from old_string indentation to the matched block indentation."""
    j = next((k for k, ln in enumerate(old) if ln.strip()), 0)
    oi = _indent(old[j])
    fi = _indent(block[j]) if j < len(block) else ""
    if oi == fi:
        return list(new)
    return ["" if not ln.strip() else (fi + ln[len(oi):] if ln.startswith(oi) else ln) for ln in new]


def _ladder(hay: list[str], old: list[str], new: list[str]):
    """-> (hit, ambiguous).

    hit is (stage, 0-based line, matched line count, replacement lines) for the
    first stage that matches the block exactly once, else None: a stage is used
    only when it is unique, so a partial or ambiguous match is never written.
    ambiguous lists (match count, 0-based lines) for every stage that matched
    more than once, so text that IS present is never reported as absent.
    """
    ambiguous: list[tuple[int, list[int]]] = []
    for stage, norm in _LADDER:
        hits = _starts(hay, old, norm, limit=_AMBIG_CAP + 1)
        if len(hits) == 1:
            i = hits[0]
            return (stage, i, len(old), _reindent(old, hay[i:i + len(old)], new)), []
        if len(hits) > 1:
            ambiguous.append((len(hits), hits))
    return None, ambiguous


def _ambiguity(hits: list[int]) -> str:
    """The exact stage's multi-match refusal, replayed for a tolerant stage."""
    count = f"at least {_AMBIG_CAP}" if len(hits) > _AMBIG_CAP else str(len(hits))
    lines = ", ".join(str(i + 1) for i in hits[:4])
    return (f"old_string matches {count} places (lines {lines}); add surrounding lines "
            "to make it unique, or set replace_all=true to change every one")


def _closest(hay: list[str], old: str) -> list[tuple[int, str]]:
    """Up to three (0-based line, snippet) candidates for the wanted text."""
    probe = [ln.strip() for ln in old.split("\n") if ln.strip()][:4]
    scaled = [ln.strip() for ln in hay[:50_000]]
    out: list[tuple[int, str]] = []
    for p in probe:
        for cand in difflib.get_close_matches(p, scaled, n=2, cutoff=0.6):
            i = scaled.index(cand)
            if all(i != j for j, _ in out):
                out.append((i, hay[i].strip()[:120]))
            if len(out) == 3:
                return out
    return out


def _not_found(hay: list[str], old: str) -> str:
    """Zero-match failure text: the closest lines, then the read that finds them."""
    msg = ("old_string was not found (no unique trailing-whitespace, indentation or "
           "collapsed-whitespace match either).")
    near = _closest(hay, old)
    if not near:
        return msg + " Read the file with read_file first and copy the text exactly."
    shown = "; ".join(f"line {i + 1}: {snip!r}" for i, snip in near)
    msg += (f" Closest lines are {shown}. Read them with read_file (offset={near[0][0] + 1}) "
            "and copy the text exactly.")
    return msg if len(msg) < 700 else msg[:696] + "..."


def _bounded_diff(old_block: list[str], new_block: list[str], budget: int) -> str:
    """Added/removed lines of the edited region, capped so the reply stays small."""
    # Exact edits often include long unchanged context. SequenceMatcher with
    # autojunk=False is quadratic on repeated lines, even for a one-line edit.
    # Remove equal edges first; the confirmation only displays changed lines.
    start = 0
    end_old, end_new = len(old_block), len(new_block)
    while start < min(end_old, end_new) and old_block[start] == new_block[start]:
        start += 1
    while end_old > start and end_new > start and old_block[end_old - 1] == new_block[end_new - 1]:
        end_old -= 1
        end_new -= 1
    old_block, new_block = old_block[start:end_old], new_block[start:end_new]
    # A short reply must not require an unbounded quadratic alignment. Large
    # interiors get an explicitly labelled region preview instead. This only
    # changes the confirmation, never the matched region or replacement bytes.
    coarse = (len(old_block) * len(new_block) > DIFF_MAX_PAIRS
              or len(old_block) + len(new_block) > DIFF_MAX_REGION_LINES)
    note = "[bounded region preview; unchanged interior lines may be included]"
    if coarse:
        budget = max(0, budget - len(note) - 1)
    if coarse:
        opcodes = [("replace", 0, len(old_block), 0, len(new_block))]
    else:
        opcodes = difflib.SequenceMatcher(
            a=old_block, b=new_block, autojunk=False).get_opcodes()
    rows: list[str] = []
    added = removed = 0
    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            continue
        for ln in old_block[i1:i2]:
            if len(rows) < DIFF_MAX_LINES:
                rows.append("-" + ln.rstrip("\r"))
            removed += 1
        for ln in new_block[j1:j2]:
            if len(rows) < DIFF_MAX_LINES:
                rows.append("+" + ln.rstrip("\r"))
            added += 1
    shown: list[str] = []
    used = 0
    shown_added = shown_removed = 0
    for row in rows:
        if len(row) > DIFF_ROW_CHARS:
            row = row[:DIFF_ROW_CHARS - 4] + " ..."
        if len(shown) >= DIFF_MAX_LINES or used + len(row) + 1 > budget:
            break
        shown.append(row)
        used += len(row) + 1
        if row.startswith("+"):
            shown_added += 1
        else:
            shown_removed += 1
    rest_added = added - shown_added
    rest_removed = removed - shown_removed
    if rest_added or rest_removed:
        shown.append(f"+{rest_added} -{rest_removed} lines not shown")
    if coarse:
        shown.append(note)
    return "\n".join(shown)


def _confirmation(path, line: int, how: str, old_block: list[str], new_block: list[str]) -> str:
    """One-line confirmation, then the bounded diff of what changed."""
    head = f"edited {path} at line {line} ({how})"
    budget = min(DIFF_MAX_CHARS, RESULT_MAX_CHARS - len(head) - 40)
    diff = _bounded_diff(old_block, new_block, budget) if budget > 0 else ""
    return head + ("\n" + diff if diff else "")


def plan_edit(box, args, *, count_failures: bool = False):
    """Compute the actual edit bytes/confirmation without writing the target.

    Recovery intents use this same exact/fuzzy/replace_all implementation, rather
    than attempting to infer a full postimage from replacement substrings.
    """
    path = args.get("path")
    old = args.get("old_string")
    new = args.get("new_string")
    every = bool(args.get("replace_all"))
    if not isinstance(old, str) or not isinstance(new, str):
        raise FileToolError("old_string and new_string must be strings")
    if old == "":
        raise FileToolError("old_string is empty; use write_file to create a file")
    if old == new:
        raise FileToolError("old_string and new_string are identical; nothing to change")
    p = box._path(path)
    if not p.is_file():
        raise FileToolError("not a file")
    if ".git" in p.relative_to(box.root).parts:
        raise FileToolError("not allowed")
    t, nl = _text(p, args.get("expected_sha256"))
    crlf = nl == "\r\n"
    if crlf:
        old, new = old.replace("\r\n", "\n").replace("\n", "\r\n"), new.replace("\r\n", "\n").replace("\n", "\r\n")
    n = t.count(old)
    if n == 1 or (n > 1 and every):
        pos = t.find(old)
        line = t.count("\n", 0, pos) + 1
        old_block = old.split("\n")
        new_block = new.split("\n")
        res = t.replace(old, new) if every else t[:pos] + new + t[pos + len(old):]
        stage = "exact"
        how = f"{n} exact match(es)"
    elif n > 1:
        if count_failures:
            MATCH_STATS["fail"] += 1
        ls = [t.count("\n", 0, m.start()) + 1 for m in re.finditer(re.escape(old), t)][:8]
        raise FileToolError(
            f"old_string matches {n} places (lines {', '.join(map(str, ls))}); add surrounding lines "
            "to make it unique, or set replace_all=true to change every one")
    else:
        hay = (t.replace("\r\n", "\n") if crlf else t).split("\n")
        hit, ambiguous = _ladder(hay, old.replace("\r\n", "\n").split("\n"),
                                 new.replace("\r\n", "\n").split("\n"))
        if hit is None:
            if count_failures:
                MATCH_STATS["fail"] += 1
            if ambiguous:
                # Present, just not unique: say so instead of "not found".
                raise FileToolError(_ambiguity(ambiguous[0][1]))
            raise FileToolError(_not_found(hay, old.replace("\r\n", "\n")))
        stage, i, k, repl = hit
        old_block = hay[i:i + k]
        new_block = repl
        hay[i:i + k] = repl
        res = "\n".join(hay)
        if crlf:
            res = res.replace("\n", "\r\n")
        line = i + 1
        how = _HOW[stage]
    if len(res) > 2_000_000:
        raise FileToolError("result too large")
    return p, res, stage, _confirmation(path, line, how, old_block, new_block)


def edit_file(box, args) -> str:
    target = box._path(args.get('path'))
    before = hashlib.sha256(target.read_bytes()).hexdigest()
    p, res, stage, confirmation = plan_edit(box, args, count_failures=True)
    MATCH_STATS[stage] += 1
    atomic_write(p, res, expected_sha256=before)
    if hasattr(box, '_observed_files'):
        box._observed_files[str(p)] = hashlib.sha256(res.encode('utf-8')).hexdigest()
    return confirmation


def atomic_write(p: Path, text: str, expected_sha256=None) -> None:
    """Write beside the target then rename; keeps the mode, writes bytes untranslated."""
    mode = p.stat().st_mode & 0o7777 if p.exists() else 0o644
    fd, tmp = tempfile.mkstemp(prefix=".ordivyr-lite-", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(text.encode("utf-8"))
        os.chmod(tmp, mode)
        if expected_sha256 is not None and (
                not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != expected_sha256):
            raise FileToolError('stale file identity: content changed while preparing the edit; current file preserved')
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _walk(box, sub, deadline, state: dict[str, bool] | None = None):
    """Yield workspace files; state["deadline"] records a walk cut short by the clock."""
    root = box._path(sub or ".")
    if root.is_file():
        yield root
        return
    for dp, dns, fns in os.walk(root, followlinks=False):
        dns[:] = sorted(d for d in dns if d not in SEARCH_SKIP and not os.path.islink(os.path.join(dp, d)))
        for f in sorted(fns):
            fp = Path(dp) / f
            if fp.is_symlink():
                continue
            yield fp
        if time.monotonic() > deadline:
            if state is not None:
                state["deadline"] = True
            return


def grep(box, args) -> str:
    pat = args.get("pattern")
    if not isinstance(pat, str) or not pat:
        raise FileToolError("pattern is required")
    flags = re.I if args.get("ignore_case") else 0
    try:
        rx = re.compile(pat if args.get("regex") else re.escape(pat), flags)
    except re.error as e:
        raise FileToolError(f"bad regex: {e}") from None
    glob = args.get("glob")
    deadline = time.monotonic() + SEARCH_SECONDS
    state: dict[str, bool] = {}
    hits: list[str] = []
    files = 0
    matched_files = 0
    total = 0
    for fp in _walk(box, args.get("path"), deadline, state):
        if glob and not fp.match(glob):
            continue
        try:
            if fp.stat().st_size > SEARCH_MAX_FILE:
                continue
            b = fp.read_bytes()
        except OSError:
            continue
        if b"\0" in b[:4096]:
            continue
        files += 1
        rel = fp.relative_to(box.root)
        found = False
        for i, ln in enumerate(b.decode("utf-8", "replace").split("\n"), 1):
            if rx.search(ln):
                # Keep counting past the display cap: the true total is the point.
                total += 1
                found = True
                if len(hits) < GREP_MAX_MATCHES:
                    hits.append(f"{rel}:{i}: {ln.strip()[:GREP_LINE_CAP]}")
        matched_files += found
    notes: list[str] = []
    if total > len(hits):
        notes.append(f"showing {len(hits)} of {total} matches in {matched_files} files; "
                     "narrow the pattern, path or glob")
    if state.get("deadline"):
        notes.append(DEADLINE_NOTE)
    body = "\n".join(hits) if hits else f"no matches ({files} files searched)"
    if notes:
        body += "\n[" + "; ".join(notes) + "]"
    return body


def find_files(box, args) -> str:
    glob = args.get("glob") or "*"
    deadline = time.monotonic() + SEARCH_SECONDS
    state: dict[str, bool] = {}
    out: list[str] = []
    more = False
    for fp in _walk(box, args.get("path"), deadline, state):
        rel = fp.relative_to(box.root)
        if fp.match(glob) or rel.match(glob):
            if len(out) >= FIND_MAX:
                more = True
                break
            out.append(str(rel))
    notes: list[str] = []
    if more:
        notes.append(f"stopped at {FIND_MAX} files; narrow the glob or path")
    if state.get("deadline"):
        notes.append(DEADLINE_NOTE)
    body = "\n".join(out) if out else "no files matched"
    if notes:
        body += "\n[" + "; ".join(notes) + "]"
    return body


def retain(box, out: str, limit: int) -> str:
    """Keep head AND tail; save the whole output in a task-local spill."""
    if not isinstance(out, str) or len(out) <= limit:
        return out
    n = box.spill(out)
    head = int(limit * 0.7)
    tail = limit - head
    return (f"{out[:head]}\n[... {len(out) - limit} chars omitted from the middle; full output "
            f"({len(out)} chars) saved as spill:{n} - read it with read_file path=\"spill:{n}\" "
            f"and offset/limit ...]\n{out[-tail:]}")
