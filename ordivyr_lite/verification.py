"""Bounded Python impact analysis. Selection is an optimization, never a proof.

No commands, imports of project code, writes, or mutation replay happen here.
Unknown/dynamic dependencies and non-Python changes require the whole suite.
"""
from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path

SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache',
        '.ordivyr-spill', 'dist', 'build', '.mypy_cache'}
MAX_FILES = 5000
MAX_BYTES = 16_000_000


def inventory(root: Path) -> dict[str, str]:
    """Hash source/config/test inputs, including additions and deletions.

    Bounds fail explicitly rather than silently returning an incomplete identity.
    Symlinks are represented but never followed.
    """
    result = {}
    size = 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP)
        links = [d for d in dirs if (Path(directory) / d).is_symlink()]
        dirs[:] = [d for d in dirs if d not in links]
        for name in sorted([*files, *links]):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                data = ('symlink:' + os.readlink(path)).encode()
            else:
                # A repository may contain FIFOs or a concurrently replaced
                # entry. Never block opening/reading a non-regular input, and
                # bound actual bytes even if a file grows after fstat.
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
                fd = os.open(path, flags)
                with os.fdopen(fd, 'rb') as stream:
                    info = os.fstat(stream.fileno())
                    if not stat.S_ISREG(info.st_mode):
                        raise ValueError('non-regular verification input: ' + rel)
                    if info.st_size + size > MAX_BYTES:
                        raise ValueError('verification inventory byte limit exceeded')
                    data = stream.read(MAX_BYTES - size + 1)
            size += len(data)
            result[rel] = hashlib.sha256(data).hexdigest()
            if len(result) > MAX_FILES or size > MAX_BYTES:
                raise ValueError('verification inventory limit exceeded')
    return result


PARTIAL = '\0partial'
_STAT_CACHE: dict[tuple[str, str], tuple[tuple[int, ...], str]] = {}
_RACY_SECONDS = 2.0


def manifest(root: Path) -> dict[str, str]:
    """Bounded manifest. Exact content hashes while the tree fits the bounds.

    Beyond MAX_FILES/MAX_BYTES it no longer raises: files past the budget are
    identified by (device, inode, size, mtime_ns, ctime_ns) signatures, reusing an earlier content
    hash when that signature is unchanged and older than the racy window, and the
    ``PARTIAL`` marker is set. Partial manifests weaken change detection for the
    overflow files, so select_tests() always runs the whole suite on them.
    Non-regular inputs still raise, exactly as inventory() does.
    """
    try:
        return inventory(root)
    except ValueError as exc:
        if 'limit exceeded' not in str(exc):
            raise
    result: dict[str, str] = {PARTIAL: '1'}
    budget = MAX_BYTES
    now = time.time()
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            rel = path.relative_to(root).as_posix()
            if path.is_symlink():
                result[rel] = hashlib.sha256(('symlink:' + os.readlink(path)).encode()).hexdigest()
                continue
            st = path.lstat()
            if not stat.S_ISREG(st.st_mode):
                raise ValueError('non-regular verification input: ' + rel)
            sig = (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            cached = _STAT_CACHE.get((str(root), rel))
            if cached and cached[0] == sig and now - max(st.st_mtime_ns, st.st_ctime_ns) / 1e9 > _RACY_SECONDS:
                result[rel] = cached[1]
                continue
            if len(result) <= MAX_FILES and st.st_size <= budget:
                flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0)
                with os.fdopen(os.open(path, flags), 'rb') as stream:
                    opened = os.fstat(stream.fileno())
                    if not stat.S_ISREG(opened.st_mode):
                        raise ValueError('non-regular verification input: ' + rel)
                    opened_sig = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
                    if opened_sig != sig:
                        raise ValueError('verification input changed while reading: ' + rel)
                    data = stream.read(min(st.st_size, budget) + 1)
                    after = os.fstat(stream.fileno())
                    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != sig:
                        raise ValueError('verification input changed while reading: ' + rel)
                budget -= len(data)
                digest = hashlib.sha256(data).hexdigest()
                _STAT_CACHE[(str(root), rel)] = (sig, digest)
            else:
                digest = 'sig:' + ':'.join(map(str, sig))
            result[rel] = digest
    return result


_TRIGGERS = (
    ('dependency', re.compile(r'(^|/)(requirements[^/]*\.txt|pyproject\.toml|setup\.(py|cfg)|Pipfile(\.lock)?|poetry\.lock|'
                              r'package(-lock)?\.json|yarn\.lock|pnpm-lock\.yaml|go\.(mod|sum)|Cargo\.(toml|lock))$')),
    ('migration', re.compile(r'(^|/)(migrations?|alembic)/|\.sql$')),
    ('schema', re.compile(r'(\.schema\.json|\.proto|\.graphql|openapi[^/]*\.(json|ya?ml)|schema[^/]*\.(json|ya?ml))$', re.I)),
    ('export', re.compile(r'(^|/)(__init__\.py|index\.(js|jsx|ts|tsx|mjs)|mod\.rs)$')),
)


def change_triggers(changed: list[str]) -> dict[str, list[str]]:
    """Changes that can break consumers outside the edited file's own tests."""
    found: dict[str, list[str]] = {}
    for path in changed:
        for name, rx in _TRIGGERS:
            if rx.search(path):
                found.setdefault(name, []).append(path)
    return found


_NODE = re.compile(r'^(?:FAILED|ERROR) (\S+)(?: - (.*))?$')


def condense_report(report: str, limit: int = 1600) -> str:
    """Failing nodes, assertion detail and the last frames; not the whole log.

    The raw checker output stays in the evidence record. If nothing structured is
    recognised the tail is returned, so no failure is ever reduced to silence.
    """
    text = str(report or '')
    if len(text) <= limit:
        return text
    lines = text.splitlines()
    keep: list[str] = []
    stage = [ln for ln in lines if ln.startswith('[') and ln.endswith(']') and len(ln) < 20][-1:]
    summary = [ln for ln in lines if re.match(r'^=+ .*(failed|error|passed).* =+$', ln) or
               re.match(r'^\d+ (failed|passed|error)', ln)][-1:]
    failed = [ln for ln in lines if _NODE.match(ln)][:6]
    detail = [ln for ln in lines if ln.startswith('E ')][:14]
    frames = [ln for ln in lines if re.match(r'^[\w./-]+\.\w+:\d+: ', ln)][-4:]
    errors = [ln for ln in lines if re.match(r'^\w*(Error|Exception)\b', ln) or 'SyntaxError' in ln][-4:]
    for group in (stage, failed, detail, frames, errors, summary):
        for ln in group:
            if ln not in keep:
                keep.append(ln.rstrip())
    out = '\n'.join(keep)
    if not keep:
        return text[-limit:]
    if len(out) > limit:
        out = out[:limit - 20] + '\n[... clipped ...]'
    return out


@dataclass
class Selection:
    tests: list[str]
    reason: str
    broad: bool = False

    def as_dict(self):
        return {'tests': self.tests, 'reason': self.reason, 'broad': self.broad}


def select_tests(root: Path, changed: list[str], files: dict[str, str]) -> Selection:
    tests = sorted(p for p in files if p.endswith('.py') and
                   (Path(p).name.startswith('test_') or Path(p).name.endswith('_test.py')))
    def full(reason):
        return Selection(tests, reason, True)
    if not changed:
        return full('no reliable changed-file set')
    if PARTIAL in files:
        return full('partial manifest (tree exceeds verification bounds)')
    triggers = change_triggers(changed)
    if triggers:
        return full('change can affect consumers: ' + ', '.join(sorted(triggers)))
    if any(not p.endswith('.py') or p not in files for p in changed):
        return full('configuration, non-Python change or deletion')
    modules = {p: p[:-3].replace('/', '.').removeprefix('src.')
               .removesuffix('.__init__') for p in files if p.endswith('.py')}
    reverse: dict[str, set[str]] = {p: set() for p in modules}
    for path, module in modules.items():
        target = root / path
        if target.is_symlink():
            return full('symlink in Python inputs')
        try:
            tree = ast.parse(target.read_text('utf-8'))
        except (ValueError, SyntaxError, UnicodeError, OSError):
            return full('unparseable Python input')
        deps = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and (
                isinstance(node.func, ast.Name) and node.func.id in ('__import__', 'exec', 'eval')
                or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module'
            ):
                return full('dynamic dependency')
            if isinstance(node, ast.Import):
                deps.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ''
                if node.level:
                    package = module.split('.') if target.name == '__init__.py' else module.split('.')[:-1]
                    package = package[:len(package) - node.level + 1]
                    base = '.'.join([*package, *([base] if base else [])])
                deps.add(base)
                deps.update(base + '.' + a.name for a in node.names)
        for dependency, name in modules.items():
            if any(d == name or d.startswith(name + '.') for d in deps):
                reverse[dependency].add(path)
    affected = set(changed)
    pending = list(changed)
    while pending:
        for consumer in reverse.get(pending.pop(), ()):
            if consumer not in affected:
                affected.add(consumer)
                pending.append(consumer)
    focused = sorted(set(tests) & affected)
    return Selection(focused, 'transitive Python imports') if focused else full('no mapped tests')


def failure_kind(report: str) -> str:
    """Conservative hints; import errors may be dependency or implementation bugs."""
    if 'validator could not run:' in report or 'timed_out=True' in report:
        return 'environment'
    if 'ModuleNotFoundError' in report or 'ImportError' in report:
        return 'import_or_dependency'
    if 'collected 0 items' in report or 'no tests ran' in report:
        return 'no_tests'
    return 'code_or_test'
