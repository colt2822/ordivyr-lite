"""Deterministic protection for explicit unchanged-file instructions.

This recognizes ordinary preservation phrases, not arbitrary natural language.
It captures existing bytes once, survives resume, and never restores a file.
New regression files remain writable. Unknown mutation tools are refused while
protection is active; operator-owned validators still run and are checked after.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import stat

from .verification import manifest, PARTIAL

READS = frozenset({'read_file', 'list_files', 'grep', 'find_files', 'fetch_url'})
WRITES = frozenset({'write_file', 'edit_file'})
CONFIG = frozenset({'pytest.ini', 'tox.ini', 'setup.cfg', 'pyproject.toml',
    'package.json', 'package-lock.json', 'pnpm-lock.yaml', 'yarn.lock',
    'tsconfig.json', 'go.mod', 'go.sum'})
MAX_BYTES = 16_000_000
_FILE = re.compile(r'(?<![\w])([\w./-]+\.[A-Za-z0-9_]+)')
_NEGATIVE = re.compile(r"\b(?:do not|don't|never|must not)\s+(?:edit|modify|change|alter|rewrite|delete|remove)\b", re.I)
_PRESERVE = re.compile(r'\b(?:keep|leave|preserve|remains?|retain|stay)\b', re.I)
_UNCHANGED = re.compile(r'\b(?:unchanged|unmodified|untouched|intact|byte[- ]for[- ]byte)\b', re.I)


def is_test(path):
    p = PurePosixPath(path)
    name = p.name
    return (any(part in ('test', 'tests', '__tests__') for part in p.parts[:-1])
        or name == 'conftest.py' or name.startswith('test_') and name.endswith('.py')
        or name.endswith(('_test.py', '_test.go'))
        or bool(re.search(r'\.(test|spec)\.(?:[cm]?[jt]sx?)$', name)))


def directives(prompt):
    tests = configs = False
    paths = set()
    for clause in re.split(r'[.!?]\s+|[\n;]+', str(prompt or '')):
        if re.search(r"\b(?:do not|don't|need not)\s+(?:keep|preserve|retain)\b", clause, re.I):
            continue
        negative, preserve, unchanged = _NEGATIVE.search(clause), _PRESERVE.search(clause), _UNCHANGED.search(clause)
        declared = re.search(r'\b(?:are|is|must be)\s+(?:protected|immutable|read[- ]only)\b', clause, re.I)
        if not (negative or preserve and unchanged or declared):
            continue
        if re.search(r'\b(?:except|excluding|other than)\b', clause, re.I):
            raise ValueError('preservation exceptions require an explicit protected-file instruction; ambiguous edits were not enabled')
        if declared:
            subject = clause[:declared.start()]
        elif negative:
            subject = clause[negative.end():]
        elif preserve.group().lower().startswith(('remain', 'stay')):
            subject = clause[:preserve.start()]
        else:
            subject = clause[preserve.end():unchanged.start()]
        tests |= bool(re.search(r'\btests?\b', subject, re.I) and not re.search(
            r'\btests?\s+(?:coverage|imports?|behavior|semantics|names|output|configuration|config)\b', subject, re.I))
        configs |= bool(re.search(r'\b(?:config(?:uration)?s?|manifests?|lockfiles?)\b', subject, re.I) and not re.search(
            r'\bconfig(?:uration)?\s+(?:defaults?|values?|behavior|options)\b', subject, re.I))
        named = _FILE.findall(subject) + re.findall(r'`([^`\n]+)`', subject)
        for value in named:
            path = PurePosixPath(value)
            if not path.is_absolute() and '..' not in path.parts:
                paths.add(path.as_posix())
    return tests, configs, paths


def _safe_relative(value):
    p = PurePosixPath(value)
    return bool(value and not p.is_absolute() and '..' not in p.parts and str(p) == value)


def _digest(root, rel):
    path = root / rel
    if not _safe_relative(rel) or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('unsupported protected path: ' + rel)
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_NONBLOCK', 0))
    with os.fdopen(fd, 'rb') as stream:
        st = os.fstat(stream.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_size > MAX_BYTES:
            raise ValueError('unsupported protected input: ' + rel)
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise ValueError('protected input exceeds bounds: ' + rel)
    return hashlib.sha256(data).hexdigest(), len(data)


class TaskConstraints:
    def __init__(self, root, files=None):
        self.root = Path(root)
        self.files = dict(files or {})
        if any(not _safe_relative(p) or not re.fullmatch(r'[0-9a-f]{64}', str(d))
               for p, d in self.files.items()):
            raise ValueError('protected-file snapshot is invalid or incomplete')

    @classmethod
    def capture(cls, root, prompt, *, initial=None, resumed=False):
        tests, configs, paths = directives(prompt)
        if not (tests or configs or paths):
            return cls(root)
        if resumed and initial is None:
            raise ValueError('preservation baseline missing on resume; inspect the original files before continuing')
        source = initial if initial is not None else manifest(Path(root))
        selected = {p: d for p, d in source.items() if p != PARTIAL and
                    (tests and is_test(p) or configs and PurePosixPath(p).name in CONFIG or p in paths)}
        if initial is None:
            total = 0
            for path in selected:
                selected[path], size = _digest(Path(root), path)
                total += size
                if total > MAX_BYTES:
                    raise ValueError('protected-file baseline exceeds bounds')
        return cls(root, selected)

    def snapshot(self):
        return {'files': dict(self.files)}

    def reminder(self):
        if not self.files:
            return ''
        return ('Explicit task constraint: existing protected test/config files are immutable. '
                'Add regression coverage in separate NEW test files. Protected paths: '
                + ', '.join(sorted(self.files)[:20]))

    def refusal(self, box, name, args):
        if not self.files or name in READS:
            return None
        if name == 'run_command' and box.readonly_commands_available():
            return None  # ToolBox enforces a fresh whole-workspace read-only namespace.
        if name not in WRITES:
            return ('task constraints protect existing files; this mutation tool cannot enforce them. '
                    'Use file tools and the harness verifier.')
        try:
            path = box._path(args.get('path'), must_exist=False).relative_to(self.root).as_posix()
        except (OSError, ValueError):
            return 'protected-path check could not resolve the write target'
        if path in self.files:
            return ('task constraint: ' + path + ' must remain byte-for-byte unchanged. '
                    'Add regression tests in a separate NEW file.')
        return None

    def violation(self):
        total = 0
        for path, expected in self.files.items():
            try:
                current, size = _digest(self.root, path)
                total += size
                if total > MAX_BYTES:
                    return 'protected inputs exceed verification bounds; inspect them before continuing'
            except (OSError, ValueError):
                return 'protected original input is missing or unsupported: ' + path + '; inspect it before continuing'
            if current != expected:
                return 'protected original input changed: ' + path + '; current bytes preserved for manual review'
        return None
