"""Persistent acceptance checklist, extracted deterministically from the task text.

No model call is made. Each sentence of the request that states an obligation is
kept as a requirement with a kind (behavior, interface, return, default,
dependency, caller, verification, completion). Requirements that name a concrete
code symbol or file to add/create/rename can be checked mechanically against the
workspace (``satisfied`` / ``open``); everything else stays ``unchecked`` and is
only ever backed by test evidence. An ``open`` item blocks a verified result.
"""
from __future__ import annotations

import ast
import os
import re
from pathlib import Path

# Requirements are task state, not a summary. Never silently clip obligations.
SKIP = {'.git', '.venv', 'venv', 'node_modules', '__pycache__', '.pytest_cache', 'dist', 'build'}
SCAN_BYTES = 2_000_000

_SENT = re.compile(r'(?<=[.!?])\s+(?=[A-Za-z`"\'(])|\n+')
_OBLIGATION = re.compile(r'\b(must|should|shall|need(?:s)? to|require[sd]?|add|create|implement|define|'
                         r'introduce|expose|rename|return[s]?|default[s]?|accept[s]?|raise[s]?|support|'
                         r'fix|repair|ensure|make|keep|preserve|remove|update|change|treat|use|'
                         r'yields?|reject|stop|wait|propagate|skip|strip|decode|traverse|merge|ignore|'
                         r'contain[s]?|include[s]?|never|do not|don.t)\b', re.I)
_KINDS = (
    ('verification', re.compile(r'\b(tests?|pytest|verify|verification|run the|go test|npm test|build passes)\b', re.I)),
    ('default', re.compile(r'\bdefaults?\b|\bwhen (?:omitted|no argument|called with no)', re.I)),
    ('return', re.compile(r'\breturns?\b|\bresult(?:s)? (?:is|are)\b|\bmust yield\b', re.I)),
    ('dependency', re.compile(r'\b(depend(?:s|ency|encies)|requirements?\.txt|pyproject|package\.json|go\.mod|install|import(?:s)? from)\b', re.I)),
    ('caller', re.compile(r'\b(callers?|call sites?|every use|all uses|update (?:all|the) (?:uses|callers))\b', re.I)),
    ('interface', re.compile(r'\b(signature|parameter|keyword|argument|public (?:api|interface)|export|attribute|method|class|function|option)\b', re.I)),
    ('completion', re.compile(r'\b(complete the|until|done when|finish)\b', re.I)),
)
_TICK = re.compile(r'`([^`\n]{1,80})`')
_IDENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')
_PATH = re.compile(r'^[\w./-]+\.[A-Za-z0-9]{1,5}$')
_CREATE = re.compile(r'\b(add|create|define|implement|introduce|expose|rename|write)\b', re.I)
_EXT = {'py', 'js', 'ts', 'go', 'json', 'toml', 'md', 'txt', 'sql'}
_IGNORABLE = re.compile(r'^(work in the supplied|you may add regression|no network|preserve all unrelated)', re.I)


def _kind(sentence: str) -> str:
    for kind, rx in _KINDS:
        if rx.search(sentence):
            return kind
    return 'behavior'


def extract(prompt: str) -> list[dict]:
    items: list[dict] = []
    for raw in _SENT.split(str(prompt or '')):
        s = ' '.join(raw.split()).lstrip('-*• ').strip()
        if len(s) < 12 or _IGNORABLE.search(s) or not _OBLIGATION.search(s):
            continue
        item = {'id': len(items) + 1, 'kind': _kind(s), 'text': s, 'status': 'unchecked'}
        if _CREATE.search(s):
            checks = []
            for tok in _TICK.findall(s):
                tok = tok.strip()
                if _PATH.match(tok) and ('/' in tok or tok.rsplit('.', 1)[-1] in _EXT):
                    checks.append({'type': 'path', 'path': tok})
                elif _IDENT.match(tok) and len(tok) > 2:
                    checks.append({'type': 'symbol', 'name': tok})
            if checks:
                scoped = re.findall(r'\b(?:in|to|inside|within)\s+`?([A-Za-z0-9_./-]+\.py)`?', s)
                if len(set(scoped)) == 1:
                    for check in checks:
                        if check['type'] == 'symbol': check['path'] = scoped[0]
                item['checks'] = checks
                item['status'] = 'open'
        items.append(item)
    return items


def _symbols(root: Path, only: Path | None = None) -> set[str]:
    names: set[str] = set()
    seen = 0
    walk = [(only.parent, [], [only.name])] if only is not None else os.walk(root, followlinks=False)
    for directory, dirs, files in walk:
        dirs[:] = [d for d in dirs if d not in SKIP]
        for name in files:
            path = Path(directory) / name
            if path.is_symlink() or not name.endswith('.py'):
                continue
            try:
                if path.stat().st_size > 400_000 or seen > SCAN_BYTES:
                    continue
                seen += path.stat().st_size
                tree = ast.parse(path.read_text('utf-8', 'replace'))
            except (OSError, SyntaxError, ValueError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(node.name)
                    if not isinstance(node, ast.ClassDef):
                        names.update(a.arg for a in (*node.args.args, *node.args.kwonlyargs, *node.args.posonlyargs))
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    names.add(node.id)
                elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Store):
                    names.add(node.attr)
    return names


def evaluate(items: list[dict], root: Path) -> list[dict]:
    """Refresh mechanical statuses in place; unchecked items are never promoted."""
    symbols: set[str] | None = None
    scoped_symbols: dict[str, set[str]] = {}
    for item in items:
        checks = item.get('checks')
        if not checks:
            continue
        ok = True
        for check in checks:
            if check['type'] == 'path':
                ok = ok and (root / check['path']).is_file()
            else:
                if check.get('path'):
                    rel = check['path']
                    if rel not in scoped_symbols:
                        path = root / rel
                        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                            scoped_symbols[rel] = set()
                        else:
                            scoped_symbols[rel] = _symbols(root, only=path)
                    ok = ok and check['name'] in scoped_symbols[rel]
                    continue
                # Python symbol presence is not meaningful for JS/Go projects.
                if not any(root.rglob('*.py')):
                    item['status'] = 'unchecked'
                    ok = None
                    break
                if symbols is None:
                    symbols = _symbols(root)
                ok = ok and check['name'] in symbols
        if ok is not None:
            item['status'] = 'satisfied' if ok else 'open'
    return items


def summary(items: list[dict]) -> dict:
    count = {'satisfied': 0, 'open': 0, 'unchecked': 0}
    for item in items:
        count[item.get('status', 'unchecked')] = count.get(item.get('status', 'unchecked'), 0) + 1
    return {'total': len(items), **count, 'open_ids': [i['id'] for i in items if i.get('status') == 'open']}


def render(items: list[dict], *, only_open: bool = False) -> str:
    rows = [f"{i['id']}. [{i.get('status', 'unchecked')}] ({i['kind']}) {i['text']}"
            for i in items if not only_open or i.get('status') == 'open']
    return '\n'.join(rows)
