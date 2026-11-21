"""Content identity for interrupted built-in file tools. Recovery is read-only.

No file mutation is replayed, even on an exact preimage: the workspace has no
exclusive-writer contract. Digests attest to bytes observed during verification,
not permanent content, author identity, remote exactly-once effects or durability.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import stat

from . import filetools as FT
from .tools import ToolError

FORMAT = 'ordivyr-lite/file-intent/1'
MAX_IMAGE = FT.READ_MAX_FILE
_HEX = re.compile(r'^[0-9a-f]{64}$')


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _args_hash(args: dict) -> str:
    return _hash(json.dumps(args, sort_keys=True, ensure_ascii=True,
                            separators=(',', ':'), allow_nan=False).encode())


def _image(data: bytes) -> dict:
    return {'exists': True, 'bytes': len(data), 'sha256': _hash(data)}


def read_image(box, path):
    """Bounded bytes read through scoped, no-follow directory descriptors.

    Reject an observed replacement/change during the read. This is deliberately
    NOT a compare-and-swap or exclusion of subsequent/uncooperative writers.
    """
    parts = path.relative_to(box.root).parts
    if not parts or '.git' in parts:
        raise ValueError('unsupported recovery target')
    fd = os.open(box.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd); fd = child
        try:
            target = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            return {'exists': False, 'bytes': 0, 'sha256': None}
        try:
            before = os.fstat(target)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_IMAGE:
                raise ValueError('unsupported recovery image')
            chunks = []; total = 0
            while True:
                chunk = os.read(target, min(65536, MAX_IMAGE + 1 - total))
                if not chunk: break
                chunks.append(chunk); total += len(chunk)
                if total > MAX_IMAGE: raise ValueError('oversized recovery image')
            after = os.fstat(target)
            current = os.stat(parts[-1], dir_fd=fd, follow_symlinks=False)
            signature = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if signature(before) != signature(after) or signature(after) != signature(current):
                raise ValueError('recovery image changed during read')
            return _image(b''.join(chunks))
        finally:
            os.close(target)
    except FileNotFoundError:
        # Missing parent implies absent target, not permission to create it.
        return {'exists': False, 'bytes': 0, 'sha256': None}
    finally:
        os.close(fd)


def capture_intent(box, name: str, args: dict, run_id: str, tool_id: str) -> dict | None:
    """No new plaintext path/content metadata; unsupported plans stay unknown."""
    try:
        if name not in {'write_file', 'edit_file'} or name in box._handlers:
            return None
        if box.validate(name, args) or not box.write or len(run_id) > 120 or len(tool_id) > 256:
            return None
        path = box._path(args.get('path'), must_exist=False)
        pre = read_image(box, path)
        if name == 'write_file':
            content = args['content']
            if not isinstance(content, str) or len(content) > 400000:
                return None
            post = _image(content.encode('utf-8')); semantics = 'write:utf8/1'
        else:
            planned_path, content, stage, _ = FT.plan_edit(box, args)
            if planned_path != path:
                return None
            post = _image(content.encode('utf-8'))
            semantics = 'edit:' + stage + (':all/1' if bool(args.get('replace_all')) else ':one/1')
        if pre != read_image(box, path) or post['bytes'] > MAX_IMAGE:
            return None
        return {'format': FORMAT, 'algorithm': 'sha256', 'run_id': run_id, 'tool_id': tool_id,
                'tool': name, 'root_sha256': _hash(str(box.root).encode()),
                'target_sha256': _hash(str(path).encode()), 'args_sha256': _args_hash(args),
                'semantics': semantics, 'pre': pre, 'post': post}
    except (OSError, ValueError, TypeError, KeyError, ToolError):
        return None


def _valid_image(value):
    return (isinstance(value, dict) and set(value) == {'exists', 'bytes', 'sha256'}
            and type(value['exists']) is bool and type(value['bytes']) is int
            and 0 <= value['bytes'] <= MAX_IMAGE
            and (isinstance(value['sha256'], str) and bool(_HEX.fullmatch(value['sha256']))
                 if value['exists'] else value['bytes'] == 0 and value['sha256'] is None))


def classify(box, name, args, run_id, tool_id, identity):
    """Return postimage/preimage/conflict/unknown without dispatching a tool."""
    try:
        if not isinstance(identity, dict) or set(identity) != {
            'format', 'algorithm', 'run_id', 'tool_id', 'tool', 'root_sha256',
            'target_sha256', 'args_sha256', 'semantics', 'pre', 'post'}:
            return 'unknown'
        if name not in {'write_file', 'edit_file'} or name in box._handlers or box.validate(name, args):
            return 'unknown'
        path = box._path(args.get('path'), must_exist=False)
        semantics = identity['semantics']
        allowed = {'write:utf8/1'} if name == 'write_file' else {
            'edit:' + stage + (':all/1' if bool(args.get('replace_all')) else ':one/1')
            for stage in ['exact', 'rstrip', 'indent', 'collapse']}
        if (identity['format'] != FORMAT or identity['algorithm'] != 'sha256'
                or identity['run_id'] != run_id or identity['tool_id'] != tool_id
                or identity['tool'] != name or semantics not in allowed
                or identity['root_sha256'] != _hash(str(box.root).encode())
                or identity['target_sha256'] != _hash(str(path).encode())
                or identity['args_sha256'] != _args_hash(args)
                or not _valid_image(identity['pre']) or not _valid_image(identity['post'])
                or not identity['post']['exists']):
            return 'unknown'
        current = read_image(box, path)
        if current == identity['post']: return 'postimage'
        if current == identity['pre']: return 'preimage'
        return 'conflict'
    except (OSError, ValueError, TypeError, ToolError):
        return 'unknown'
