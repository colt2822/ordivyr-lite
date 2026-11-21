"""Repository lifecycle and tool identity regressions, independent of model output."""
import hashlib
import os
import subprocess

import pytest

from ordivyr_lite.loop import Runner
from ordivyr_lite.memory import CodeIndex
from ordivyr_lite.mcpclient import exposed_name, McpClient, McpError
from ordivyr_lite.tools import ToolBox, ToolError


def test_cold_hint_and_small_tree(tmp_path, monkeypatch):
    monkeypatch.setenv('ORDIVYR_LITE_HOME', str(tmp_path / 'home'))
    root = tmp_path / 'repo'; root.mkdir()
    (root / 'billing.py').write_text('def invoiceBalance(): return 0\n')
    assert Runner(None, ToolBox(root), 'invoiceBalance')._context_hint() == ''
    for i in range(25):
        (root / f'other{i}.py').write_text(f'def unrelated{i}(): pass\n')
    assert 'billing.py' in Runner(None, ToolBox(root), 'invoiceBalance')._context_hint()


def test_live_and_loaded_index_observe_lifecycle(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    cache = tmp_path / 'cache'
    a = root / 'a.py'; a.write_text('def beforemarker(): pass\n')
    live = CodeIndex(root, cache_dir=cache)
    assert live.search('beforemarker')[0]['path'] == 'a.py'
    original = a.stat()
    a.write_text('def afterxmarker(): pass\n')
    os.utime(a, ns=(original.st_atime_ns, original.st_mtime_ns))
    for index in (live, CodeIndex(root, cache_dir=cache)):
        assert index.search('afterxmarker')[0]['path'] == 'a.py'
        assert index.search('beforemarker') == []
    a.rename(root / 'renamed.py')
    assert live.search('afterxmarker')[0]['path'] == 'renamed.py'
    (root / 'created.py').write_text('def createdmarker(): pass\n')
    assert live.search('createdmarker')[0]['path'] == 'created.py'
    (root / 'renamed.py').unlink()
    assert live.search('afterxmarker') == []
    assert live.refresh()['reindexed'] == 0


def test_branch_and_worktree_isolation(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], stderr=subprocess.DEVNULL)
    git('init', '-b', 'main')
    (root / 'a.py').write_text('def mainmarker(): pass\n')
    git('add', 'a.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'base')
    index = CodeIndex(root, cache_dir=tmp_path / 'cache')
    index.search('mainmarker')
    git('checkout', '-b', 'feature')
    (root / 'a.py').write_text('def featuremarker(): pass\n')
    git('add', 'a.py')
    git('-c', 'user.name=Test', '-c', 'user.email=test@example.invalid', 'commit', '-m', 'feature')
    assert index.search('featuremarker')
    git('checkout', 'main')
    assert index.search('mainmarker') and not index.search('featuremarker')
    other = tmp_path / 'other'
    git('worktree', 'add', str(other), 'feature')
    separate = CodeIndex(other, cache_dir=tmp_path / 'cache')
    assert separate.cache_path != index.cache_path
    assert separate.search('featuremarker') and not separate.search('mainmarker')


def test_replaced_parent_symlink_has_no_stale_hits(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    nested = root / 'nested'; nested.mkdir()
    (nested / 'a.py').write_text('oldmarker')
    outside = tmp_path / 'outside'; outside.mkdir()
    (outside / 'a.py').write_text('secretmarker')
    index = CodeIndex(root, cache_dir=tmp_path / 'cache')
    assert index.search('oldmarker')
    (nested / 'a.py').unlink(); nested.rmdir(); nested.symlink_to(outside, target_is_directory=True)
    assert not index.search('oldmarker') and not index.search('secretmarker')


def test_schema_fidelity_validation_and_registration(tmp_path):
    schema = {'type': 'object', 'properties': {
        'rows': {'type': 'array', 'items': {'type': 'object', 'properties': {
            'n': {'type': 'integer', 'minimum': 0}}, 'required': ['n'], 'additionalProperties': False}},
        'mode': {'type': 'string', 'enum': ['safe']}},
        'required': ['rows', 'mode'], 'additionalProperties': False}
    box = ToolBox(tmp_path); calls = []
    box.add_tool('custom', 'old', schema, lambda a: calls.append(a) or 'old')
    schema['properties']['mode']['enum'].append('mutated')
    wire = box.request_spec(['custom'])[0]['function']['parameters']
    assert wire['properties']['mode']['enum'] == ['safe']
    for args in ({'rows': [{'n': -1}], 'mode': 'safe'}, {'rows': [{'n': True}], 'mode': 'safe'},
                 {'rows': [{'n': '1'}], 'mode': 'safe'}, {'rows': [], 'mode': 'bad'},
                 {'rows': [], 'mode': 'safe', 'extra': 1}):
        assert box.call('custom', args).startswith('ERROR:')
    assert not calls
    assert box.call('custom', {'rows': [{'n': 1}], 'mode': 'safe'}) == 'old'
    box.add_tool('custom', 'new', {'type': 'object'}, lambda a: 'new')
    assert box.request_spec(['custom'])[0]['function']['description'] == 'new'
    assert box.call('custom', {}) == 'new'
    with pytest.raises(ToolError):
        box.add_tool('read_file', 'shadow', {}, lambda a: 'bad')


def test_mcp_names_do_not_alias():
    assert exposed_name('docs-one', 'get') != exposed_name('docs_one', 'get')
    assert exposed_name('a' * 25, 'get') != exposed_name('a' * 26, 'get')
    assert exposed_name('docs', 'get') != exposed_name('Docs', 'get')
    assert len(exposed_name('a' * 100, 'b' * 100)) <= 64
    assert exposed_name('a_b', 'c') != exposed_name('a', 'b_c')


def test_oversized_mcp_schema_is_refused(monkeypatch):
    client = McpClient('docs', [])
    monkeypatch.setattr(client, 'request', lambda *a: {'tools': [
        {'name': 'large', 'inputSchema': {'description': 'x' * 2100}}]})
    with pytest.raises(McpError, match='schema exceeds'):
        client.list_tools()


def test_stale_edit_digest_and_read_identity(tmp_path):
    a = tmp_path / 'a.py'; a.write_text('before = 1\n')
    box = ToolBox(tmp_path); runner = Runner(None, box, 'fix')
    sig = runner._path_signature('a.py'); st = a.stat()
    digest = hashlib.sha256(a.read_bytes()).hexdigest()
    a.write_text('before = 2\n'); os.utime(a, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert runner._path_signature('a.py') != sig
    args = {'path': 'a.py', 'old_string': 'before', 'new_string': 'after', 'expected_sha256': digest}
    assert 'stale file identity' in box.call('edit_file', args)
    assert a.read_text() == 'before = 2\n'
    args['expected_sha256'] = hashlib.sha256(a.read_bytes()).hexdigest()
    assert not box.call('edit_file', args).startswith('ERROR:')
    assert a.read_text() == 'after = 2\n'


def test_rare_declared_symbol_survives_body_vocabulary(tmp_path):
    root = tmp_path / 'repo'; root.mkdir()
    (root / 'hidden.py').write_text((' '.join(f'word{i}' for i in range(500)) + '\n') * 3
                                    + 'def uncommonExportedSymbol(): pass\n')
    index = CodeIndex(root, cache_dir=tmp_path / 'cache')
    assert index.search('uncommonExportedSymbol')[0]['path'] == 'hidden.py'


def test_digest_precondition_retains_recovery_identity(tmp_path):
    from ordivyr_lite import recovery
    path=tmp_path/'source.py';path.write_bytes(b'value = 1\n')
    box=ToolBox(tmp_path)
    args={'path':'source.py','old_string':'1','new_string':'2',
          'expected_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    identity=recovery.capture_intent(box,'edit_file',args,'run','call')
    assert identity is not None
    assert recovery.classify(box,'edit_file',args,'run','call',identity)=='preimage'
    assert not box.call('edit_file',args).startswith('ERROR:')
    assert recovery.classify(box,'edit_file',args,'run','call',identity)=='postimage'
    assert path.read_bytes()==b'value = 2\n'
