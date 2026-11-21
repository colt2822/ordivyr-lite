"""Incomplete repository identities cannot certify reusable validation."""
import os
from contextlib import closing

from ordivyr_lite import verification as V
from ordivyr_lite import validators as VD
from ordivyr_lite.loop import Runner
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply


def workspace(root):
    (root / 'value.txt').write_text('good')
    (root / 'test_value.py').write_text("from pathlib import Path\ndef test_value(): assert Path('value.txt').read_text() == 'good'\n")


def replace_preserving_mtime(path):
    before = path.stat()
    path.write_text('evil')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))


def test_partial_manifest_cannot_reuse_validation_after_external_edit(tmp_path, monkeypatch):
    workspace(tmp_path); monkeypatch.setattr(V, 'MAX_BYTES', 0)
    with closing(ToolBox(tmp_path, sandbox_mode='process')) as box:
        validator = VD.Validator('tests', sandbox_mode='process'); validator.prepare(box)
        runner = Runner(make_router(FakeTransport(lambda b: openai_reply(text='done'))), box,
                        'Check the value.', stream=False, validator=validator)
        assert runner._verify_current_state()[1].ok is True
        replace_preserving_mtime(tmp_path / 'value.txt')
        assert runner._verify_current_state()[1].ok is False
        assert runner.stats.skipped_validations == 0


def test_partial_manifest_without_changes_still_runs_fresh_checks(tmp_path, monkeypatch):
    workspace(tmp_path); monkeypatch.setattr(V, 'MAX_BYTES', 0)
    with closing(ToolBox(tmp_path, sandbox_mode='process')) as box:
        validator = VD.Validator('tests', sandbox_mode='process'); validator.prepare(box)
        runner = Runner(make_router(FakeTransport(lambda b: openai_reply(text='done'))), box,
                        'Check the value.', stream=False, validator=validator)
        assert runner._verify_current_state()[1].ok is True
        assert runner._verify_current_state()[1].ok is True
        assert runner.stats.skipped_validations == 0


def test_partial_identity_detects_same_size_mtime_edit_after_tests(tmp_path, monkeypatch):
    workspace(tmp_path); monkeypatch.setattr(V, 'MAX_BYTES', 0)
    original = VD._run_command
    def mutate_after_tests(*args, **kwargs):
        result = original(*args, **kwargs)
        assert result[0] is True
        replace_preserving_mtime(tmp_path / 'value.txt')
        return result
    monkeypatch.setattr(VD, '_run_command', mutate_after_tests)
    with closing(ToolBox(tmp_path, sandbox_mode='process')) as box:
        validator = VD.Validator('tests', sandbox_mode='process'); validator.prepare(box)
        result = validator(box, [])
        assert result.ok is False and 'changed while tests ran' in result.report


def test_partial_stat_cache_invalidates_same_size_mtime_edit(tmp_path, monkeypatch):
    (tmp_path / 'a.txt').write_text('good'); (tmp_path / 'b.txt').write_text('overflow')
    monkeypatch.setattr(V, 'MAX_FILES', 1)
    monkeypatch.setattr(V, '_RACY_SECONDS', -1)
    V._STAT_CACHE.clear()
    before = V.manifest(tmp_path)
    assert V.PARTIAL in before
    replace_preserving_mtime(tmp_path / 'a.txt')
    after = V.manifest(tmp_path)
    assert after['a.txt'] != before['a.txt']
