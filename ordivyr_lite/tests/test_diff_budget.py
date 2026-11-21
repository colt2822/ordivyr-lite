"""Large confirmations must remain bounded without changing committed bytes."""
from unittest.mock import patch
import pytest
from ordivyr_lite import filetools as FT
from ordivyr_lite.tools import ToolBox

@pytest.mark.parametrize('size', [501, 3000, 12000])
def test_dispersed_edits_never_align_large_repetitive_interior(tmp_path, size):
    before = 'def value():\n' + '    pass\n' * size + '    return 0\n'
    after = 'def value():\n    x = 42\n' + '    pass\n' * (size-2) + '    x += 1\n    return x\n'
    target = tmp_path/'code.py'
    target.write_bytes(before.encode())
    with patch.object(FT.difflib, 'SequenceMatcher', side_effect=AssertionError('unbounded alignment')):
        output = ToolBox(tmp_path, profile='workspace').call('edit_file',
            {'path':'code.py', 'old_string':before, 'new_string':after})
    assert target.read_bytes() == after.encode()
    assert output.startswith('edited ')
    assert 'bounded region preview' in output
    assert len(output) <= FT.RESULT_MAX_CHARS
    namespace = {}
    exec(compile(target.read_bytes(), 'code.py', 'exec'), namespace)
    assert namespace['value']() == 43


def test_large_insert_has_bounded_preview_without_alignment():
    with patch.object(FT.difflib, 'SequenceMatcher', side_effect=AssertionError('alignment')):
        out = FT._bounded_diff([], ['new']*10000, 600)
    assert '+9992 -0 lines not shown' in out
    assert 'bounded region preview' in out


def test_small_diff_stays_precise():
    assert FT._bounded_diff(['a','x','a','y','a'], ['a','X','a','Y','a'], 600) == '-x\n+X\n-y\n+Y'


def test_single_change_in_long_context_remains_precise():
    before = ['same']*10000 + ['old'] + ['same']*10000
    after = ['same']*10000 + ['new'] + ['same']*10000
    assert FT._bounded_diff(before, after, 600) == '-old\n+new'
