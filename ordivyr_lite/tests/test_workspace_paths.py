"""Workspace absolute paths preserve the same confinement and effect identity."""
import pytest

from ordivyr_lite.constraints import TaskConstraints
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.providers import ToolCall
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tests.test_recovery_integrity import make_interrupted, resume
from ordivyr_lite.tools import ToolBox


def test_file_tools_accept_absolute_workspace_paths(tmp_path):
    target=tmp_path/'value.py';target.write_text('VALUE=1\n')
    box=ToolBox(tmp_path)
    try:
        assert 'VALUE=1' in box.call('read_file',{'path':str(target)})
        assert 'edited' in box.call('edit_file',{'path':str(target),'old_string':'VALUE=1','new_string':'VALUE=2'})
        assert not box.call('write_file',{'path':str(tmp_path/'new.txt'),'content':'new'}).startswith('ERROR')
        assert 'value.py' in box.call('list_files',{'path':str(tmp_path)})
        assert 'value.py' in box.call('grep',{'path':str(tmp_path),'pattern':'VALUE=2'})
        assert 'value.py' in box.call('find_files',{'path':str(tmp_path),'glob':'*.py'})
        assert target.read_text()=='VALUE=2\n'
    finally:box.close()


def test_outside_parent_and_symlink_paths_stay_refused(tmp_path):
    root=tmp_path/'root';root.mkdir();outside=tmp_path/'root-extra';outside.mkdir()
    (outside/'secret').write_text('preserve')
    (root/'inside').write_text('inside')
    (root/'link-out').symlink_to(outside,target_is_directory=True)
    (root/'link-in').symlink_to(root/'inside')
    box=ToolBox(root)
    try:
        for path in [outside/'secret',root/'..'/'root'/'inside',root/'link-out'/'secret',root/'link-in']:
            assert box.call('read_file',{'path':str(path)}).startswith('ERROR')
            assert box.call('write_file',{'path':str(path),'content':'bad'}).startswith('ERROR')
        assert (outside/'secret').read_text()=='preserve'
        assert (root/'inside').read_text()=='inside'
    finally:box.close()


def test_absolute_alias_does_not_bypass_stale_read_or_protected_files(tmp_path):
    target=tmp_path/'value.py';target.write_text('VALUE=1\n')
    box=ToolBox(tmp_path)
    try:
        box.call('read_file',{'path':str(target)});target.write_text('OWNER=2\n')
        assert 'stale' in box.call('write_file',{'path':'value.py','content':'bad'})
        assert target.read_text()=='OWNER=2\n'
        box.task_constraints=TaskConstraints.capture(tmp_path,'Keep value.py unchanged.')
        assert 'task constraint' in box.call('write_file',{'path':str(target),'content':'bad'})
        assert target.read_text()=='OWNER=2\n'
    finally:box.close()


def test_runner_tracks_workspace_paths_without_rewriting_recovery_arguments(tmp_path):
    (tmp_path/'value.py').write_text('VALUE=1\n')
    args={'path':str(tmp_path/'value.py'),'old_string':'VALUE=1','new_string':'VALUE=2'}
    transport=FakeTransport([openai_reply(calls=[('edit_file',args)]),openai_reply(text='done')])
    box=ToolBox(tmp_path)
    try:
        run=Runner(make_router(transport),box,'Change value.py to VALUE=2.',stream=False,
                   limits=RunLimits(max_turns=4))
        result=run.run()
        assert result.stop_reason=='done' and run.wrote==['value.py']
        assert result.stats.tool_errors==0
        assert args['path']==str(tmp_path/'value.py')
    finally:box.close()


def test_absolute_path_cannot_bypass_declared_scope(tmp_path):
    (tmp_path/'allowed').mkdir();outside=tmp_path/'outside.txt';outside.write_text('preserve')
    box=ToolBox(tmp_path)
    run=Runner(None,box,'task');run.ledger.scope=['allowed']
    try:
        out=run._run_tool(ToolCall('x','write_file',{'path':str(outside),'content':'bad'}))
        assert 'outside the declared scope' in out and outside.read_text()=='preserve'
        assert not run._run_tool(ToolCall('y','write_file',{'path':str(tmp_path/'allowed'/'new.txt'),'content':'ok'})).startswith('ERROR')
    finally:box.close()


@pytest.mark.parametrize('conflict',[False,True])
def test_absolute_write_recovery_stays_bound_to_original_arguments(tmp_path,conflict):
    args={'path':str(tmp_path/'target.txt'),'content':'agent\n'}
    box,journal,target=make_interrupted(tmp_path,'write_file',args,b'base\n')
    try:
        target.write_bytes(b'owner\n' if conflict else b'agent\n')
        runner,messages=resume(box,journal)
        assert ('conflict' in messages[-1]) if conflict else ('already been applied' in messages[-1])
        assert target.read_bytes()==(b'owner\n' if conflict else b'agent\n')
        if not conflict:assert runner.wrote==['target.txt']
    finally:box.close();journal.close()
