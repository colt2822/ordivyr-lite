import json

from ordivyr_lite.constraints import TaskConstraints
from ordivyr_lite.loop import Runner, RunLimits
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator


def test_protected_task_never_advertises_unenforceable_shell_and_still_verifies(tmp_path,monkeypatch):
    monkeypatch.setattr('ordivyr_lite.sandbox.bwrap_path',lambda:None)
    (tmp_path/'value.py').write_text('VALUE=0\n')
    (tmp_path/'test_value.py').write_text('from value import VALUE\ndef test_value():assert VALUE==1\n')
    old=(tmp_path/'test_value.py').read_bytes()
    transport=FakeTransport([
        openai_reply(calls=[('edit_file',{'path':'value.py','old_string':'VALUE=0','new_string':'VALUE=1'})]),
        openai_reply(text='Implemented and checked.')])
    box=ToolBox(tmp_path,profile='trusted',sandbox_mode='process')
    try:
        result=Runner(make_router(transport),box,'Fix value.py. Keep existing tests unchanged. Run the relevant tests.',
                      validator=Validator('auto',sandbox_mode='process'),stream=False,limits=RunLimits(max_turns=4)).run()
        assert result.stop_reason=='done'
        assert result.stats.tool_errors==0 and result.completion['verified']
        for request in transport.requests:
            assert 'run_command' not in [t['function']['name'] for t in request.get('tools',[])]
        system=transport.requests[0]['messages'][0]['content']
        assert 'Do not call run_command' in system and 'harness runs detected project tests' in system
        assert 'Tools enabled: '+', '.join(box.available_names()) in system
        assert (tmp_path/'test_value.py').read_bytes()==old
        # Advertising changes never weaken enforcement of a guessed tool name.
        assert box.call('run_command',{'command':'echo overwritten > test_value.py'}).startswith('ERROR: task constraints')
        assert (tmp_path/'test_value.py').read_bytes()==old
    finally:box.close()


def test_followup_chat_refreshes_effective_tools_for_new_protection(tmp_path,monkeypatch):
    monkeypatch.setattr('ordivyr_lite.sandbox.bwrap_path',lambda:None)
    (tmp_path/'test_value.py').write_text('def test_value(): assert True\n')
    transport=FakeTransport([openai_reply(text='No change needed.')])
    box=ToolBox(tmp_path,profile='trusted',sandbox_mode='process')
    try:
        Runner(make_router(transport),box,'Inspect files. Keep existing tests unchanged.',
               initial_messages=[{'role':'system','content':'Earlier invocation allowed run_command.'}],
               stream=False,limits=RunLimits(max_turns=2)).run()
        first=transport.requests[0]
        assert 'run_command' not in [t['function']['name'] for t in first['tools']]
        assert any('Do not call run_command' in m.get('content','') for m in first['messages'] if m['role']=='user')
    finally:box.close()


def test_prefetch_includes_commonjs_without_repeating_root_guidance(tmp_path):
    (tmp_path/'AGENTS.md').write_text('PROJECT_RULE_UNIQUE: preserve exports.\n')
    (tmp_path/'value.cjs').write_text('exports.VALUE=0;\n')
    transport=FakeTransport([openai_reply(text='Inspected.')])
    box=ToolBox(tmp_path)
    try:
        Runner(make_router(transport),box,'Read value.cjs.',stream=False,limits=RunLimits(max_turns=2)).run()
        messages=transport.requests[0]['messages']
        assert 'exports.VALUE=0' in json.dumps(messages)
        assert json.dumps(messages).count('PROJECT_RULE_UNIQUE')==1
    finally:box.close()
