from pathlib import Path
import json

import pytest

from ordivyr_lite.loop import Runner,RunLimits
from ordivyr_lite.session import Session
from ordivyr_lite.tests.support import FakeTransport,make_router,openai_reply
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.validators import Validator


PROMPT='Implement the correct VALUE in value.py. It must equal 1. Keep existing tests unchanged.'


def setup(tmp_path):
    (tmp_path/'value.py').write_text('VALUE=0\n')
    (tmp_path/'test_value.py').write_text('from value import VALUE\ndef test_value():assert VALUE==1\n')
    return ToolBox(tmp_path,sandbox_mode='process')


def edit(old,new):
    return openai_reply(calls=[('edit_file',{'path':'value.py','old_string':f'VALUE={old}','new_string':f'VALUE={new}'})])


def test_review_uses_fresh_context_same_budget_and_no_tools(tmp_path):
    box=setup(tmp_path)
    first=openai_reply(text='ANCHOR_OLD_BELIEF',calls=[('edit_file',{'path':'value.py','old_string':'VALUE=0','new_string':'VALUE=2'})])
    transport=FakeTransport([first,edit(2,3),openai_reply(text='REPAIR: VALUE is 3 but the request requires 1.'),edit(3,1),openai_reply(text='done')])
    try:
        result=Runner(make_router(transport),box,PROMPT,validator=Validator('auto',sandbox_mode='process'),
                      stream=False,limits=RunLimits(max_turns=8,fresh_failure_review=True)).run()
        assert result.stop_reason=='done' and result.completion['verified']
        assert result.stats.model_calls==5==len(transport.requests)
        assert result.stats.tool_calls==3
        critic=transport.requests[2]
        assert not critic.get('tools') and critic['max_tokens']<=512
        assert 'ANCHOR_OLD_BELIEF' not in json.dumps(critic['messages'])
        assert 'VALUE=3' in json.dumps(critic['messages'])
        assert PROMPT in critic['messages'][1]['content']
        assert 'ANCHOR_OLD_BELIEF' in json.dumps(transport.requests[3]['messages'])
        assert result.completion['fresh_review_valid'] and result.completion['independent_review'] is False
        assert (tmp_path/'test_value.py').read_text().endswith('VALUE==1\n')
    finally:box.close()


def test_review_tool_calls_are_never_dispatched(tmp_path):
    box=setup(tmp_path)
    transport=FakeTransport([edit(0,2),edit(2,3),
        openai_reply(calls=[('write_file',{'path':'forbidden.txt','content':'not dispatched'})]),
        edit(3,1),openai_reply(text='done'),openai_reply(text='done')])
    try:
        result=Runner(make_router(transport),box,PROMPT,validator=Validator('auto',sandbox_mode='process'),stream=False,
                      limits=RunLimits(max_turns=8,fresh_failure_review=True)).run()
        assert not (tmp_path/'forbidden.txt').exists()
        assert result.stats.refused_calls==1 and result.stats.tool_calls==3
        assert result.completion['fresh_review_completed'] and not result.completion['fresh_review_valid']
        assert result.stats.model_calls==len(transport.requests)<=8
    finally:box.close()


def test_review_cannot_overrun_turn_budget(tmp_path):
    box=setup(tmp_path)
    transport=FakeTransport([edit(0,2),edit(2,3),openai_reply(text='done')])
    try:
        result=Runner(make_router(transport),box,PROMPT,validator=Validator('auto',sandbox_mode='process'),stream=False,
                      limits=RunLimits(max_turns=3,fresh_failure_review=True)).run()
        assert result.stats.model_calls==3 and not result.completion.get('fresh_review_used')
        assert all(request.get('tools') for request in transport.requests)
    finally:box.close()


def test_pending_review_resumes_without_replaying_edits(tmp_path):
    box=setup(tmp_path);session=Session.create(tmp_path.parent/(tmp_path.name+'-review.jsonl'))
    cancel=__import__('threading').Event()
    def event(value):
        if value['type']=='fresh_review_scheduled':cancel.set()
    first=FakeTransport([edit(0,2),edit(2,3)])
    try:
        result=Runner(make_router(first),box,PROMPT,session=session,validator=Validator('auto',sandbox_mode='process'),
                      stream=False,on_event=event,cancel_event=cancel,
                      limits=RunLimits(max_turns=8,fresh_failure_review=True)).run()
        assert result.stop_reason=='cancelled'
        state=session.latest_snapshot()
        assert state['fresh_review_pending'] and state['fresh_failure_count']==2
        assert (tmp_path/'value.py').read_text()=='VALUE=3\n'
        second=FakeTransport([openai_reply(text='REPAIR: set VALUE to 1.'),edit(3,1),openai_reply(text='done')])
        resumed=Runner(make_router(second),box,'',session=session,resume_state=state,
                       validator=Validator('auto',sandbox_mode='process'),stream=False,
                       limits=RunLimits(max_turns=8)).run()
        assert resumed.stop_reason=='done'
        assert not second.requests[0].get('tools') and resumed.stats.model_calls==5
        assert resumed.stats.tool_calls==3
    finally:box.close();session.close()


def test_interrupted_review_request_keeps_strict_resume_fail_closed(tmp_path):
    box=setup(tmp_path);session=Session.create(tmp_path.parent/(tmp_path.name+'-strict-review.jsonl'))
    count=[0]
    def respond(body):
        count[0]+=1
        if count[0]==1:return edit(0,2)
        if count[0]==2:return edit(2,3)
        raise KeyboardInterrupt('interrupted during review request')
    first=FakeTransport(respond)
    try:
        with pytest.raises(KeyboardInterrupt):
            Runner(make_router(first),box,PROMPT,session=session,validator=Validator('auto',sandbox_mode='process'),
                   stream=False,limits=RunLimits(max_turns=8,fresh_failure_review=True)).run()
        state=session.latest_snapshot()
        assert state['fresh_review_pending']
        second=FakeTransport([])
        result=Runner(make_router(second),box,'',session=session,resume_state=state,resume_policy='strict',
                      validator=Validator('auto',sandbox_mode='process'),stream=False,
                      limits=RunLimits(max_turns=8)).run()
        assert result.stop_reason=='unverified' and not second.requests
        assert 'uncertain' in result.error
    finally:box.close();session.close()


def test_healthy_task_keeps_original_call_count(tmp_path):
    box=ToolBox(tmp_path)
    transport=FakeTransport([openai_reply(calls=[('write_file',{'path':'new.py','content':'VALUE=1\n'})]),openai_reply(text='done')])
    try:
        result=Runner(make_router(transport),box,'Create new.py.',stream=False,
                      limits=RunLimits(max_turns=4,fresh_failure_review=True)).run()
        assert result.stop_reason=='done' and result.stats.model_calls==2
        assert not result.completion.get('fresh_review_used')
    finally:box.close()


def test_critic_does_not_inherit_a_self_authored_oracle(tmp_path):
    from ordivyr_lite.review import packet
    box=setup(tmp_path)
    (tmp_path/'test_extra.py').write_text('WRONG_TEST_EXPECTATION_POISON=True\n')
    (tmp_path/'debug_value.py').write_text('DEBUG_EXPECTATION_POISON=True\n')
    run=Runner(make_router(FakeTransport([])),box,PROMPT,stream=False)
    run.wrote=['value.py','test_extra.py','debug_value.py']
    try:
        messages=packet(run,'FAILURE_EXPECTATION_POISON')
        text=json.dumps(messages)
        assert 'VALUE=0' in text
        assert 'WRONG_TEST_EXPECTATION_POISON' not in text
        assert 'DEBUG_EXPECTATION_POISON' not in text
        assert 'FAILURE_EXPECTATION_POISON' not in text
    finally:box.close()
