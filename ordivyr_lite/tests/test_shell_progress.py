from ordivyr_lite.loop import Runner,RunLimits
from ordivyr_lite.tools import ToolBox
from ordivyr_lite.tests.support import FakeTransport,make_router,openai_reply


def run(tmp_path, replies):
    box=ToolBox(tmp_path,sandbox_mode='process',allow_shell=True)
    events=[]
    try:
        result=Runner(make_router(FakeTransport(replies)),box,'Fix the project files.',stream=False,
                      on_event=events.append,limits=RunLimits(max_turns=30)).run()
        return result,events
    finally:box.close()


def test_successful_diagnostics_do_not_hide_no_progress(tmp_path):
    replies=[openai_reply(calls=[('run_command',{'command':'true'})]) for _ in range(24)]
    result,events=run(tmp_path,replies)
    assert result.stop_reason=='no_progress'
    assert result.stats.state_changed_turns==0
    assert sum(e['type']=='no_change_nudge' for e in events)==1
    assert not list(tmp_path.iterdir())


def test_failed_command_mutation_counts_as_actual_progress(tmp_path):
    result,_=run(tmp_path,[openai_reply(calls=[('run_command',{'command':"python3 -c \"from pathlib import Path; Path('value.py').write_text('VALUE=1\\n'); raise SystemExit(1)\""})]),openai_reply(text='done')])
    assert result.stats.state_changed_turns==1
    assert result.stats.tool_errors==1
    assert (tmp_path/'value.py').read_text()=='VALUE=1\n'


def test_same_content_rewrite_does_not_reset_progress(tmp_path):
    (tmp_path/'a').write_text('same')
    result,_=run(tmp_path,[openai_reply(calls=[('run_command',{'command':'printf same > a'})]),openai_reply(text='done')])
    assert result.stats.state_changed_turns==0
    assert (tmp_path/'a').read_text()=='same'
