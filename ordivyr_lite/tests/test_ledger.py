from __future__ import annotations

import json

import pytest

from ordivyr_lite.ledger import Ledger, extract_constraints, eviction_order
from ordivyr_lite.loop import RunLimits, Runner
from ordivyr_lite.session import Session
from ordivyr_lite.tests.support import FakeTransport, make_router, openai_reply
from ordivyr_lite.tools import ToolBox

PROMPT = ("Implement csv stats.\n- must keep the public API stable\n- never touch tests/\n"
          "3. add a CLI entry point")


def run(responses, work, *, plan=False, scope=None, limits=None, session=None,
        resume_state=None, tmp_path=None):
    t = FakeTransport(responses)
    box = ToolBox(work, profile="workspace", plan=plan)
    r = Runner(make_router(t, tmp_path=tmp_path), box, PROMPT, limits=limits or RunLimits(),
               session=session, resume_state=resume_state, scope=scope)
    return r, t


@pytest.fixture
def work(tmp_path):
    (tmp_path / "big.txt").write_text("word " * 400)
    return tmp_path


def test_constraints_come_from_the_prompt_deterministically():
    got = extract_constraints(PROMPT)
    assert "must keep the public API stable" in got
    assert "never touch tests/" in got
    assert "add a CLI entry point" in got


def test_ledger_roundtrips_and_clamps_garbage():
    led = Ledger.from_prompt(PROMPT, scope=["src"])
    led.apply_plan(add="parse\ntest", done="1")
    again = Ledger.from_dict(json.loads(json.dumps(led.to_dict())))
    assert again.to_dict() == led.to_dict()
    assert Ledger.from_dict("nonsense").to_dict()["milestones"] == []


def test_scope_blocks_out_of_scope_writes_and_records_drift(work):
    responses = [openai_reply(calls=[("write_file", {"path": "src/ok.py", "content": "x=1\n"})]),
                 openai_reply(calls=[("write_file", {"path": "tests/bad.py", "content": "x=1\n"})]),
                 openai_reply(text="done")]
    r, _ = run(responses, work, scope=["src"])
    r.run()
    assert (work / "src" / "ok.py").exists()
    assert not (work / "tests" / "bad.py").exists()
    assert any("outside scope" in d for d in r.ledger.drift)


def test_dotfile_scope_is_not_mangled():
    assert Ledger(scope=[".github"]).allows(".github/workflows/ci.yml")
    assert not Ledger(scope=[".github"]).allows("github/x")


def test_compaction_checkpoint_carries_objective_constraints_and_plan(work):
    big = openai_reply(calls=[("read_file", {"path": "big.txt"})], prompt_tokens=500, completion_tokens=10)
    responses = ([openai_reply(calls=[("update_plan", {"add": "parse\nwire cli"})], prompt_tokens=500)]
                 + [big for _ in range(6)] + [openai_reply(text="done"), openai_reply(text="done")])
    r, t = run(responses, work, plan=True, limits=RunLimits(compact_at=100))
    res = r.run()
    assert res.stats.compactions >= 1
    ckpts = [m["content"] for m in r.messages
             if m.get("role") == "user" and str(m.get("content", "")).startswith("[checkpoint]")]
    assert ckpts, "compaction must leave a checkpoint"
    text = ckpts[-1]
    assert "never touch tests/" in text and "open milestones" in text and "Implement csv stats" in text


def test_plan_check_blocks_premature_done_once(work):
    responses = [openai_reply(calls=[("update_plan", {"add": "step one"})]),
                 openai_reply(text="all done"),
                 openai_reply(calls=[("update_plan", {"done": "1"})]),
                 openai_reply(text="really done")]
    r, t = run(responses, work, plan=True)
    res = r.run()
    assert res.stop_reason == "done" and res.text == "really done"
    assert any("[plan check]" in str(m.get("content")) for m in r.messages if m.get("role") == "user")


def test_plan_check_does_not_trap_a_model_that_ignores_it(work):
    responses = [openai_reply(calls=[("update_plan", {"add": "step one"})]),
                 openai_reply(text="all done"), openai_reply(text="still done")]
    r, _ = run(responses, work, plan=True)
    res = r.run()
    assert res.stop_reason == "done"       # one nudge, then the model's answer stands


def test_plan_tool_absent_by_default(work):
    r, t = run([openai_reply(text="ok")], work)
    r.run()
    names = [x["function"]["name"] for x in t.requests[0]["tools"]]
    assert "update_plan" not in names


def test_ledger_survives_resume_snapshot(work, tmp_path):
    sess = Session.create(tmp_path / "s.jsonl")
    responses = [openai_reply(calls=[("update_plan", {"add": "alpha\nbeta", "done": "1"})]),
                 openai_reply(text="done")]
    r, _ = run(responses, work, plan=True, scope=["src"], session=sess, limits=RunLimits(max_turns=1))
    r.run()
    state = sess.latest_snapshot()
    assert state is not None and state["ledger"]["scope"] == ["src"]
    assert [m["status"] for m in state["ledger"]["milestones"]] == ["done", "pending"]


def test_render_is_bounded_and_keeps_objective():
    led = Ledger.from_prompt("Objective X. " * 5)
    for i in range(30):
        led.note_verified("fact %d " % i + "y" * 150)
        led.note_failed("fail %d " % i + "z" * 150)
    led.apply_plan(add="\n".join(f"milestone {i}" for i in range(20)))
    out = led.render(budget=900)
    assert out.startswith("[ledger] objective: Objective X.")
    assert len(out) <= 900 + 1500


def test_eviction_order_names_only_known_sections():
    known = {"verified", "failed", "milestones_done", "milestones_open", "constraints", "scope"}
    assert set(eviction_order(1.5)) <= known


def test_resume_restores_ledger_and_restates_it_after_recovery(work, tmp_path):
    sess = Session.create(tmp_path / "s.jsonl")
    first = [openai_reply(calls=[("update_plan", {"add": "alpha\nbeta", "done": "1"})]),
             openai_reply(calls=[("read_file", {"path": "big.txt"})])]
    r1, _ = run(first, work, plan=True, scope=["src"], session=sess, limits=RunLimits(max_turns=2))
    r1.run()
    state = sess.latest_snapshot()
    # A fresh process: new Runner built only from the saved state.
    r2, t2 = run([openai_reply(calls=[("update_plan", {"done": "2"})]), openai_reply(text="done")],
                 work, plan=True, session=Session.create(tmp_path / "s2.jsonl"), resume_state=state)
    res = r2.run()
    assert res.stop_reason == "done"
    assert r2.ledger.scope == ["src"]
    assert [m["status"] for m in r2.ledger.milestones] == ["done", "done"]
    first_req = t2.requests[0]["messages"]
    assert first_req[-1]["role"] == "user" and "[resumed]" in first_req[-1]["content"]
    assert "open milestones" in first_req[-1]["content"]
