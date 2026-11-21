"""Board v1.1 job/event/result adapter over Lite's existing execution path.

board_contract.py is vendored unchanged from Board, stdlib only. Installing Lite
does not install Board. Model URLs/keys are supplied by the worker, never a job.
This first adapter supports a personal worker's bounded file tools and tests.
"""
from __future__ import annotations

from pathlib import Path

from . import board_contract as C
from .job import run_job
from . import JOB_PROTOCOL, PROTOCOL_VERSION, __version__

TOOLS = frozenset({'read_file', 'list_files', 'grep', 'find_files', 'edit_file', 'write_file'})


def execute(job, workdir, route, *, session_path=None, on_event=None, cancel_event=None,
            cfg=None, allow_paid=False, run=run_job):
    jid = str(job.get('job_id') or '') if isinstance(job, dict) else ''
    def failed(code, text):
        return C.make_result(jid, 'failed', 'error', error=(code, text))
    if not isinstance(job, dict) or not C.version_ok(job.get('contract_version')):
        return failed('unsupported_version', 'Lite supports Board contract major 1')
    bad = C.validate_job(job)
    if bad:
        return failed('invalid_job', '; '.join(bad))
    requested = (job.get('agent') or {}).get('tools')
    if requested is not None and not set(requested) <= TOOLS:
        return failed('tool_not_granted', 'personal Lite adapter supports file tools only')
    if job.get('validators'):
        return failed('invalid_job', 'external validator commands require a qualified adapter; use Lite auto tests')
    if job.get('workspace'):
        return failed('invalid_job', 'this personal adapter starts a generated-artifact workspace; named repositories are not qualified')
    allowed = sorted(t for t in TOOLS if C.tool_allowed(job, t))
    lim = job.get('limits') or {}
    limits = {new: lim[old] for old, new in {
        'max_turns': 'max_turns', 'max_tool_calls': 'max_tool_calls',
        'max_seconds': 'max_seconds', 'budget_usd': 'max_cost_usd',
        'max_output_tokens': 'output_tokens', 'token_budget': 'max_tokens'}.items() if old in lim}
    # Model identity is bound by the Board worker. Direct users must provide the
    # same model; an external job cannot swap credentials to a different model.
    if route.get('model') != (job.get('model_ref') or {}).get('model'):
        return failed('invalid_job', 'worker route does not match model_ref.model')
    spec = {'protocol': JOB_PROTOCOL, 'version': PROTOCOL_VERSION, 'job_id': jid,
            'task': {'prompt': job['prompt'], 'workdir': str(Path(workdir).resolve()),
                     'system': (job.get('agent') or {}).get('system') or None},
            'model': dict(route), 'limits': limits,
            'permissions': {'profile': 'workspace' if any(t in C.WRITE_TOOLS for t in allowed) else 'read-only',
                            'tools': allowed, 'sandbox': 'auto', 'network': False,
                            'allow_shell': False, 'allow_web': False},
            'validation': {'mode': 'auto'},
            'session': {'enabled': session_path is not None,
                        **({'path': str(session_path)} if session_path is not None else {})}}
    # No automatic resume here: the Board worker holds interrupted attempts.
    # Standalone Lite's resume command remains available after inspection.
    try:
        result = run(spec, cfg=cfg, allow_paid=allow_paid, on_event=on_event,
                     cancel_event=cancel_event, exact_tools=True)
    except Exception as exc:
        return failed('internal', type(exc).__name__ + ': ' + str(exc)[:500])
    stats = result.get('stats') or {}
    completion = result.get('completion') or {}
    raw_stop = result.get('stop_reason')
    stop = raw_stop if raw_stop in C.STOP_REASONS else ('budget' if raw_stop == 'cost' else 'error')
    status = 'cancelled' if stop == 'cancelled' else 'completed' if stop == 'done' else 'failed'
    comp = C.completion(**{k: completion.get(k) for k in C.COMPLETION_LEVELS})
    usage = {'prompt_tokens': (stats.get('input_tokens', 0) + stats.get('cached_input_tokens', 0)
                               + stats.get('cache_write_tokens', 0)),
             'completion_tokens': stats.get('output_tokens', 0),
             'cache_read_tokens': stats.get('cached_input_tokens', 0),
             'cache_write_tokens': stats.get('cache_write_tokens', 0),
             'turns': stats.get('model_calls', 0), 'tool_calls': stats.get('tool_calls', 0),
             'cost_usd': stats.get('cost_usd', 0.0), 'wall_seconds': stats.get('wall_ms', 0.0) / 1000}
    evidence = [{'name': 'workspace_validation', 'ok': comp['tests_passed'],
                 'detail': completion.get('validation') or {}, 'independent': False}]
    return C.make_result(jid, status, stop, result.get('text', ''), usage,
        error=('cancelled', 'job cancelled') if status == 'cancelled' else
              ('validation_failed', result.get('error') or raw_stop) if status == 'failed' else None,
        artifacts=result.get('artifacts') or [], evidence=evidence, comp=comp,
        runtime={'name': 'lite', 'version': __version__, 'model': route['model']})
