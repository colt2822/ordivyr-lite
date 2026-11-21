from ordivyr_lite import board_adapter as A
from ordivyr_lite import board_contract as C


def job(**updates):
    return {'contract_version': '1.1', 'job_id': 'j-test', 'org': 'org', 'project': 'project',
            'agent': {'id': 'a', 'tools': ['read_file'], 'permissions': {'write': False}},
            'prompt': 'Read a file', 'model_ref': {'endpoint': 'local', 'model': 'test'}, **updates}


def test_adapter_keeps_exact_tools_and_usage(tmp_path):
    seen = []
    def run(spec, **kwargs):
        seen.append((spec, kwargs))
        return {'stop_reason': 'done', 'stats': {'model_calls': 2, 'input_tokens': 7,
            'cached_input_tokens': 4, 'output_tokens': 3, 'wall_ms': 1000},
            'completion': C.completion(model_said_done=True), 'text': 'ok'}
    result = A.execute(job(), tmp_path, {'model': 'test'}, run=run)
    assert not C.validate_result(result)
    assert seen[0][0]['permissions']['tools'] == ['read_file'] and seen[0][1]['exact_tools']
    assert result['usage']['turns'] == 2 and result['usage']['cache_read_tokens'] == 4
    assert result['usage']['prompt_tokens'] == 11
    assert result['completion']['independent_review'] is None


def test_adapter_refuses_shell_unknown_version_and_route_swap(tmp_path):
    def no_run(*args, **kwargs):
        raise AssertionError('must refuse before runtime')
    for bad_job, code in [(job(contract_version='2.0'), 'unsupported_version'),
            (job(agent={'id': 'a', 'tools': ['run_command'], 'permissions': {}}), 'tool_not_granted'),
            (job(model_ref={'endpoint': 'local', 'model': 'other'}), 'invalid_job')]:
        result = A.execute(bad_job, tmp_path, {'model': 'test'}, run=no_run)
        assert result['error']['code'] == code and not C.validate_result(result)


def test_capped_artifact_is_not_completed_job(tmp_path):
    result = A.execute(job(), tmp_path, {'model': 'test'}, run=lambda *a, **k: {
        'stop_reason': 'steps', 'completion': {'tests_passed': True}, 'stats': {}})
    assert result['status'] == 'failed' and result['stop'] == 'steps'
    assert not C.validate_result(result)
