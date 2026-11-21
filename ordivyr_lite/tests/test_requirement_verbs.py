from ordivyr_lite.requirements import extract


def test_acceptance_ledger_keeps_yield_rejection_and_negative_obligations():
    prompt=('Implement a stream processor. Full batches contain exactly four values. '
            'When requested, yield nonempty trailing batches. Reject invalid bounds. '
            'Do not mutate the input. Stop scheduling after the first error. '
            'Wait for active jobs to settle. Propagate the original error.')
    texts=[r['text'] for r in extract(prompt)]
    assert len(texts)==8
    assert any('yield nonempty' in s for s in texts)
    assert any('Reject invalid' in s for s in texts)
    assert any('Do not mutate' in s for s in texts)
    assert any('contain exactly' in s for s in texts)
