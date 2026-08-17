import pytest

from scripts.run_five_seeds import aggregate_runs


def test_five_seed_aggregation_requires_matching_metadata():
    runs = [{'model_seed': seed, 'split_hash': 'x', 'dataset': 'demo',
             'backbone': 'lightgcn', 'method': 'dual2fair', 'eval_mode': 'full',
             'NDCG': .5, 'Hit': .7, 'DUF': .1, 'DIF': .2, 'UIF': 1.2}
            for seed in (42, 43, 44, 45, 46)]
    aggregate = aggregate_runs(runs)
    assert aggregate['NDCG']['mean'] == .5
    with pytest.raises(ValueError):
        aggregate_runs(runs[:-1])
    broken = [dict(run) for run in runs]
    broken[-1]['split_hash'] = 'other'
    with pytest.raises(ValueError):
        aggregate_runs(broken)
    broken = [dict(run) for run in runs]
    broken[-1]['UIF'] = None
    with pytest.raises(ValueError):
        aggregate_runs(broken)


def test_standard_reference_runs_may_have_null_uif():
    runs = [{'model_seed': seed, 'split_hash': 'x', 'dataset': 'demo',
             'backbone': 'lightgcn', 'method': 'standard', 'eval_mode': 'full',
             'NDCG': .5, 'Hit': .7, 'DUF': .1, 'DIF': .2, 'UIF': None}
            for seed in (42, 43, 44, 45, 46)]
    aggregate = aggregate_runs(runs)
    assert aggregate['UIF'] is None
    assert aggregate['run_level_metrics']['UIF'] == [None] * 5
