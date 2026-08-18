import numpy as np

from scripts.run_multi_seed import aggregate_runs


def uif_from_means(runs):
    ndcg = np.mean([run['NDCG'] for run in runs])
    duf = np.mean([run['DUF'] for run in runs])
    dif = np.mean([run['DIF'] for run in runs])
    return (0.5 * duf / 0.1 + 0.5 * dif / 0.2) / ndcg


def test_run_level_uif_values_are_averaged_directly():
    runs = []
    for seed, ndcg, duf, dif in zip(
            [42, 43, 44, 45, 46],
            [0.2, 0.3, 0.4, 0.5, 0.6],
            [0.05, 0.04, 0.03, 0.02, 0.01],
            [0.03, 0.04, 0.05, 0.06, 0.07]):
        uif = (0.5 * duf / 0.1 + 0.5 * dif / 0.2) / ndcg
        runs.append({'model_seed': seed, 'split_hash': 'x', 'dataset': 'demo',
                     'backbone': 'lightgcn', 'method': 'dual2fair', 'eval_mode': 'full',
                     'NDCG': ndcg, 'Hit': 1.0, 'DUF': duf, 'DIF': dif, 'UIF': uif})
    aggregate = aggregate_runs(runs)
    assert aggregate['UIF']['mean'] == np.mean([run['UIF'] for run in runs])
    assert aggregate['run_level_metrics']['UIF'] == [run['UIF'] for run in runs]
    assert not np.isclose(aggregate['UIF']['mean'], uif_from_means(runs))
