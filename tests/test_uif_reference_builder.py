import json

from scripts.build_uif_references import validate_and_build


def write_run(tmp_path, seed, val_duf, val_dif, test_duf, test_dif):
    path = tmp_path / f'seed_{seed}.json'
    payload = {
        'dataset': 'demo', 'backbone': 'lightgcn', 'method': 'standard',
        'model_seed': seed, 'split_hash': 'split', 'eval_mode': 'full',
        'validation_results': {'DUF': val_duf, 'DIF': val_dif, 'NDCG': .5},
        'DUF': test_duf, 'DIF': test_dif, 'NDCG': .4,
    }
    path.write_text(json.dumps(payload))
    return str(path)


def test_uif_reference_builder_averages_five_standard_runs(tmp_path):
    paths = [write_run(tmp_path, seed, seed * .01, seed * .02,
                       seed * .03, seed * .04)
             for seed in [42, 43, 44, 45, 46]]
    result = validate_and_build(paths, [42, 43, 44, 45, 46])
    assert result['dataset'] == 'demo'
    assert result['backbone'] == 'lightgcn'
    assert result['seeds'] == [42, 43, 44, 45, 46]
    assert result['val']['DUF'] == sum(seed * .01 for seed in [42, 43, 44, 45, 46]) / 5
    assert result['val']['DIF'] == sum(seed * .02 for seed in [42, 43, 44, 45, 46]) / 5
    assert result['test']['DUF'] == sum(seed * .03 for seed in [42, 43, 44, 45, 46]) / 5
    assert result['test']['DIF'] == sum(seed * .04 for seed in [42, 43, 44, 45, 46]) / 5
