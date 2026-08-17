from scripts.select_config import select_config


def candidate(l1, l2, ndcg, uif):
    return {'lambda1': l1, 'lambda2': l2,
            'mean_validation_ndcg': ndcg,
            'mean_validation_uif': uif}


def test_selects_lowest_uif_among_eligible():
    standard = {'NDCG': {'mean': 1.0}}
    result = select_config(standard, [candidate(.1, .1, .99, 2.0),
                                      candidate(.5, .1, .98, 1.0),
                                      candidate(.01, .1, .90, .1)], .98)
    assert result['selected']['lambda1'] == .5
    assert result['fallback_used'] is False


def test_fallback_selects_highest_ndcg_when_none_eligible():
    standard = {'NDCG': {'mean': 1.0}}
    result = select_config(standard, [candidate(.1, .1, .90, .1),
                                      candidate(.5, .1, .95, 2.0)], .98)
    assert result['selected']['lambda1'] == .5
    assert result['fallback_used'] is True


def test_ties_use_ndcg_then_lower_lambdas():
    standard = {'NDCG': {'mean': 1.0}}
    result = select_config(standard, [candidate(.5, .1, .99, 1.0),
                                      candidate(.1, .5, .99, 1.0),
                                      candidate(.1, .1, .98, 1.0)], .98)
    assert result['selected']['lambda1'] == .1
    assert result['selected']['lambda2'] == .5
