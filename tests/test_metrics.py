import numpy as np

from evaluation.metrics import compute_dif_affine_invariant, compute_uif


def test_dif_positive_affine_invariance():
    scores = np.array([[0.1, 0.4, 0.2], [0.3, 0.2, 0.9]])
    test = {0: 1, 1: 2}
    train = {0: set(), 1: set()}
    first = compute_dif_affine_invariant(scores, test, train, 2)
    second = compute_dif_affine_invariant(7.0 * scores + 13.0, test, train, 2)
    assert np.isclose(first, second)


def test_standard_uif_normalization():
    ndcg = {0: 0.4, 1: 0.6}
    duf = np.var(list(ndcg.values()))
    dif = 0.2
    value = compute_uif(ndcg, dif, 0.5, 0.5, duf, dif, require_baseline=True)
    assert np.isclose(value, 1.0 / 0.5)


def test_uif_weights_change_result():
    ndcg = {0: 0.2, 1: 0.8}
    first = compute_uif(ndcg, 0.3, 0.25, 0.75, 0.1, 0.2)
    second = compute_uif(ndcg, 0.3, 0.75, 0.25, 0.1, 0.2)
    assert not np.isclose(first, second)
