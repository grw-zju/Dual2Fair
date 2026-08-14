import numpy as np
import torch

from data.dataset_utils import load_dataset
from evaluation.evaluator import Evaluator
from evaluation.metrics import compute_rank_dif, rank_relevance_from_order


class MatrixModel:
    def __init__(self, scores):
        self.scores = torch.as_tensor(scores, dtype=torch.float32)

    def forward(self, users, items):
        return self.scores[users.cpu(), items.cpu()]


def test_rank_dif_invariant_to_monotonic_transforms():
    rng = np.random.RandomState(4)
    scores = rng.normal(size=(4, 7))

    def dif(values):
        exposure = np.zeros(7)
        relevance = np.zeros(7)
        evaluated = np.ones(7, dtype=bool)
        for row in values:
            order = np.lexsort((np.arange(7), -row))
            exposure[order[:3]] += 1
            relevance[order] += (7 - np.arange(1, 8) + 1) / 7
        return compute_rank_dif(exposure, relevance, evaluated)
    assert np.isclose(dif(scores), dif(3 * scores + 7))
    assert np.isclose(dif(scores), dif(np.exp(scores)))


def test_validation_and_test_candidate_masks():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    validation = Evaluator(dataset, split='val')
    test = Evaluator(dataset, split='test')
    user = sorted(dataset.get_val_dict())[0]
    val_candidates = set(validation.candidate_items(user))
    test_candidates = set(test.candidate_items(user))
    assert not (val_candidates & dataset.user_items[user])
    assert dataset.get_val_dict()[user] in val_candidates
    assert dataset.get_val_dict()[user] not in test_candidates
    assert dataset.get_test_dict()[user] in test_candidates


def test_blockwise_full_evaluator_runs_without_dense_score_matrix():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    scores = np.random.RandomState(2).normal(size=(dataset.n_users, dataset.n_items))
    evaluator = Evaluator(dataset, split='val', baseline_duf=.1, baseline_dif=.1)
    result = evaluator.evaluate(model=MatrixModel(scores))
    assert result['evaluation_protocol'] == 'full_warm_start_val'
    assert result['UIF'] is not None


def test_rank_relevance_helper_handles_item_ids():
    relevance = rank_relevance_from_order([5, 2, 7], n_items=8)
    assert np.isclose(relevance[5], 1.0)
    assert np.isclose(relevance[2], 2 / 3)
    assert np.isclose(relevance[7], 1 / 3)
    assert relevance[0] == 0.0
