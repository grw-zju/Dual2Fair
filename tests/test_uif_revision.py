import numpy as np
import pytest
import torch
import yaml

from data.dataset_utils import load_dataset
from evaluation.evaluator import Evaluator
from evaluation.metrics import average_per_run_uif, compute_uif
from run import _make_evaluator


def test_split_specific_uif_references_are_required():
    scores = {0: .4, 1: .6}
    validation = compute_uif(scores, .2, baseline_duf=.01, baseline_dif=.2)
    test = compute_uif(scores, .2, baseline_duf=.02, baseline_dif=.4)
    assert validation != test
    with pytest.raises(ValueError):
        compute_uif(scores, .2, None, .2)


def test_average_per_run_uif_not_metric_then_formula():
    runs = [{'UIF': 1.0}, {'UIF': 3.0}]
    assert average_per_run_uif(runs) == 2.0


def test_evaluator_requires_split_references_when_configured():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    user_embeddings = np.random.RandomState(1).normal(size=(dataset.n_users, 4))
    item_embeddings = np.random.RandomState(2).normal(size=(dataset.n_items, 4))
    evaluator = Evaluator(dataset, split='val', require_uif_reference=True)
    with pytest.raises(ValueError, match='UIF reference constants'):
        evaluator.evaluate(user_embeddings=user_embeddings, item_embeddings=item_embeddings)


def test_default_config_marks_uif_references_required():
    with open('config/default.yaml') as handle:
        settings = yaml.safe_load(handle)
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    evaluator = _make_evaluator(dataset, settings, torch.device('cpu'), split='val')
    assert evaluator.require_uif_reference is True
    assert evaluator.baseline_duf is None
    assert evaluator.baseline_dif is None
