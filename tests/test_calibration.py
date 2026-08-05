import copy
import tempfile

import torch

from models.dual2fair import Dual2Fair


def test_train_only_frequencies(dataset):
    assert sum(dataset.user_freq.values()) == len(dataset.train_data)
    assert sum(dataset.item_freq.values()) == len(dataset.train_data)
    assert len(dataset.interactions) > len(dataset.train_data)


def test_deterministic_full_user_and_item_calibration(dataset, config, backbone_factory):
    torch.manual_seed(7)
    first_backbone = backbone_factory('lightgcn')
    first = Dual2Fair(first_backbone, dataset, config, 'cpu', 'lightgcn')
    output = first.update_calibration_state()
    assert output.calibrated_user_representations.shape[0] == dataset.n_users
    assert output.calibrated_item_representations.shape[0] == dataset.n_items
    state = copy.deepcopy(first_backbone.state_dict())
    second_backbone = backbone_factory('lightgcn')
    second_backbone.load_state_dict(state)
    second = Dual2Fair(second_backbone, dataset, config, 'cpu', 'lightgcn')
    output2 = second.update_calibration_state()
    assert torch.allclose(output.calibrated_user_representations,
                          output2.calibrated_user_representations)
    assert torch.allclose(output.calibrated_item_representations,
                          output2.calibrated_item_representations)


def test_checkpoint_restores_calibration(dataset, config, backbone_factory):
    model = Dual2Fair(backbone_factory('lightgcn'), dataset, config, 'cpu', 'lightgcn')
    model.update_calibration_state()
    expected = model.compute_all_scores().detach().clone()
    with tempfile.NamedTemporaryFile(suffix='.pt') as handle:
        model.save_checkpoint(handle.name, epoch=3, global_step=9)
        restored = Dual2Fair(backbone_factory('lightgcn'), dataset, config, 'cpu', 'lightgcn')
        checkpoint = restored.load_checkpoint(handle.name)
    actual = restored.compute_all_scores().detach()
    assert checkpoint['epoch'] == 3
    assert torch.allclose(expected, actual, atol=1e-6)
    assert torch.allclose(actual, restored.compute_all_scores().detach(), atol=1e-6)
