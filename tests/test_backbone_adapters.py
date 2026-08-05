import torch

from models.dual2fair import Dual2Fair


def test_neumf_disabled_calibration_preserves_scorer(dataset, config, backbone_factory):
    backbone = backbone_factory('neumf')
    config['dual2fair']['rho_u'] = 1.0
    config['dual2fair']['rho_v'] = 1.0
    model = Dual2Fair(backbone, dataset, config, 'cpu', 'neumf')
    model.update_calibration_state()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    assert torch.allclose(model(users, items), backbone(users, items), atol=1e-6)


def test_vaecf_decoder_equivalence_and_gradient(dataset, config, backbone_factory):
    backbone = backbone_factory('vaecf')
    rows = torch.from_numpy(dataset.interact_mat[:4].toarray()).float()
    before = backbone.get_item_embeddings().detach().clone()
    loss = backbone.train_batch(rows, torch.device('cpu'))
    loss.backward()
    assert backbone.get_item_embeddings().grad is not None
    optimizer = torch.optim.SGD(backbone.parameters(), lr=0.01)
    optimizer.step()
    assert not torch.allclose(before, backbone.get_item_embeddings())
    backbone.eval()
    all_scores = backbone.compute_all_scores()
    users = torch.tensor([0, 1])
    items = torch.tensor([0, 1])
    assert torch.allclose(backbone(users, items), all_scores[users, items], atol=1e-6)


def test_training_evaluation_scorer_consistency(dataset, config, backbone_factory):
    backbone = backbone_factory('lightgcn')
    model = Dual2Fair(backbone, dataset, config, 'cpu', 'lightgcn')
    model.update_calibration_state()
    users = torch.tensor([0, 1, 2])
    items = torch.tensor([0, 1, 2])
    scores = model.compute_all_scores()
    assert torch.allclose(model(users, items), scores[users, items], atol=1e-6)
