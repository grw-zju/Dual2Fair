import torch

from models.dual2fair.optimization import MirrorAlternatingOptimizer


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    def clear_calibration_state(self):
        pass


def test_independent_seed_initialization():
    states = []
    for seed in (42, 43, 44, 45, 46):
        torch.manual_seed(seed)
        states.append(torch.nn.Linear(3, 2).weight.detach().clone())
    assert all(not torch.equal(states[0], state) for state in states[1:])


def test_mirror_second_gradient_at_intermediate_parameters():
    model = ToyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    mirror = MirrorAlternatingOptimizer(alpha1=1.0, alpha2=0.1,
                                        interval=1, learning_rate=0.1)
    initial = model.weight.detach().clone()
    mirror.step(model, optimizer,
                lambda: (model.weight * 0.0).sum(),
                lambda: (model.weight ** 2).sum())
    snapshot = mirror.second_gradient_parameter_snapshot[0]
    assert not torch.allclose(snapshot, initial)
    assert torch.allclose(snapshot, torch.tensor([0.8]), atol=1e-6)
