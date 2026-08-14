import torch

from models.dual2fair.hierarchical_opt import HierarchicalAlternatingOptimizer


class ToyCalibration(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Parameter(torch.tensor(1.0))
        self.W_u_c = torch.nn.Parameter(torch.tensor(1.0))
        self.W_u_h = torch.nn.Parameter(torch.tensor(1.0))
        self.W_v = torch.nn.Parameter(torch.tensor(1.0))
        self.P_u = torch.nn.Parameter(torch.tensor(1.0))
        self.P_v = torch.nn.Parameter(torch.tensor(1.0))
        self.rho_u_tilde = torch.nn.Parameter(torch.tensor(-2.0))
        self.rho_v_tilde = torch.nn.Parameter(torch.tensor(-2.0))
        self.ema_state = {name: parameter.detach().clone()
                          for name, parameter in self.named_parameters()}
        self.ema_decay = .9

    def accuracy_parameters(self):
        return list(self.parameters())

    def fairness_correction_parameters(self):
        return [self.W_u_c, self.W_u_h, self.W_v]

    def compute_user_fixed_coupling_loss(self):
        return self.W_u_c.pow(4) + self.W_u_h.pow(2)

    def compute_item_fixed_coupling_loss(self):
        return self.W_v.pow(4)

    def build_calibrated_embeddings(self):
        pass

    def clear_scoring_state(self):
        pass

    def update_ema(self):
        with torch.no_grad():
            for name, parameter in self.named_parameters():
                self.ema_state[name].mul_(self.ema_decay).add_(parameter, alpha=.1)


def test_fairness_parameter_restriction_and_mirror_recompute():
    model = ToyCalibration()
    optimizer = HierarchicalAlternatingOptimizer(
        model, fairness_learning_rate=.01, mirror_interval=1)
    frozen_before = [model.backbone.clone(), model.P_u.clone(), model.P_v.clone(),
                     model.rho_u_tilde.clone(), model.rho_v_tilde.clone()]
    optimizer.refresh_correction(.5, .5)
    frozen_after = [model.backbone, model.P_u, model.P_v,
                    model.rho_u_tilde, model.rho_v_tilde]
    assert all(torch.allclose(before, after)
               for before, after in zip(frozen_before, frozen_after))
    first, second = optimizer.last_gradients
    assert any(not torch.allclose(left, right)
               for left, right in zip(first, second))


def test_ema_equation():
    model = ToyCalibration()
    old = {name: value.clone() for name, value in model.ema_state.items()}
    with torch.no_grad():
        model.backbone.add_(2.0)
    model.update_ema()
    assert torch.allclose(model.ema_state['backbone'],
                          .9 * old['backbone'] + .1 * model.backbone)


def test_legacy_optimization_package_imports_current_optimizer():
    from models.dual2fair.optimization import HierarchicalAlternatingOptimizer as Compat
    assert Compat is HierarchicalAlternatingOptimizer
