import copy

import torch


class HierarchicalAlternatingOptimizer:
    def __init__(self, model, accuracy_learning_rate=1e-3,
                 fairness_learning_rate=5e-4, mirror_interval=3,
                 mirror_alpha1=1.0, mirror_alpha2=0.1,
                 weight_decay=0.0):
        self.model = model
        self.accuracy_optimizer = torch.optim.Adam(
            model.accuracy_parameters(), lr=accuracy_learning_rate,
            weight_decay=weight_decay)
        self.fairness_optimizer = torch.optim.Adam(
            model.fairness_correction_parameters(), lr=fairness_learning_rate)
        self.fairness_learning_rate = float(fairness_learning_rate)
        self.mirror_interval = int(mirror_interval)
        self.mirror_alpha1 = float(mirror_alpha1)
        self.mirror_alpha2 = float(mirror_alpha2)
        self.refresh_count = 0
        self.last_gradients = None

    def accuracy_step(self, loss_closure):
        self.accuracy_optimizer.zero_grad()
        loss = loss_closure()
        loss.backward()
        self.accuracy_optimizer.step()
        self.model.clear_scoring_state()
        return loss.detach()

    def _fairness_loss(self, lambda_user, lambda_item):
        return (lambda_user * self.model.compute_user_fixed_coupling_loss()
                + lambda_item * self.model.compute_item_fixed_coupling_loss())

    def _gradients(self, loss):
        parameters = self.model.fairness_correction_parameters()
        gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
        return parameters, [g if g is not None else torch.zeros_like(p)
                           for g, p in zip(gradients, parameters)]

    @staticmethod
    def _apply(parameters, gradients, scale):
        with torch.no_grad():
            for parameter, gradient in zip(parameters, gradients):
                parameter.add_(gradient, alpha=scale)

    def refresh_correction(self, lambda_user, lambda_item, strategy='hierarchical_mirror'):
        self.refresh_count += 1
        self.model.build_calibrated_embeddings()
        first_loss = self._fairness_loss(lambda_user, lambda_item)
        parameters = self.model.fairness_correction_parameters()
        use_mirror = (strategy == 'hierarchical_mirror'
                      and self.refresh_count % self.mirror_interval == 0)
        if use_mirror:
            _, first_gradients = self._gradients(first_loss)
            self._apply(parameters, first_gradients,
                        -self.mirror_alpha1 * self.fairness_learning_rate)
            self.model.build_calibrated_embeddings()
            second_loss = self._fairness_loss(lambda_user, lambda_item)
            _, second_gradients = self._gradients(second_loss)
            self._apply(parameters, second_gradients,
                        self.mirror_alpha2 * self.fairness_learning_rate)
            self.last_gradients = (tuple(g.detach().clone() for g in first_gradients),
                                   tuple(g.detach().clone() for g in second_gradients))
        else:
            self.fairness_optimizer.zero_grad()
            first_loss.backward()
            first_gradients = tuple(
                (parameter.grad.detach().clone() if parameter.grad is not None
                 else torch.zeros_like(parameter))
                for parameter in parameters)
            self.fairness_optimizer.step()
            self.last_gradients = (first_gradients, None)
        self.model.clear_scoring_state()
        return first_loss.detach()

    def step_iteration(self, loss_closure):
        loss = self.accuracy_step(loss_closure)
        self.model.update_ema()
        return loss

    def state_dict(self):
        return {
            'accuracy_optimizer': self.accuracy_optimizer.state_dict(),
            'fairness_optimizer': self.fairness_optimizer.state_dict(),
            'refresh_count': self.refresh_count}

    def load_state_dict(self, state):
        self.accuracy_optimizer.load_state_dict(state['accuracy_optimizer'])
        self.fairness_optimizer.load_state_dict(state['fairness_optimizer'])
        self.refresh_count = state['refresh_count']
