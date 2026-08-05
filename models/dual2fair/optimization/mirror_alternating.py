import torch


class MirrorAlternatingOptimizer:
    def __init__(self, alpha1=1.0, alpha2=0.1, interval=3,
                 learning_rate=0.001, clip_norm=None):
        if interval <= 0:
            raise ValueError('interval must be positive')
        self.alpha1 = float(alpha1)
        self.alpha2 = float(alpha2)
        self.interval = int(interval)
        self.learning_rate = float(learning_rate)
        self.clip_norm = clip_norm
        self.step_count = 0
        self.second_gradient_parameter_snapshot = None

    def _manual_update(self, parameters, scale):
        with torch.no_grad():
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=scale)

    def step(self, model, optimizer, recommendation_closure, fairness_closure):
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        if not parameters:
            raise ValueError('Optimization parameter group is empty')
        optimizer.zero_grad()
        recommendation_loss = recommendation_closure()
        recommendation_loss.backward()
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(parameters, self.clip_norm)
        optimizer.step()
        model.clear_calibration_state()
        self.step_count += 1

        optimizer.zero_grad()
        fairness_loss = fairness_closure()
        fairness_loss.backward()
        if self.clip_norm:
            torch.nn.utils.clip_grad_norm_(parameters, self.clip_norm)

        if self.step_count % self.interval == 0:
            self._manual_update(parameters, -self.alpha1 * self.learning_rate)
            model.clear_calibration_state()
            self.second_gradient_parameter_snapshot = [
                parameter.detach().clone() for parameter in parameters]
            optimizer.zero_grad()
            second_fairness_loss = fairness_closure()
            second_fairness_loss.backward()
            if self.clip_norm:
                torch.nn.utils.clip_grad_norm_(parameters, self.clip_norm)
            self._manual_update(parameters, self.alpha2 * self.learning_rate)
            optimizer.zero_grad()
            model.clear_calibration_state()
            return recommendation_loss.detach(), fairness_loss.detach(), second_fairness_loss.detach()

        self._manual_update(parameters, -self.learning_rate)
        optimizer.zero_grad()
        model.clear_calibration_state()
        return recommendation_loss.detach(), fairness_loss.detach(), None
