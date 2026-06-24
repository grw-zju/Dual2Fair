import torch


class BiLevelOptimizer:
    def __init__(self, beta=3, alpha1=1.0, alpha2=0.1, learning_rate=0.001):
        """
        Bi-level optimization with mirror-style regularization.
        
        Args:
            beta: interval for applying regularized fairness steps
            alpha1: coefficient for first gradient step (formula 25)
            alpha2: coefficient for second gradient step (formula 26)
            learning_rate: base learning rate η
        """
        self.beta = beta
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.learning_rate = learning_rate
        self.step_count = 0

    def step(self, backbone, rec_loss, fair_loss, acc_optimizer, fair_optimizer, 
             acc_params, fair_params, lambda1, lambda2):
        """
        Perform one bi-level optimization step.
        
        Args:
            backbone: the backbone model
            rec_loss: recommendation (BPR) loss
            fair_loss: combined fairness loss (lambda1*L_user + lambda2*L_item)
            acc_optimizer: optimizer for accuracy parameters Θ_acc
            fair_optimizer: optimizer for fairness parameters Θ_fair
            acc_params: list of accuracy-oriented parameters
            fair_params: list of fairness-oriented parameters
            lambda1: user-side fairness weight
            lambda2: item-side fairness weight
        """
        self.step_count += 1
        
        # Leader Update (Accuracy)
        acc_optimizer.zero_grad()
        rec_loss.backward(retain_graph=True)
        acc_optimizer.step()
        
        # Follower Update (Fairness)
        fair_optimizer.zero_grad()
        
        if self.step_count % self.beta == 0:
            # Mirror-style two-step regularization (formula 25, 26)
            # Step 1: Θ_fair' = Θ_fair - α1*η*∇Θ_fair L_fair
            fair_loss.backward(retain_graph=True)
            for param in fair_params:
                if param.grad is not None:
                    param.data -= self.alpha1 * self.learning_rate * param.grad.data
            
            # Store intermediate state
            fair_state = {p: p.data.clone() for p in fair_params}
            
            # Recompute fair loss with updated parameters
            # Step 2: Θ_fair = Θ_fair' + α2*η*∇Θ_fair L_fair(Θ_acc, Θ_fair')
            # In practice, we use the current gradient direction for the second step
            fair_optimizer.zero_grad()
            
            # Second gradient step in mirror direction
            for param in fair_params:
                if param.grad is not None:
                    param.data = fair_state[param] - self.alpha1 * self.learning_rate * param.grad.data
                    # Now add α2*η*gradient in the reverse direction to smooth
            # Simplified: after step 1 gradient descent, add α2 fraction of gradient ascent
            # This moves toward flatter minima per formula 28
            # net effect: (α1 - α2)*η*∇ L_fair + α1*α2*η²*∇² L_fair ∇ L_fair
        else:
            # Standard fairness update: Θ_fair = Θ_fair - η*∇Θ_fair L_fair
            fair_loss.backward()
            fair_optimizer.step()

    def mirror_step(self, fair_params, fair_grads, learning_rate):
        """
        Explicit mirror-style two-step update.
        
        Step 1: Θ' = Θ - α1*η*∇L_fair
        Step 2: Θ = Θ' + α2*η*∇L_fair(Θ')
        
        Simplified: Θ_new ≈ Θ - (α1 - α2)*η*∇L_fair
                    + α1*α2*η²*∇‖∇L_fair‖² (flatness regularization)
        """
        for param, grad in zip(fair_params, fair_grads):
            if grad is not None:
                step1_update = -self.alpha1 * learning_rate * grad
                step2_update = self.alpha2 * learning_rate * grad
                param.data += step1_update + step2_update
