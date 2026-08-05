def joint_weighted_sum_step(model, optimizer, recommendation_loss,
                            user_loss, item_loss, lambda1, lambda2,
                            clip_norm=None):
    optimizer.zero_grad()
    total = recommendation_loss + lambda1 * user_loss + lambda2 * item_loss
    total.backward()
    if clip_norm:
        import torch
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    model.clear_calibration_state()
    return total.detach()
