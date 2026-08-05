def alternating_step(model, optimizer, recommendation_closure,
                     fairness_closure, clip_norm=None):
    import torch
    optimizer.zero_grad()
    recommendation_loss = recommendation_closure()
    recommendation_loss.backward()
    if clip_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    model.clear_calibration_state()

    optimizer.zero_grad()
    fairness_loss = fairness_closure()
    fairness_loss.backward()
    if clip_norm:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    optimizer.step()
    model.clear_calibration_state()
    return recommendation_loss.detach(), fairness_loss.detach()
