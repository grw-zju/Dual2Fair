import torch


class PopularityIPS:
    def __init__(self, alpha=0.5, max_weight=10.0, device=None):
        self.alpha = float(alpha)
        self.max_weight = float(max_weight)
        self.device = device or torch.device('cpu')

    def weighted_bpr_loss(self, backbone, user_ids, positive_items,
                          negative_items, item_frequency):
        positive_scores = backbone.forward(user_ids, positive_items)
        negative_scores = backbone.forward(user_ids, negative_items)
        frequencies = torch.as_tensor(
            [item_frequency.get(int(item), 0) for item in positive_items.detach().cpu()],
            dtype=positive_scores.dtype, device=positive_scores.device)
        propensity = (frequencies + 1.0).pow(self.alpha)
        weights = (propensity.mean() / propensity).clamp(max=self.max_weight)
        return -(weights * torch.nn.functional.logsigmoid(
            positive_scores - negative_scores)).mean()
