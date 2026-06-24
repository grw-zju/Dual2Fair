import torch


def frequency_tensor(item_freq, n_items, device):
    freq = torch.zeros(n_items, device=device)
    for iid, count in item_freq.items():
        if iid < n_items:
            freq[iid] = float(count)
    return freq


def split_hot_cold(item_freq, n_items, device, hot_ratio=0.2):
    freq = frequency_tensor(item_freq, n_items, device)
    sorted_indices = torch.argsort(freq, descending=True)
    n_hot = max(1, int(n_items * hot_ratio))
    return sorted_indices[:n_hot], sorted_indices[n_hot:], freq


def group_mean_gap(embeddings, group_a, group_b):
    if len(group_a) == 0 or len(group_b) == 0:
        return embeddings.new_tensor(0.0)
    a = embeddings[group_a]
    b = embeddings[group_b]
    return torch.norm(a.mean(dim=0) - b.mean(dim=0), p=2).pow(2)


def soft_exposure_disparity(user_embeddings, item_embeddings, item_freq, n_items, device,
                            user_limit=1024, hot_ratio=0.2):
    hot_items, cold_items, freq = split_hot_cold(item_freq, n_items, device, hot_ratio)
    if len(cold_items) == 0:
        return item_embeddings.new_tensor(0.0)

    scores = user_embeddings[:min(user_embeddings.shape[0], user_limit)] @ item_embeddings.T
    exposure = torch.softmax(scores, dim=1).mean(dim=0)
    quality = (freq + 1.0) / (freq.sum() + n_items)
    exposure_quality = exposure / quality.clamp_min(1e-8)

    hot_mean = exposure_quality[hot_items].mean()
    cold_mean = exposure_quality[cold_items].mean()
    return torch.abs(hot_mean - cold_mean)
