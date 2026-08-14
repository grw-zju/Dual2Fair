from data.dataset_utils import load_dataset


def test_user_activity_groups_are_deterministic_and_complete():
    dataset = load_dataset('demo', min_ui=3, min_ii=1)
    higher, sparse = dataset.get_user_activity_groups(0.95)
    assert len(higher) + len(sparse) == dataset.n_users
    assert set(higher).isdisjoint(sparse)
    assert (higher, sparse) == dataset.get_user_activity_groups(0.95)
    assert min(dataset.user_freq[user] for user in higher) >= max(
        dataset.user_freq[user] for user in sparse)
