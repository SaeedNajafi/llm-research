import torch


def z_scoring(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the rewards using z-scoring."""
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    rewards_mean = torch.mean(rewards, dim=1, keepdim=True)
    rewards_std = torch.std(rewards, dim=1, keepdim=True)
    return (rewards - rewards_mean) / (rewards_std + 1e-12)


def normalize(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the rewards to be between [-1, 1]"""
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    rewards_max, _ = torch.max(rewards, dim=1, keepdim=True)
    rewards_min, _ = torch.min(rewards, dim=1, keepdim=True)
    return 2 * (rewards - rewards_min) / (rewards_max - rewards_min + 1e-12) - 1.0


def mml_normalize(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the rewards based on MML objective. MML
    objective tends to apply softmax over the log probability which is the
    reward function.

    The rewards should be scaled by the number of samples but that will
    be canceled out if we divide the loss by the number of samples per
    example.
    """
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    return torch.nn.functional.softmax(rewards, dim=1)


def rloo_normalize(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Leave-one-out normalization."""
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    sum_rewards = torch.sum(rewards, dim=1, keepdim=True)
    other_sums = sum_rewards - rewards
    other_baselines = other_sums / (num_samples - 1.0)
    return rewards - other_baselines
