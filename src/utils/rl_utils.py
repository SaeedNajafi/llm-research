import torch


def z_scoring(signal: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the signal using z-scoring."""
    signal_mean = torch.mean(signal)
    signal_std = torch.std(signal)
    return (signal - signal_mean) / (signal_std + 1e-12)


def normalize(signal: torch.FloatTensor) -> torch.FloatTensor:
    """Perform normalization of the signal to be between [-1, 1]"""
    signal_max = torch.max(signal)
    signal_min = torch.min(signal)
    return 2 * (signal - signal_min) / (signal_max - signal_min + 1e-12) - 1.0


def rloo_normalize(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Leave-one-out normalization."""
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    sum_rewards = torch.sum(rewards, dim=1, keepdim=True)
    other_sums = sum_rewards - rewards
    other_baselines = other_sums / (num_samples - 1.0)
    return rewards - other_baselines
