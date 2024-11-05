from typing import List

import torch
from torch.distributions import Categorical


def compute_entropy_loss(
    labels_to_consider: List[torch.FloatTensor],
    actual_lens: torch.LongTensor,
    token_log_ps: List[torch.FloatTensor],
    logits: List[torch.FloatTensor],
    approximate_entropy: bool = False,
    perform_length_normalization: bool = True,
) -> torch.Tensor:
    """Compute loss for per-step entropy."""

    batch_size = len(labels_to_consider)
    objective = 0.0
    for b_idx in range(batch_size):
        labels_per_example = labels_to_consider[b_idx]
        actual_lens_per_example = actual_lens[b_idx]
        if not perform_length_normalization:
            actual_lens_per_example = 1.0
        entropy_masks_per_example = torch.where(labels_per_example == -100, 0, 1)
        token_log_ps_per_example = token_log_ps[b_idx]
        if not approximate_entropy:
            distribution_per_example = Categorical(logits=logits[b_idx])
            entropy_per_example = distribution_per_example.entropy() * entropy_masks_per_example
            prefix_log_ps_per_example = torch.cumsum(token_log_ps_per_example, dim=0)
            # token log p will be multiplied by the prior log p.
            shifted_prefix_log_ps_per_example = torch.roll(prefix_log_ps_per_example, 1, dims=0)
            shifted_prefix_log_ps_per_example[0] = 1.0
            part_one = shifted_prefix_log_ps_per_example * entropy_per_example.detach()
            part_two = entropy_per_example
            objective += torch.sum(part_one + part_two, dim=0) / actual_lens_per_example
        else:
            objective += torch.sum(-token_log_ps_per_example * entropy_masks_per_example, dim=0) / actual_lens_per_example
    loss = -objective / batch_size
    return loss


def form_returns(rewards: List[List[torch.FloatTensor]]) -> List[List[torch.FloatTensor]]:
    """Compute returns based on any rewards."""
    returns = []
    for batch_idx in range(len(rewards)):
        sample_returns = []
        for sample_idx in range(len(rewards[batch_idx])):
            reward_sequence = rewards[batch_idx][sample_idx]
            cum_sum = torch.cumsum(reward_sequence, dim=0)
            full_sum = cum_sum[-1]
            sequence_returns = torch.cat((full_sum.reshape(1), full_sum - cum_sum[0:-1]), dim=0)
            sample_returns.append(sequence_returns.tolist())
        returns.append(sample_returns)
    return returns


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
