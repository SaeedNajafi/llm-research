from typing import List

import torch
from torch.distributions import Categorical


def compute_entropy_loss(
    labels_to_consider: List[List[torch.FloatTensor]],
    actual_lens: torch.LongTensor,
    token_log_ps: List[List[torch.FloatTensor]],
    logits: List[List[torch.FloatTensor]],
) -> torch.Tensor:
    """Compute loss for per-step entropy."""
    batch_size = len(labels_to_consider)
    loss = 0.0
    for b_idx in range(batch_size):
        objective = 0.0
        sample_size = len(labels_to_consider[b_idx])
        for s_idx in range(sample_size):
            labels_per_sample = labels_to_consider[b_idx][s_idx]
            actual_lens_per_sample = actual_lens[b_idx, s_idx]
            entropy_masks_per_sample = torch.where(labels_per_sample == -100, 0, 1)
            distribution_per_sample = Categorical(logits=logits[b_idx][s_idx])
            entropy_per_sample = distribution_per_sample.entropy() * entropy_masks_per_sample
            token_log_ps_per_sample = token_log_ps[b_idx][s_idx]
            prefix_log_ps_per_sample = torch.cumsum(token_log_ps_per_sample, dim=0)
            part_one = prefix_log_ps_per_sample * entropy_per_sample.detach()
            part_two = entropy_per_sample
            objective += torch.sum(part_one + part_two, dim=0) / actual_lens_per_sample
        loss += -objective / sample_size
    return loss / batch_size


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
