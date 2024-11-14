import torch
from torch.distributions import Categorical


def compute_entropy_loss(
    labels_to_consider: torch.FloatTensor,
    actual_lens: torch.LongTensor,
    token_log_ps: torch.FloatTensor,
    logits: torch.FloatTensor,
    approximate_entropy: bool = False,
    perform_length_normalization: bool = True,
) -> torch.Tensor:
    """Compute loss based on per-step entropy."""
    if not perform_length_normalization:
        actual_lens = torch.ones_like(actual_lens)
    entropy_masks = torch.where(labels_to_consider == -100, 0, 1)
    if not approximate_entropy:
        distributions = Categorical(logits=logits)
        entropies = distributions.entropy() * entropy_masks
        prefix_log_ps = torch.cumsum(token_log_ps, dim=2)
        shifted_prefix_log_ps = torch.roll(prefix_log_ps, 1, dims=2)
        # batch_size, sample_size, seq_len
        shifted_prefix_log_ps[:, :, 0] = 1.0
        part_one = shifted_prefix_log_ps * entropies.detach()
        part_two = entropies
        objective = torch.sum(part_one + part_two, dim=2) / actual_lens
    else:
        objective = torch.sum(-token_log_ps * entropy_masks, dim=2) / actual_lens
    return -objective.mean()


def form_returns(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Compute returns based on any rewards."""
    # Reverse on sequence dimension.
    rewards_flipped = torch.flip(rewards, dims=(2,))

    # Sum the reversed rewards sequence.
    reversed_cum_sum = torch.cumsum(rewards_flipped, dim=2)

    # Reverse again to form the returns G_t.
    return torch.flip(reversed_cum_sum, dims=(2,))


def normalize_signals(signals: torch.FloatTensor, normalization_type: str) -> torch.FloatTensor:
    """Zscore or normalize between [0, 1]."""
    if normalization_type == "no_normalize":
        return signals
    elif normalization_type == "zscore":
        mean_s = signals.mean()
        std_s = signals.std()
        return (signals - mean_s) / (std_s + 1e-8)
    elif normalization_type == "linear":
        max_s = signals.max()
        min_s = signals.min()
        return (signals - min_s) / (max_s - min_s + 1e-8)


def rloo_normalize(rewards: torch.FloatTensor) -> torch.FloatTensor:
    """Leave-one-out normalization."""
    batch_size, num_samples = rewards.size()
    assert num_samples > 1
    sum_rewards = torch.sum(rewards, dim=1, keepdim=True)
    other_sums = sum_rewards - rewards
    other_baselines = other_sums / (num_samples - 1.0)
    return rewards - other_baselines
