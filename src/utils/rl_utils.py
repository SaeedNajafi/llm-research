import torch


def compute_entropy(self, labels_to_consider: List[List[torch.FloatTensor]],
                    actual_lens: torch.LongTensor, logits: List[List[torch.FloatTensor]]) -> torch.Tensor:
    """Compute per-step entropy."""
    #     if FLAGS.compute_per_step_entropy:
    #         entropy_masks = torch.where(labels_to_consider == -100, 0, 1)
    #         distribution = Categorical(logits=logits)
    #         sequence_entropy = torch.sum(distribution.entropy() * entropy_masks, dim=1) / actual_lens
    #         sequence_entropy = sequence_entropy.view(batch_size, FLAGS.rl_sample_size)

    #     if FLAGS.compute_per_step_entropy:
    #         entropy_loss_part_one = -torch.mean(torch.mean(sequence_log_probs * sequence_entropy.detach(), dim=1), dim=0)
    #         entropy_loss_part_two = -torch.mean(torch.mean(sequence_entropy, dim=1), dim=0)
    #         entropy_loss = entropy_loss_part_one + entropy_loss_part_two
    #         loss += FLAGS.entropy_coef * entropy_loss


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
