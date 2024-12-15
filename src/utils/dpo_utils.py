"""Module to implement different objectives related to preference optimization."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# Source: https://github.com/eric-mitchell/direct-preference-optimization
def preference_loss(
    policy_chosen_logps: torch.FloatTensor,
    policy_rejected_logps: torch.FloatTensor,
    beta: float,
    label_smoothing: float = 0.0,
    ipo: bool = False,
    reference_free: bool = False,
    reference_chosen_logps: Optional[torch.FloatTensor] = None,
    reference_rejected_logps: Optional[torch.FloatTensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the
            chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the
            rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the
            chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the
            rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the
            range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that
            preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and
            implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the
            chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    if reference_free:
        ref_logratios = 0
    else:
        assert reference_chosen_logps is not None
        assert reference_rejected_logps is not None
        ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1 / (2 * beta)) ** 2
        # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf;
        # Label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    # chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    # rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    # return losses, chosen_rewards, rejected_rewards
    return losses
