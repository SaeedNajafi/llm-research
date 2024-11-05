import pytest
import torch

from src.utils.general_utils import set_random_seed
from src.utils.rl_utils import compute_entropy_loss, form_returns, normalize_signals


@pytest.fixture(scope="session", autouse="True")
def fix_seed() -> None:
    seed = len("testing")
    set_random_seed(seed)


def test_compute_entropy_loss() -> None:
    actual_lens = torch.FloatTensor([1, 2])
    logits = [
        torch.FloatTensor([[100, 100, 100, 100], [200, 200, 200, 200]]),
        torch.FloatTensor([[100, 100, 100, 100], [100, 100, 100, 100], [100, 100, 100, 100]]),
    ]
    token_log_ps = [torch.FloatTensor([0, -1]), torch.FloatTensor([0, -1, -2])]
    labels_to_consider = [torch.FloatTensor([-100, 1]), torch.FloatTensor([-100, 1, 2])]

    approximate_loss_no_norm = compute_entropy_loss(
        labels_to_consider=labels_to_consider,
        actual_lens=actual_lens,
        token_log_ps=token_log_ps,
        logits=logits,
        approximate_entropy=True,
        perform_length_normalization=False,
    )
    assert approximate_loss_no_norm.item() == -2.0

    approximate_loss_with_norm = compute_entropy_loss(
        labels_to_consider=labels_to_consider,
        actual_lens=actual_lens,
        token_log_ps=token_log_ps,
        logits=logits,
        approximate_entropy=True,
        perform_length_normalization=True,
    )
    assert approximate_loss_with_norm.item() == -1.25

    labels_to_consider = [torch.FloatTensor([-100, -100]), torch.FloatTensor([-100, 1, 2])]
    approximate_loss_with_norm_mask = compute_entropy_loss(
        labels_to_consider=labels_to_consider,
        actual_lens=actual_lens,
        token_log_ps=token_log_ps,
        logits=logits,
        approximate_entropy=True,
        perform_length_normalization=True,
    )
    assert approximate_loss_with_norm_mask.item() == -0.75

    loss_with_norm_mask = compute_entropy_loss(
        labels_to_consider=labels_to_consider,
        actual_lens=actual_lens,
        token_log_ps=token_log_ps,
        logits=logits,
        approximate_entropy=False,
        perform_length_normalization=True,
    )
    assert round(loss_with_norm_mask.item(), 5) == -0.34657


def test_form_returns() -> None:
    rewards = [torch.FloatTensor([1, 2, 3]), torch.FloatTensor([0, 1, 1])]
    returns = form_returns(rewards)
    assert len(returns) == 2
    assert returns[0] == [6, 5, 3]
    assert returns[1] == [2, 2, 1]


def test_normalize_signals() -> None:
    signals = [[1.0, 2.0, 3.0], [4.0, 5.0]]
    norm_signals = normalize_signals(signals, normalization_type="linear", terminal_reward_only=True)
    assert len(norm_signals) == 2
    assert norm_signals[0].tolist() == [1.0, 2.0, 0.0]
    assert [round(each, 5) for each in norm_signals[1].tolist()] == [4.0, 1.0]

    norm_signals = normalize_signals(signals, normalization_type="no_normalize", terminal_reward_only=True)
    assert norm_signals[0].tolist() == [1.0, 2.0, 3.0]
    assert norm_signals[1].tolist() == [4.0, 5.0]

    norm_signals = normalize_signals(signals, normalization_type="zscore", terminal_reward_only=True)
    assert [round(each, 5) for each in norm_signals[0].tolist()] == [1.0, 2.0, -0.70711]
    assert [round(each, 5) for each in norm_signals[1].tolist()] == [4.0, 0.70711]

    norm_signals = normalize_signals(signals, normalization_type="zscore", terminal_reward_only=False)
    assert [round(each, 5) for each in norm_signals[0].tolist()] == [-1.26491, -0.63246, 0.0]
    assert [round(each, 5) for each in norm_signals[1].tolist()] == [0.63246, 1.26491]

    norm_signals = normalize_signals(signals, normalization_type="no_normalize", terminal_reward_only=False)
    assert norm_signals[0].tolist() == [1.0, 2.0, 3.0]
    assert norm_signals[1].tolist() == [4.0, 5.0]
