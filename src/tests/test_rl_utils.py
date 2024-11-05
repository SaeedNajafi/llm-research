import pytest
import torch

from src.utils.general_utils import set_random_seed
from src.utils.rl_utils import compute_entropy_loss


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
