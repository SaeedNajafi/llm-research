import pytest
import torch

from src.utils.general_utils import set_random_seed
from src.utils.rl_utils import form_returns, normalize_signals


@pytest.fixture(scope="session", autouse="True")
def fix_seed() -> None:
    seed = len("testing")
    set_random_seed(seed)


def test_form_returns() -> None:
    rewards = torch.FloatTensor([[[1, 2, 3], [0, 1, 1]]])
    returns = form_returns(rewards)
    batch_size, sample_size, seq_len = returns.size()
    assert batch_size == 1
    assert sample_size == 2
    assert seq_len == 3
    assert returns.tolist()[0][0] == [6, 5, 3]
    assert returns.tolist()[0][1] == [2, 2, 1]


def test_normalize_signals() -> None:
    signals = torch.FloatTensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    norm_signals = normalize_signals(signals, normalization_type="no_normalize")
    assert norm_signals.tolist() == signals.tolist()

    norm_signals = normalize_signals(signals, normalization_type="zscore")
    assert [round(each, 5) for each in norm_signals.tolist()[0]] == [-1.11803, 0.0, 1.11803]
    assert [round(each, 5) for each in norm_signals.tolist()[1]] == [-1.11803, 0.0, 1.11803]

    norm_signals = normalize_signals(signals, normalization_type="linear")
    assert [round(each, 5) for each in norm_signals.tolist()[0]] == [0.0, 0.5, 1.0]
    assert [round(each, 5) for each in norm_signals.tolist()[1]] == [0.0, 0.5, 1.0]
