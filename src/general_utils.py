"""Module for general utils."""

from typing import Any, Dict

import torch
from absl import flags
from torch.utils.data import Dataset

FLAGS = flags.FLAGS


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


class DictDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text
    tasks.

    May need to return actual text instead of ids for better processing
    in the code.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """Store the reference to the tokenized data."""
        self.data = data
        self.keys = list(self.data.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values or just strings."""
        ret = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str) or isinstance(val[idx], torch.Tensor):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data[self.keys[0]])
