"""Module for general utils."""

import datetime
import gc
import os
import random
from typing import Any, Dict

import numpy
import torch
from absl import logging
from torch.utils.data import Dataset


def get_date_of_run() -> str:
    """create date and time for file save uniqueness
    example: 2022-05-07-08:31:12_PM'
    """
    date_of_run = datetime.datetime.now().strftime("%Y-%m-%d-%I:%M:%S_%p")
    logging.info(f"--> current date and time of run = {date_of_run}")
    return date_of_run


def set_random_seed(seed: int) -> None:
    """Set the random seed, which initializes the random number generator.

    Ensures that runs are reproducible and eliminates differences due to
    randomness.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def clear_gpu_cache() -> None:
    """Clear the GPU cache for all ranks."""
    torch.cuda.empty_cache()
    gc.collect()


class DictDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text
    tasks.

    May need to return actual text instead of ids for easier processing
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
