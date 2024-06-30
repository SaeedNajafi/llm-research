"""Module for general utils."""

import datetime
import gc
import os
import random

import numpy
import torch
from absl import logging


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
