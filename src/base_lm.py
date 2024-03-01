"""This a base class that implements some common functions for models."""

from typing import Dict, List

import torch
from absl import flags

from src.checkpoint_utils import model_load, model_save
from src.model_utils import clear_cache, optimizer_to, set_random_seed

FLAGS = flags.FLAGS


class BaseLM(torch.nn.Module):
    """Parent class of all child LMs."""

    def __init__(self, device: str, model_name: str) -> None:
        super().__init__()

        if not torch.cuda.is_available():
            raise Exception("CUDA is not available. Code has been tested for cuda GPUs.")

        set_random_seed(FLAGS.seed)
        self.device = device
        self.model_name = model_name

        # for some subclasses, we will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        self.model: torch.nn.Module = None
        self.optimizer: torch.optim.Optimizer = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler = None

    def save_to_checkpoint(self, model_path: str, checkpoint_name: str) -> None:
        """Save the model components to the disk."""
        model_save(
            model=self.model,
            model_path=model_path,
            checkpoint_name=f"_{self.model_name}_{checkpoint_name}",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def load_from_checkpoint(self, model_path: str, checkpoint_name: str, peft_load: bool = False) -> None:
        """Load the model components from the disk."""
        model, optimizer, scheduler = model_load(
            model=self.model,
            model_path=model_path,
            checkpoint_name=f"_{self.model_name}_{checkpoint_name}",
            peft_load=peft_load,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def to_device(self) -> None:
        """Move the required modules to the gpu on the given device."""
        self.model.to(self.device)
        self.loss_func.to(self.device)
        optimizer_to(self.optimizer, self.device)

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""
        clear_cache()
        self.model.eval()

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode!"""
        clear_cache()
        self.model.train()

    def data_to_device(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Move the batch tensors specified by keys into the gpu and return a
        dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}
