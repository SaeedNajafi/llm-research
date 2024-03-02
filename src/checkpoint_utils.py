import os
import shutil
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

import torch
import torch.distributed as dist
from absl import flags, logging
from peft import PeftModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import Sampler

FLAGS = flags.FLAGS


def save_data(data: Any, path: str) -> None:
    """Save the object to the file."""
    logging.info(f"Saving to {path}")
    torch.save(data, f"{path}_temp")
    if os.path.exists(path):
        os.remove(path)
    os.replace(f"{path}_temp", path)


def model_save(
    model: torch.nn.Module,
    model_path: str,
    checkpoint_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
) -> None:
    """Save the modules to the model_path for the specified checkpoint name."""
    if model_path is None:
        raise Exception("model_path cannot be None!")

    if not os.path.exists(model_path):
        Path(model_path).mkdir(parents=True, exist_ok=True)

    full_path = os.path.join(model_path, checkpoint_name)

    model_full_path = f"{full_path}_model.bin"
    logging.info(f"Saving model to {model_full_path}")
    # This will call the PEFTmodel save.
    model.save_pretrained(f"{model_full_path}_temp")

    # According to the GNU spec of rename, the state of path
    # is atomic, i.e. it will either be modified or not modified, but not in
    # between, during a system crash (i.e. preemtion)
    if os.path.exists(model_full_path):
        shutil.rmtree(model_full_path)

    os.replace(f"{model_full_path}_temp", model_full_path)

    if optimizer is not None:
        save_data(optimizer.state_dict(), f"{full_path}_optimizer.bin")

    if scheduler is not None:
        save_data(scheduler, f"{full_path}_scheduler.bin")


def save_state(
    model: torch.nn.Module,
    checkpoint_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    sampler: Optional[Sampler] = None,
    dataloader_iter: Optional[Iterator] = None,
    model_path: Optional[str] = None,
) -> None:
    """Save the modules to the model_path for the specified checkpoint name.

    Save the extra modules for the training state. Save only for the
    rank 0 process.
    """
    if dist.get_rank() == 0:
        m_path = FLAGS.model_path
        if model_path is not None:
            m_path = model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)

        full_path = os.path.join(m_path, checkpoint_name)

        model_full_path = f"{full_path}_model"
        logging.info(f"(rank_{dist.get_rank()}) - Saving model to {model_full_path}")
        # This will call the PEFTmodel save.
        model.save_pretrained(f"{model_full_path}_temp")

        # according to the GNU spec of rename, the state of path
        # is atomic, i.e. it will either be modified or not modified, but not in
        # # between, during a system crash (i.e. preemtion)
        os.replace(f"{model_full_path}_temp", model_full_path)

        save_data(vars(FLAGS), f"{full_path}_flags")

        if optimizer is not None:
            save_data(optimizer.state_dict(), f"{full_path}_optimizer")
        if scheduler is not None:
            save_data(scheduler.state_dict(), f"{full_path}_scheduler")
        if sampler is not None and dataloader_iter is not None:
            save_data(sampler.state_dict(dataloader_iter), f"{full_path}_sampler")

        rng = torch.random.get_rng_state()
        save_data(rng, f"{full_path}_rng")

    # All processes should wait and join here before function exit.
    dist.barrier()


def load_data(model_object: Any, path: str) -> None:
    """Load the state dict into the model_object."""
    logging.info(f"Loading from {path}")
    model_object.load_state_dict(torch.load(path, map_location="cpu"))


def model_load(
    model: torch.nn.Module,
    model_path: str,
    checkpoint_name: str,
    peft_load: bool,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
) -> Tuple[torch.nn.Module, Optional[Optimizer], Optional[LRScheduler]]:
    """Load the model and sub-components."""
    full_path = os.path.join(model_path, checkpoint_name)

    logging.info(f"Loading from {full_path}_model")
    if peft_load:
        model = PeftModel.from_pretrained(model, f"{full_path}_model.bin")
    else:
        model = model.from_pretrained(f"{full_path}_model.bin")

    if optimizer is not None:
        load_data(optimizer, f"{full_path}_optimizer.bin")

    if scheduler is not None:
        logging.info(f"Loading from {full_path}_scheduler.bin")
        scheduler = torch.load(f"{full_path}_scheduler.bin", map_location="cpu")

    return model, optimizer, scheduler


def load_state(
    model: torch.nn.Module,
    checkpoint_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    sampler: Optional[Sampler] = None,
    model_path: Optional[str] = None,
) -> None:
    """Load the model and training state."""
    m_path = FLAGS.model_path
    if model_path is not None:
        m_path = model_path

    full_path = os.path.join(m_path, checkpoint_name)

    logging.info(f"(rank_{dist.get_rank()}) - Loading from {full_path}_model")
    PeftModel.from_pretrained(model, f"{full_path}_model")

    if optimizer is not None:
        load_data(optimizer, f"{full_path}_optimizer")

    if scheduler is not None:
        load_data(scheduler, f"{full_path}_scheduler")

    if sampler is not None:
        load_data(sampler, f"{full_path}_sampler")

    logging.info(f"(rank_{dist.get_rank()}) - Loading from {full_path}_rng")
    rng = torch.load("{full_path}_rng")
    torch.random.set_rng_state(rng)
