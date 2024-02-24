import os
from typing import Any, Iterator, Optional

import torch
import torch.distributed as dist
from absl import flags, logging
from peft import PeftModel
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data.sampler import Sampler

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")


def save_data(data: Any, path: str) -> None:
    """Save the object to the file."""
    logging.DEBUG(f"Saving to {path}")
    torch.save(data, f"{path}_temp")
    os.replace(f"{path}_temp", path)


def save_state(
    model: torch.nn.Module,
    checkpoint_name: str,
    optimizer: Optional[Optimizer] = None,
    scheduler: Optional[LRScheduler] = None,
    sampler: Optional[Sampler] = None,
    dataloader_iter: Optional[Iterator] = None,
    rng: Optional[torch.Tensor] = None,
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
        logging.DEBUG(f"Saving model to {model_full_path}")
        # This will call the PEFTmodel save.
        model.save_pretrained(f"{model_full_path}_temp")

        # according to the GNU spec of rename, the state of checkpoint_path
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
        if rng is not None:
            save_data(rng, f"{full_path}_rng")

    # All processes should wait and join here before function exit.
    dist.barrier()


def load_model(model: torch.nn.Module, input_dir: str, weights_name: str) -> None:
    """Load the model's weights. Only the PEFT weights.

    Args:
    ----
        model: The model.
        input_dir: The checkpointing directory.
    """
    # configure map_location properly
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    # ddp_model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=map_location))
    input_model_file = os.path.join(input_dir, weights_name)
    print(f"Loading model from {input_model_file}")
    PeftModel.from_pretrained(model, input_model_file)
    print(f"Model loaded from {input_model_file}")
