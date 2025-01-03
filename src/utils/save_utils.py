import json
import os
import re
from typing import Any, Dict, List, Tuple

import torch
import torch.distributed as dist
from absl import flags, logging
from torch import nn
from torch.distributed.fsdp import FullOptimStateDictConfig, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

FLAGS = flags.FLAGS
flags.DEFINE_string("checkpoint_folder", "./checkpoints/gemma2-1024-13", "a path for checkpoint.")
flags.DEFINE_string("weights_base_folder", "/model-weights", "a path for pre-trained weights.")


def checkpoint_exists(output_dir: str) -> bool:
    """Check if a checkpoint exists.

    Args:
    ----
        output_dir: The main saving directory.

    Returns:
    -------
        Returns whether a checkpoint exists.
    """
    if os.path.isdir(os.path.join(output_dir, "checkpoints")):
        return True
    return False


def save_metadata(
    out_dir: str,
    meta_dict: Dict[str, int | torch.Tensor],
) -> None:
    """Save training metadata.

    Args:
    ----
        out_dir: The directory to save to.
        meta_dict: The dictionary containing the meta data.
    """
    os.makedirs(out_dir, exist_ok=True)
    torch.save(meta_dict, os.path.join(out_dir, "meta_data.pkl"))


def save_flags(out_dir: str) -> None:
    """This function saves the absl flags into a file."""
    save_path = os.path.join(out_dir, "flagfile.txt")
    if os.path.exists(save_path):
        os.remove(save_path)
    FLAGS.append_flags_into_file(save_path)
    logging.info(f"training params are saved in {save_path}.")


def load_metadata(
    in_dir: str,
) -> Tuple[int, int]:
    """Load training metadata.

    Args:
    ----
        in_dir: The directory where the meta data is saved.

    Returns:
    -------
        A tuple containing the checkpointed step, epoch, and the processed
            training dataset ids.
    """
    save_path = os.path.join(in_dir, "meta_data.pkl")
    meta_dict = torch.load(save_path)
    checkpointed_step = meta_dict["step"]
    checkpointed_epoch = meta_dict["epoch"]
    return checkpointed_step, checkpointed_epoch


def get_latest_checkpoint_dir(folder_path: str) -> str:
    """Find the latest checkpoint directory using regex.

    Args:
    ----
        folder_path: The path to where checkpoints are saved.

    Returns:
    -------
        The subpath (i.e. two levels) of the latest checkpoint's directory.
    """
    epoch_pattern = re.compile(r"^epoch_(\d+)$")
    folder_pattern = re.compile(r"^checkpoint_(\d+)$")

    def _find_largest(pattern: re.Pattern, folder: str) -> str:
        max_integer = -1
        max_folder_name = ""

        for folder_name in os.listdir(folder):
            match = pattern.match(folder_name)
            if match:
                current_integer = int(match.group(1))
                if current_integer > max_integer:
                    max_integer = current_integer
                    max_folder_name = folder_name
        return max_folder_name

    epoch_folder = _find_largest(epoch_pattern, folder_path)
    if epoch_folder != "":
        folder_path = os.path.join(folder_path, epoch_folder)
    checkpoint_folder = _find_largest(folder_pattern, folder_path)
    if checkpoint_folder != "":
        return os.path.join(epoch_folder, checkpoint_folder)
    else:
        return epoch_folder


def save_peft_adapter(model: Any, output_path: str, distributed_strategy: str) -> None:
    """Save peft adapter to filesystem in a FSDP environment."""
    if distributed_strategy == "ddp":
        if dist.get_rank() == 0:
            model.module.save_pretrained(output_path)
    elif distributed_strategy == "fsdp":
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        ):
            if dist.get_rank() == 0:
                model.save_pretrained(output_path)


def save_model_and_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    output_dir: str,
    rank: int,
    include_model_state: bool = True,
) -> None:
    """Save model and optimizer states.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        output_dir: The checkpointing directory.
        rank: The worker's rank.
        include_model_state: Whether to include full model state dict.
            If using LoRA, set to False to saves only adapter optimizer state
            but not base model weights.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "model_optim.bin")
    if model.distributed_strategy == "ddp":
        if rank == 0:
            full_state = {"optim_state": optimizer.state_dict()}
            if include_model_state:
                full_state["model_state"] = model.model.module.state_dict()
            torch.save(full_state, save_path)
            logging.info(f"States saved to {save_path}.")

    elif model.distributed_strategy == "fsdp":
        with FSDP.state_dict_type(
            model.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            state_dict = model.model.state_dict()
            opt_state = FSDP.optim_state_dict(model.model, optimizer)
            full_state = {"optim_state": opt_state}
            if include_model_state:
                full_state["model_state"] = state_dict
            if rank == 0:
                torch.save(full_state, save_path)
                logging.info(f"States saved to {save_path}.")


def load_model_and_optimizer(
    optimizer: Optimizer,
    model: nn.Module,
    rank: int,
    input_dir: str,
    optimizer_only: bool = False,
    distributed_strategy: str = "ddp",
) -> None:
    """Load optimizer states and model weight, if found.

    Args:
    ----
        optimizer: The sharded optimizer.
        model: The sharded model.
        input_dir: The checkpointing directory.
        optimizer_only: If enabled, load only optimizer state dict but
            not model parameters. Useful for PEFT where base model
            does not change.
    """
    if dist.get_rank() == 0:
        logging.info(f"Loading states from {input_dir}.")

    if distributed_strategy == "ddp":
        input_dir = os.path.join(input_dir, "model_optim.bin")
        state_dict = torch.load(input_dir, weights_only=True, map_location=model.device)
        if not optimizer_only:
            model.module.load_state_dict(state_dict["model_state"])
        optimizer.load_state_dict(state_dict["optim_state"])

    elif distributed_strategy == "fsdp":
        input_dir = os.path.join(input_dir, "model_optim.bin")
        state_dict = torch.load(input_dir, weights_only=True, map_location="cpu")
        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False),
            FullOptimStateDictConfig(rank0_only=False),
        ):
            if not optimizer_only:
                model.load_state_dict(state_dict["model_state"])
            optim_state_dict = FSDP.optim_state_dict_to_load(model, optimizer, state_dict["optim_state"])
            optimizer.load_state_dict(optim_state_dict)

    if dist.get_rank() == 0:
        logging.info(f"States loaded from {input_dir}.")


def save_scheduler(
    scheduler: LRScheduler,
    output_dir: str,
    rank: int,
) -> None:
    """Save scheduler states.

    Args:
    ----
        scheduler: The LR scheduler.
        output_dir: The checkpointing directory.
        rank: The worker's rank.
    """

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        sched_name = "scheduler.bin"
        output_scheduler_file = os.path.join(output_dir, sched_name)
        logging.info(f"Saving scheduler state to {output_scheduler_file}.")
        state_dict = scheduler.state_dict()
        torch.save(state_dict, output_scheduler_file)
        logging.info(f"Scheduler state saved to {output_scheduler_file}.")


def load_scheduler(scheduler: LRScheduler, input_dir: str, rank: int, distributed_strategy: str, device: str) -> None:
    """Load scheduler states.

    Args:
    ----
        scheduler: The LR scheduler.
        input_dir: The checkpointing directory.
        rank: The worker's rank.
    """
    if distributed_strategy == "ddp":
        sched_name = "scheduler.bin"
        input_scheduler_file = os.path.join(input_dir, sched_name)
        if rank == 0:
            logging.info(f"Loading scheduler state from {input_scheduler_file}.")
        state_dict = torch.load(input_scheduler_file, weights_only=True, map_location=device)
        scheduler.load_state_dict(state_dict)
        if rank == 0:
            logging.info(f"Scheduler state loaded from {input_scheduler_file}.")

    elif distributed_strategy == "fsdp":
        sched_name = "scheduler.bin"
        input_scheduler_file = os.path.join(input_dir, sched_name)
        if rank == 0:
            logging.info(f"Loading scheduler state from {input_scheduler_file}.")
        state_dict = torch.load(input_scheduler_file, weights_only=True, map_location=device)
        scheduler.load_state_dict(state_dict)
        if rank == 0:
            logging.info(f"Scheduler state loaded from {input_scheduler_file}.")


def save_checkpoint(model: Any, step: int, epoch: int) -> None:
    """Save all states.

    Args:
    ----
        epoch: The current training epoch.
        step: The current training step.
    """
    rank = dist.get_rank()
    meta_dict = {
        "step": step,
        "epoch": epoch,
    }
    save_dir = os.path.join(
        FLAGS.checkpoint_folder,
        "checkpoints",
        f"epoch_{epoch}",
        f"checkpoint_{step}",
    )
    if rank == 0:
        # save metadata.
        save_metadata(save_dir, meta_dict)

        # save flags.
        save_flags(FLAGS.checkpoint_folder)

    # If peft is enabled, save only the peft adapters
    # and adapter optimizer state, but not base LLM weights.
    save_model_and_optimizer(
        model.optimizer,
        model,
        save_dir,
        rank,
        include_model_state=not FLAGS.use_peft,
    )

    if FLAGS.use_peft:
        save_peft_adapter(model.model, save_dir, distributed_strategy=model.distributed_strategy)

    save_scheduler(model.scheduler, save_dir, rank)

    dist.barrier()


def load_checkpoint(model: Any, checkpoint_dir: str) -> Tuple[int, int]:
    """Load all states.

    Args:
    ----
        checkpoint_dir: The directory under which all checkpoints are
            saved.

    Returns:
    -------
        The checkpointed epoch to be used by the outer loop.
        The checkpointed step to be used by the outer loop.
    """
    rank = dist.get_rank()
    step, epoch = load_metadata(checkpoint_dir)
    if FLAGS.use_peft:
        # peft adapters are not restored in this method.
        # These should have been restored before applying FSDP.
        assert model.is_peft_adapter_restored

    # Skip overwriting base model weights if peft is enabled.
    load_model_and_optimizer(model.optimizer, model.model, rank, checkpoint_dir, optimizer_only=model.is_peft_adapter_restored)
    load_scheduler(model.scheduler, checkpoint_dir, rank, distributed_strategy=model.distributed_strategy, device=model.device)
    dist.barrier()
    return step, epoch


def find_checkpoint(model: Any) -> Tuple[int, int]:
    """Find and load checkpoint if it exists.

    Args:
    ----
        checkpoint_dir: The checkpointing directory.

    Returns:
    -------
        The checkpointed epoch. If no checkpoint exists, it returns a
        default value of 0.

        The checkpointed step. If no checkpoint exists, it returns a
        default value of 0.
    """
    checkpoint = checkpoint_exists(FLAGS.checkpoint_folder)
    if checkpoint:
        main_ckpt_dir = os.path.join(FLAGS.checkpoint_folder, "checkpoints")
        latest_ckpt_dir = get_latest_checkpoint_dir(main_ckpt_dir)
        full_ckpt_dir = os.path.join(main_ckpt_dir, latest_ckpt_dir)
        logging.info(f"Checkpoint found at {full_ckpt_dir}.")
        checkpointed_step, checkpointed_epoch = load_checkpoint(model, full_ckpt_dir)
    else:
        checkpointed_epoch = 0
        checkpointed_step = 0
    return checkpointed_step, checkpointed_epoch


def save_to_json(
    output_filename: str,
    train_perplexities: List[float],
    train_losses: List[float],
    train_step_perplexities: List[float],
    train_step_losses: List[float],
    val_step_prediction_perplexities: List[float],
    val_step_prediction_losses: List[float],
    val_step_true_perplexities: List[float],
    val_step_true_losses: List[float],
    val_prediction_perplexities: List[float],
    val_prediction_losses: List[float],
    val_true_perplexities: List[float],
    val_true_losses: List[float],
    val_scores: List[Dict[str, float]],
) -> None:
    """Save some metrics to json."""
    metrics_data = {
        "train_perplexities": train_perplexities,
        "train_losses": train_losses,
        "train_step_perplexities": train_step_perplexities,
        "train_step_losses": train_step_losses,
        "val_step_prediction_perplexities": val_step_prediction_perplexities,
        "val_step_prediction_losses": val_step_prediction_losses,
        "val_step_true_perplexities": val_step_true_perplexities,
        "val_step_true_losses": val_step_true_losses,
        "val_prediction_perplexities": val_prediction_perplexities,
        "val_prediction_losses": val_prediction_losses,
        "val_true_perplexities": val_true_perplexities,
        "val_true_losses": val_true_losses,
        "val_scores": val_scores,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
