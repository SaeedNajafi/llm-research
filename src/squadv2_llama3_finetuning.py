# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# Modified based on my applications in squad.

import os
from typing import Any

import fire
import torch

from src.configs import FsdpConfig, LoraConfig, TrainConfig, setup_wandb, update_config
from src.data_utility import create_squadv2_dataloader
from src.llama3 import LlamaQA
from src.metrics import qa_metric_squadv2_metrics
from src.model_utils import set_random_seed
from utils.train_utils import clear_gpu_cache, setup, setup_environ_flags, train


def main(**kwargs: Any) -> None:
    # Update the configuration for the training and sharding process
    train_config, fsdp_config, lora_config = TrainConfig(), FsdpConfig(), LoraConfig()
    update_config((train_config, fsdp_config, lora_config), **kwargs)

    # Set the seeds for reproducibility
    set_random_seed(train_config.seed)

    setup()
    # torchrun specific
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    if torch.distributed.is_initialized():
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        clear_gpu_cache()
        setup_environ_flags(rank)

    wandb_run = None

    if train_config.use_wandb:
        if rank == 0:
            wandb_run = setup_wandb(train_config, fsdp_config, lora_config, **kwargs)

    # Initialize the model here.
    model = LlamaQA(train_config, fsdp_config, lora_config, local_rank, rank)

    if wandb_run:
        if train_config.use_peft:
            wandb_run.config.update(model.peft_config)

    train_dataloader = create_squadv2_dataloader(
        model,
        train_config.train_file_name,
        fold_name="train",
        experiment_type=train_config.experiment_type,
        batch_size=train_config.batch_size_training,
        world_size=world_size,
        rank=rank,
    )

    eval_dataloader = create_squadv2_dataloader(
        model,
        train_config.dev_file_name,
        fold_name="dev",
        experiment_type=train_config.experiment_type,
        batch_size=train_config.val_batch_size,
        world_size=world_size,
        rank=rank,
    )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        train_config,
        fsdp_config,
        rank,
        world_size,
        wandb_run,
        qa_metric_squadv2_metrics,
    )
    if rank == 0:
        [print(f"Key: {k}, Value: {v}") for k, v in results.items()]
        if train_config.use_wandb:
            for k, v in results.items():
                wandb_run.summary[k] = v


if __name__ == "__main__":
    fire.Fire(main)
