# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# Modified based on my applications in squad.

import os
from typing import Any

import torch
import wandb
from absl import app, flags
from torch.distributed.fsdp import FullStateDictConfig  # general model non-sharded, non-flattened params
from torch.distributed.fsdp import StateDictType  # general model non-sharded, non-flattened params
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from src.llm import Gemma2QA, Llama3QA, Llama31QA
from src.metrics import qa_metric_squadv2_metrics
from src.trainers import LossCalculator
from src.utils.data_utility import create_squadv2_dataloader
from src.utils.general_utils import clear_gpu_cache, set_random_seed
from src.utils.save_utils import find_checkpoint
from src.utils.train_utils import evaluation, setup, setup_environ_flags, train

FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "train | inference")
flags.DEFINE_integer("seed", 42, "random seed.")
flags.DEFINE_string("project_name", "llm_research", "name for these runs.")
flags.DEFINE_string("experiment_type", "normal_no_icl", "normal_no_icl | normal_icl | explanation_icl | explanation_no_icl")
flags.DEFINE_integer("train_batch_size", 8, "train batch size.")
flags.DEFINE_integer("eval_batch_size", 8, "eval batch size.")
flags.DEFINE_string("objective_type", "reinforce", "Different objectives to get the loss for training the llm.")


def setup_wandb() -> Any:
    """Setup the wandb account."""
    init_dict = {"project": FLAGS.project_name}
    run = wandb.init(**init_dict)
    wandb.config.update(FLAGS)
    return run


def main(argv: Any) -> None:
    del argv

    # Set the seeds for reproducibility
    set_random_seed(FLAGS.seed)

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
    if rank == 0:
        wandb_run = setup_wandb()

    # Initialize the model here.
    if FLAGS.llm_name == "gemma2":
        model = Gemma2QA(local_rank, rank)

    elif FLAGS.llm_name == "llama3":
        model = Llama3QA(local_rank, rank)

    elif FLAGS.llm_name == "llama3.1":
        model = Llama31QA(local_rank, rank)

    if wandb_run:
        if FLAGS.use_peft:
            wandb_run.config.update(model.peft_config)

    if FLAGS.objective_type in ["teacher_forcing", "reinforce"]:
        loss_calculator = LossCalculator(policy_lm=model, objective_type=FLAGS.objective_type)

    if FLAGS.mode == "train":
        train_dataloader = create_squadv2_dataloader(
            model,
            fold_name="train",
            experiment_type=FLAGS.experiment_type,
            batch_size=FLAGS.train_batch_size,
            world_size=world_size,
            rank=rank,
        )

        eval_dataloader = create_squadv2_dataloader(
            model,
            fold_name="dev",
            experiment_type=FLAGS.experiment_type,
            batch_size=FLAGS.eval_batch_size,
            world_size=world_size,
            rank=rank,
        )

        # Start the training process
        results = train(
            model,
            loss_calculator,
            train_dataloader,
            eval_dataloader,
            rank,
            world_size,
            wandb_run,
            qa_metric_squadv2_metrics,
        )

        if rank == 0:
            for k, v in results.items():
                wandb_run.summary[k] = v

    elif FLAGS.mode == "inference":
        test_dataloader = create_squadv2_dataloader(
            model,
            fold_name="test",
            experiment_type=FLAGS.experiment_type,
            batch_size=FLAGS.eval_batch_size,
            world_size=world_size,
            rank=rank,
        )

        _, _ = find_checkpoint(model)

        # Start the test process
        evaluation(
            model,
            "test",
            test_dataloader,
            FLAGS.prediction_file,
            rank,
            world_size,
            wandb_run,
            qa_metric_squadv2_metrics,
        )

    elif FLAGS.mode == "deploy":
        # load the rest of the model weights if not peft.
        _, _ = find_checkpoint(model)
        if rank == 0:
            deploy_dir = os.path.join(FLAGS.checkpoint_folder, "final_model")
            os.makedirs(deploy_dir, exist_ok=True)
            model.tokenizer.save_pretrained(deploy_dir)
            if model.distributed_strategy == "ddp":
                # merge lora to the base model.
                merged_model = model.model.module.merge_and_unload() if FLAGS.use_peft else model.model.module
            elif model.distributed_strategy == "fsdp":
                # merge lora to the base model.
                merged_model = model.model.merge_and_unload() if FLAGS.use_peft else model.model

            if model.distributed_strategy == "ddp":
                merged_model.save_pretrained(deploy_dir, safe_serialization=False)
            elif model.distributed_strategy == "fsdp":
                with FSDP.state_dict_type(
                    merged_model,
                    StateDictType.FULL_STATE_DICT,
                    FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
                ):
                    merged_model.save_pretrained(deploy_dir, safe_serialization=False)


if __name__ == "__main__":
    app.run(main)
