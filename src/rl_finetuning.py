# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.


# Modified based on my applications in squad.

import os
from typing import Any

import torch
import wandb
from absl import app, flags

from src.llm import Gemma2QA, Llama3QA, Llama31QA
from src.trainers import LossCalculator
from src.utils.data_utility import create_squadv2_dataloader
from src.utils.general_utils import clear_gpu_cache, set_random_seed
from src.utils.train_utils import setup, setup_environ_flags

FLAGS = flags.FLAGS

flags.DEFINE_string("mode", "train", "train | inference")
flags.DEFINE_integer("seed", 42, "random seed.")
flags.DEFINE_string("project_name", "llm_research", "name for these runs.")
flags.DEFINE_string("experiment_type", "normal_no_icl", "normal_no_icl | normal_icl | explanation_icl | explanation_no_icl")
flags.DEFINE_integer("train_batch_size", 8, "train batch size.")
flags.DEFINE_integer("eval_batch_size", 8, "eval batch size.")


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

    test_dataloader = create_squadv2_dataloader(
        model,
        fold_name="test",
        experiment_type=FLAGS.experiment_type,
        batch_size=FLAGS.eval_batch_size,
        world_size=world_size,
        rank=rank,
    )

    loss_calculator = LossCalculator(policy_lm=model, iterative_computation=False, reward_normalization_type="mml_normalize")
    num_samples = 4
    idx = 0
    for batch in test_dataloader:
        answers, log_ps = model.generation_pass(batch, num_samples)
        cleaned_answers = [answer.removeprefix("assistant\n\n") for answer in answers]
        templated_answers = [model.output_template.format(output=f"Final Answer: {answer}") for answer in cleaned_answers]
        sample_outputs = [
            templated_answers[b_idx * num_samples : (b_idx + 1) * num_samples] for b_idx in range(FLAGS.eval_batch_size)
        ]
        sample_rewards = [[1.1, 1.2, 1.3, 1.4], [1.1, 1.2, 1.3, 1.4]]
        loss = loss_calculator.reinforce_style(batch, sample_outputs, sample_rewards)
        print(idx, loss)
        clear_gpu_cache()
        idx += 1
        if idx == 10:
            break


if __name__ == "__main__":
    app.run(main)
