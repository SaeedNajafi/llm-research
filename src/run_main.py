"""Main binary to run the application to train or test with the model."""

import os
from typing import Any

import torch
import torch.distributed as dist
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP

from src.load_gemma import load_peft_model_and_tokenizer
from src.model_utils import set_random_seed

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "the seed number")


def setup() -> None:
    """Initialize the process group."""
    dist.init_process_group("nccl")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def main(argv: Any) -> None:
    """Main function to launch the train and inference scripts."""

    del argv

    # set random seed.
    set_random_seed(FLAGS.seed)

    # get ranks
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    print(f"Rank: {rank}, World size: {world_size}")
    if dist.is_initialized():
        torch.cuda.set_device(local_rank)
        torch.cuda.empty_cache()

    dist.barrier()

    model, tokenizer = load_peft_model_and_tokenizer(load_in_4bit=True, adapter_name="lora", is_trainable=False)
    model = model.to("cuda")
    ddp_model = DDP(model, device_ids=[local_rank])

    del ddp_model


if __name__ == "__main__":
    setup()
    app.run(main)
    cleanup()
