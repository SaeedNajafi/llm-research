"""Main binary to run the application to train or test with the model."""

import os
from typing import Any

import torch
import torch.distributed as dist
from absl import app, flags
from torch.nn.parallel import DistributedDataParallel as DDP

from bitsandbytes.optim.adamw import PagedAdamW8bit
from src.load_gemma import load_peft_model_and_tokenizer
from src.model_utils import set_random_seed

from huggingface_hub import notebook_login
from datasets import load_dataset
from datasets import ClassLabel
import random
import pandas as pd
from IPython.display import display, HTML
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
from transformers import set_seed
import math

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

    model, tokenizer = load_peft_model_and_tokenizer(load_in_4bit=True, adapter_name="lora", is_trainable=True)
    gig_factor = 1024 * 1024 * 1024
    print(torch.cuda.mem_get_info()[1]/gig_factor)
    print(torch.cuda.mem_get_info()[0]/gig_factor)
    ddp_model = DDP(model, device_ids=[local_rank])
    print(torch.cuda.mem_get_info()[0]/gig_factor)

    
    def show_random_elements(dataset, num_examples=10):
        assert num_examples <= len(dataset), "Can't pick more elements than there are in the dataset."
        picks = []
        for _ in range(num_examples):
            pick = random.randint(0, len(dataset)-1)
            while pick in picks:
                pick = random.randint(0, len(dataset)-1)
            picks.append(pick)
    
        df = pd.DataFrame(dataset[picks])
        for column, typ in dataset.features.items():
            if isinstance(typ, ClassLabel):
                df[column] = df[column].transform(lambda i: typ.names[i])
        display(HTML(df.to_html()))

    def tokenize_function(examples):
        return tokenizer(examples["text"])

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
            # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    datasets = load_dataset('wikitext', 'wikitext-2-v1')
    train_split = datasets["train"]
    smaller_train = train_split.train_test_split(train_size=100)
    datasets["train"] = smaller_train["train"]
    eval_split = datasets["validation"]
    smaller_eval = eval_split.train_test_split(test_size=100)
    datasets["validation"] = smaller_eval["test"]
    show_random_elements(datasets["train"])
    tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    block_size = 128
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        batch_size=16,
        num_proc=4,
    )

    # load optimizer
    optimizer = PagedAdamW8bit(ddp_model.parameters(), lr=0.001)
    print(torch.cuda.mem_get_info()[0]/gig_factor)
    training_args = TrainingArguments(
        f"wikitext2",
        evaluation_strategy = "epoch",
        push_to_hub=False,
    )
    trainer = Trainer(
        model=ddp_model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        optimizers=(optimizer, None)
    )
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    del ddp_model
    del optimizer
    del model

if __name__ == "__main__":
    setup()
    app.run(main)
    cleanup()
