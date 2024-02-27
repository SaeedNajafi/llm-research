import gc
import os
import random

import numpy
import tensorflow as tf
import torch
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("seed", 42, "the seed number")


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int) -> torch.Tensor:
    """Shift input ids one token to the right.

    source: https://github.com/huggingface/transformers/blob/main/src/transformers/models/bart/modeling_bart.py#L100
    This function can be used for both t5 and bart.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def optimizer_to(optim: torch.optim.Optimizer, device: str) -> None:
    """Move the optimizer to a specific device."""
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def encoder_decoder_log_of_labels(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    decoder_mask: torch.Tensor,
    labels: torch.Tensor,
    loss_func: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Do a forward computation and compute the log probability for the given
    labels."""

    # shift the gold labels one step to the right and do teacher-forcing by giving the gold previous token
    # and then compute the probability for the next token at each step.
    # labels = [pos, it, ive]
    # decoder_input = [<BOS>, pos, it]
    # we want to see what is log probability for the target sequence "positive".

    shifted_labels = shift_tokens_right(labels, model.config.pad_token_id, model.config.decoder_start_token_id)
    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        decoder_attention_mask=decoder_mask,
        decoder_input_ids=shifted_labels,
        labels=None,
    )

    log_p = -loss_func(
        output.logits.view(-1, output.logits.size(-1)),
        labels.view(-1),
    )
    batch_size, sequence_length, vocab_size = output.logits.size()
    # compute per-token log probability in a sequence.
    # log_p has log probabilities for the following target output: [pos, it, ive]
    log_p = log_p.view(batch_size, sequence_length)

    # pad tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill_(labels == -100, 0.0)

    # good_log_p now has the log probability of the output sequence tokens.
    # sum over the sequence length to compute the log probability for a full sequence.
    return torch.sum(good_log_p, dim=1).squeeze()


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

    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
