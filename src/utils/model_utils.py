"""Functions to compute log likelihood from the transformer models.

Also, it has functions to load model, peft config, and fsdp strategy for
the model.
"""

import functools
import os
import re
from typing import Any, Callable, Dict

import torch
import torch.distributed as dist
from absl import flags, logging
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, PreTrainedModel

from src.my_modelling_llama import LlamaForCausalLM
from src.utils.save_utils import checkpoint_exists, get_latest_checkpoint_dir

FLAGS = flags.FLAGS

# Lora related arguments.
flags.DEFINE_integer("r", 512, "rank in the lora.")

# It seems we are using rsLoRA with an alpha of 22.63 and rank 512 means,
# lora_alpha = 22.63*(512**.5) = 512.
flags.DEFINE_integer("lora_alpha", 512, "alpha hyper-parameter in lora.")
flags.DEFINE_float("lora_dropout", 0.3, "dropout in lora", upper_bound=0.5, lower_bound=0.0)
flags.DEFINE_list("target_modules", "q_proj,v_proj,o_proj,k_proj", "target modules in the lora.")
flags.DEFINE_string("path_to_peft_adapter_to_restore", "", "path where the adapter is.")
flags.DEFINE_boolean("use_peft", True, "whether to use peft for training or not?")


# FSDP related arguments.
flags.DEFINE_boolean("low_cpu_mem_usage", True, "helpful for fsdp?")
flags.DEFINE_boolean("use_mp", True, "mixed precision training?")
flags.DEFINE_string("attn_implementation", "flash_attention_2", "flash_attention_2 | eager")
flags.DEFINE_boolean("use_activation_checkpointing", True, "whether to use activation checkpointing.")
flags.DEFINE_boolean("enable_nf4", False, "whether to apply nf4 4-bit quantization.")

flags.DEFINE_string("sharding_strategy", "NO_SHARD", "NO_SHARD | HYBRID_SHARD | SHARD_GRAD_OP")
flags.DEFINE_boolean("ddp", True, "is this a pure ddp run?")
flags.DEFINE_string("llm_name", "gemma2", "gemma2 | llama3")


def get_lora_model_from_base_model(base_model: PreTrainedModel) -> PeftModel:
    """Initialize lora peft configuration from a non-lora model.

    Args:
    ----
        base_model: HuggingFace Transformer model to wrap.
    Returns:
    -------
        PeftModel
    """
    # See github.com/pytorch/pytorch/pull/102212
    base_model.load_state_dict(base_model.state_dict(), assign=True)
    checkpoint = checkpoint_exists(FLAGS.checkpoint_folder)
    if checkpoint:
        checkpoint_path = os.path.join(FLAGS.checkpoint_folder, "checkpoints")
        peft_adapter_path = os.path.join(checkpoint_path, get_latest_checkpoint_dir(checkpoint_path))
        lora_model = PeftModel.from_pretrained(
            base_model,
            peft_adapter_path,
            is_trainable=True,
        )
        logging.info(f"Restored peft_adapter from {peft_adapter_path}.")
        lora_model.is_peft_adapter_restored = True

    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=FLAGS.r,
            lora_alpha=FLAGS.lora_alpha,
            lora_dropout=FLAGS.lora_dropout,
            bias="none",
            init_lora_weights=True,
            target_modules=FLAGS.target_modules,
        )
        lora_model = get_peft_model(base_model, lora_config)
        lora_model.is_peft_adapter_restored = False

    lora_model = lora_model.bfloat16()
    assert isinstance(lora_model, PeftModel)
    lora_model.print_trainable_parameters()
    return lora_model


def load_model(
    path: str,
    local_rank: int,
    use_safetensors: bool = True,
    device: str = "cuda:0",
    is_fsdp: bool = False,
) -> PreTrainedModel:
    """Load the model.

    Args:
    ----
        path: The path where the model and tokenizer are stored.
        local_rank: The local rank of the current worker.
        use_safetensors: Whether to use HF safe tensors. Note that this format
            loads significantly faster.

    Returns:
    -------
        The model.
    """
    # load model
    model_args: Dict[str, Any] = {"use_cache": True, "use_safetensors": use_safetensors}

    if not is_fsdp:
        model_args["device_map"] = device

    if FLAGS.use_mp:
        model_args["torch_dtype"] = torch.bfloat16
    if FLAGS.attn_implementation != "":
        if not FLAGS.use_mp:
            msg = "Use FA with bf16 (mixed precision)"
            raise ValueError(msg)
        model_args["attn_implementation"] = FLAGS.attn_implementation

    if FLAGS.enable_nf4:
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model_args["quantization_config"] = nf4_config

    if FLAGS.llm_name in ["llama3", "llama3.2"]:
        # Load my own lm modelling code.
        model_class = LlamaForCausalLM
    else:
        model_class = AutoModelForCausalLM

    if FLAGS.ddp:
        model = model_class.from_pretrained(
            path,
            **model_args,
        )
    else:
        # for fsdp
        if not FLAGS.low_cpu_mem_usage or local_rank == 0:
            model = model_class.from_pretrained(
                path,
                **model_args,
            )
        else:
            with torch.device("meta"):
                model = model_class.from_pretrained(
                    path,
                    **model_args,
                )
    return model


def lora_requires_grad_policy_fn(module: nn.Module) -> bool:
    """Policy that "turns off" FSDP Flat Param for LoRA-enabled layers.

    FSDP requires consistent requires_grad for each flat param.

    Since LoRA requires_grad tensors are embedded within each layer,
    this policy "turns off" FSDP flat param optimization by requiring a
    separate flat param block for each tensor.
    """
    if len(list(module.named_children())) == 0 and getattr(module, "weight", None) is not None and module.weight.requires_grad:
        return True
    return False


def fsdp_config(
    layer_to_wrap: nn.Module,
    local_rank: int,
) -> Dict[str, Any]:
    """Get FSDP config.

    Args:
    ----
        layer_to_wrap: The layer we are wrapping using FSDP.
        local_rank: The local rank of the current worker.

    Returns:
    -------
        A dictionary containing the configurations.
    """

    def _module_init_fn(module: nn.Module) -> Callable:
        """Return the function used for initializing modules on FSDP
        workers."""
        return module.to_empty(
            device=torch.cuda.current_device(),
            recurse=False,
        )

    strategy_exists = hasattr(ShardingStrategy, FLAGS.sharding_strategy)
    if not strategy_exists:
        msg = f"The sharding strategy {FLAGS.sharding_strategy} does not exist."
        raise ValueError(msg)

    ret_dict = {}
    if FLAGS.use_mp:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
        )
        ret_dict["mixed_precision"] = mp_policy

    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )

    if FLAGS.use_peft:
        # turns off FSDP Flat Param in LoRA layers.
        lambda_requires_grad_policy = functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lora_requires_grad_policy_fn,
        )
        auto_wrap_policy = functools.partial(
            _or_policy,
            policies=[lambda_requires_grad_policy, transformer_wrap_policy],
        )
    else:
        auto_wrap_policy = transformer_wrap_policy

    sharding_strategy = getattr(ShardingStrategy, FLAGS.sharding_strategy)

    ret_dict["auto_wrap_policy"] = auto_wrap_policy
    ret_dict["sharding_strategy"] = sharding_strategy
    ret_dict["device_id"] = torch.cuda.current_device()
    ret_dict["forward_prefetch"] = True
    if FLAGS.low_cpu_mem_usage:
        ret_dict["param_init_fn"] = _module_init_fn if local_rank != 0 else None
        ret_dict["sync_module_states"] = True
    return ret_dict


def shard_model(
    model: nn.Module,
    layer_to_wrap: type[nn.Module],
    local_rank: int,
) -> nn.Module:
    """Shard the model to workers using FSDP.

    Args:
    ----
        model: The model to be sharded.
        layer_to_wrap: The layer we are wrapping using FSDP.
        local_rank: The local rank of the current worker.

    Returns:
    -------
        The sharded module with the requested configurations.
    """
    fsdp_cfg = fsdp_config(
        layer_to_wrap,
        local_rank,
    )
    if dist.get_rank() == 0:
        print(f"FSDP config: {fsdp_cfg}")
    model = FSDP(model, **fsdp_cfg)
    msg = f"Model sharded. Per device model parameters are {sum(p.numel() for p in model.parameters())}"
    logging.info(msg)

    if FLAGS.use_activation_checkpointing:
        hook_activation_checkpointing(model, layer_to_wrap)
    return model


def hook_activation_checkpointing(
    model: nn.Module,
    layer: type[nn.Module],
) -> None:
    """Set activation checkpointing.

    Args:
    ----
        model: The model we are using.
        layer: The layer to which we hook activation checkpointing to.
    """
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=lambda submodule: isinstance(submodule, layer),
    )


def get_submodule_by_pattern(
    module: nn.Module,
    pattern: str,
) -> type[nn.Module] | None:
    """Return the first module.cls that matches pattern at least partially.

    With reference to get_module_class_from_name from HuggingFace
    accelerate `FullyShardedDataParallelPlugin`.

    Args:
    ----
        module: Layer container
        pattern: regular expression string.

    Returns:
    -------
        nn.Module: matched layer (nn.Module),
        or
        None: if not matched.
    """
    modules_children = list(module.children())
    module_name = module.__class__.__name__
    if re.search(pattern, module_name) is not None:
        return module.__class__

    if len(modules_children) == 0:
        return None

    for child_module in modules_children:
        module_class = get_submodule_by_pattern(child_module, pattern)
        if module_class is not None:
            return module_class

    return None


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


def log_of_labels(
    logits: torch.Tensor, labels: torch.Tensor, loss_func: torch.nn.CrossEntropyLoss, per_step_scores: bool = False
) -> torch.Tensor:
    """Compute the actual log of labels given pre-computed logits.

    This function is also useful for both Roberta model and getting
    generation logits for sampling methods.
    """
    log_p = -loss_func(
        logits.view(-1, logits.size(-1)),
        labels.view(-1),
    )

    batch_size, sequence_length, vocab_size = logits.size()

    # compute per-token log probability in a sequence.
    log_p = log_p.view(batch_size, sequence_length)

    # non-masked tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill(labels == -100, 0.0)

    # good_log_p now has the log probability of the output
    # sequence tokens corresponding to the labels at the [MASK] location.
    if per_step_scores:
        return torch.sum(good_log_p, dim=1), good_log_p
    return torch.sum(good_log_p, dim=1)


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
    return log_of_labels(output.logits, labels, loss_func)


def decoder_only_log_of_labels(
    logits: torch.Tensor, labels: torch.Tensor, loss_func: torch.nn.CrossEntropyLoss, per_step_scores: bool = False
) -> torch.Tensor:
    """Compute the actual log of labels given pre-computed logits."""

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return log_of_labels(shift_logits, shift_labels, loss_func, per_step_scores)


def lm_logits(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
) -> torch.Tensor:
    """Do a forward computation and compute the logits for the given input_ids
    for a language model."""

    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        labels=None,
    )
    return output.logits


def print_model_size(model: torch.nn.Module, model_path: str, rank: int = 0) -> None:
    """Print model name, the number of trainable parameters and initialization
    time.

    Args:
        model: The PyTorch model.
        model_path: name or path for model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logging.info(f"--> Model {model_path}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info(f"--> {model_path} has {total_params / 1e6} Million params.")
