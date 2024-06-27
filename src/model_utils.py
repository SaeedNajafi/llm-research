import functools
import os
import random
import re
from dataclasses import asdict
from typing import Any, Callable, Dict

import numpy
import torch
import torch.distributed as dist
from peft import LoraConfig, PeftModel, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import _or_policy, lambda_auto_wrap_policy, transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, PreTrainedModel

from src.configs import lora_config as LORA_CONFIG


def get_lora_model_from_base_model(
    base_model: PreTrainedModel,
    lora_config: LORA_CONFIG,
    path_to_peft_adapter_to_restore: str | None = None,
) -> PeftModel:
    """Initialize lora peft configuration from a non-lora model.

    Args:
    ----
        base_model: HuggingFace Transformer model to wrap.
        path_to_peft_adapter_to_restore: optionally, initialize peft adapters
            using tensors loaded from the filesystem.

    Returns:
    -------
        PeftModel
    """
    # See github.com/pytorch/pytorch/pull/102212
    base_model.load_state_dict(base_model.state_dict(), assign=True)

    if path_to_peft_adapter_to_restore is not None:
        lora_model = PeftModel.from_pretrained(
            base_model,
            path_to_peft_adapter_to_restore,
            is_trainable=True,
        )
        print(f"Restored peft_adapter from {path_to_peft_adapter_to_restore}.")
    else:
        lora_model = get_peft_model(base_model, LoraConfig(**asdict(lora_config)))

    lora_model = lora_model.bfloat16()
    assert isinstance(lora_model, PeftModel)
    lora_model.print_trainable_parameters()
    return lora_model


def load_model(
    path: str,
    use_mp: bool,
    use_fa: bool,
    load_in_8bit: bool,
    local_rank: int,
    low_cpu_mem_usage: bool,
    use_safetensors: bool = True,
) -> PreTrainedModel:
    """Load the model.

    Args:
    ----
        path: The path where the model and tokenizer are stored.
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        use_safetensors: Whether to use HF safe tensors. Note that this format
            loads significantly faster.

    Returns:
    -------
        The model.
    """
    # load model
    model_args: Dict[str, Any] = {"use_cache": False, "use_safetensors": use_safetensors, "load_in_8bit": load_in_8bit}

    if use_mp:
        model_args["torch_dtype"] = torch.bfloat16
    if use_fa:
        if not use_mp:
            msg = "Use FA with bf16 (mixed precision)"
            raise ValueError(msg)
        model_args["attn_implementation"] = "flash_attention_2"

    if not low_cpu_mem_usage or local_rank == 0:
        model = AutoModelForCausalLM.from_pretrained(
            path,
            **model_args,
        )
    else:
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_pretrained(
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
    use_mp: bool,
    layer_to_wrap: nn.Module,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
    is_lora_enabled: bool = False,
) -> Dict[str, Any]:
    """Get FSDP config.

    Args:
    ----
        use_mp: Whether to use mixed-precision.
        layer_to_wrap: The layer we are wrapping using FSDP.
        strategy: The sharding strategy to use.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        is_lora_enabled: Whether to enable LoRA support.

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

    strategy_exists = hasattr(ShardingStrategy, strategy)
    if not strategy_exists:
        msg = f"The sharding strategy {strategy} does not exist."
        raise ValueError(msg)

    ret_dict = {}
    if use_mp:
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

    if is_lora_enabled:
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

    sharding_strategy = getattr(ShardingStrategy, strategy)

    ret_dict["auto_wrap_policy"] = auto_wrap_policy
    ret_dict["sharding_strategy"] = sharding_strategy
    ret_dict["device_id"] = torch.cuda.current_device()
    ret_dict["forward_prefetch"] = True
    if low_cpu_mem_usage:
        ret_dict["param_init_fn"] = _module_init_fn if local_rank != 0 else None
        ret_dict["sync_module_states"] = True
    return ret_dict


def shard_model(
    model: nn.Module,
    layer_to_wrap: type[nn.Module],
    use_mp: bool,
    use_activation_checkpointing: bool,
    strategy: str,
    local_rank: int,
    low_cpu_mem_usage: bool,
    is_lora_enabled: bool = False,
) -> nn.Module:
    """Shard the model to workers using FSDP.

    Args:
    ----
        model: The model to be sharded.
        layer_to_wrap: The layer we are wrapping using FSDP.
        use_mp: Whether to use mixed-precision.
        use_activation_checkpointing: Whether to use activation checkpointing.
        strategy: The sharding strategy to use.
        local_rank: The local rank of the current worker.
        low_cpu_mem_usage: Whether to only load model weights on main rank, and
            then scatter them to the other workers.
        is_lora_enabled: Whether to enable support for LoRA, where requires_grad
            is True for only a subset of parameter tensors. Enabling might
            significantly reduce training throughput, so enable this only when
            actually using LoRA.

    Returns:
    -------
        The sharded module with the requested configurations.
    """
    fsdp_cfg = fsdp_config(
        use_mp,
        layer_to_wrap,
        strategy,
        local_rank,
        low_cpu_mem_usage,
        is_lora_enabled,
    )
    if dist.get_rank() == 0:
        print(f"FSDP config: {fsdp_cfg}")
    model = FSDP(model, **fsdp_cfg)
    print(
        "Model sharded. Per device model parameters are ",
        f"{sum(p.numel() for p in model.parameters())}",
    )

    if use_activation_checkpointing:
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


def log_of_labels(logits: torch.Tensor, labels: torch.Tensor, loss_func: torch.nn.CrossEntropyLoss) -> torch.Tensor:
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


def llama_log_of_labels(logits: torch.Tensor, labels: torch.Tensor, loss_func: torch.nn.CrossEntropyLoss) -> torch.Tensor:
    """Compute the actual log of labels given pre-computed logits."""

    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    return log_of_labels(shift_logits, shift_labels, loss_func)


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

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
