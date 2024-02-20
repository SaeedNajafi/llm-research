"""Based on vectorllm model for sharding.

source: https://github.com/VectorInstitute/vectorlm/blob/master/vectorlm/utils/model_utils.py
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
from absl import flags
from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import BitsAndBytesConfig, LlamaForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "llama2_pretrained_model", "/model-weights/Llama-2-7b-hf", "initial pre-trained model to use as backbone LM."
)

flags.DEFINE_integer("maximum_sequence_length", 1024, "The maximum sequence length.")
flags.DEFINE_integer("r", 8, "rank hyper-parameter for lora.")
flags.DEFINE_integer("lora_alpha", 32, "alpha hyper-parameter for lora.")
flags.DEFINE_float("lora_dropout", 0.1, "dropout rate hyper-parameter for lora.")
flags.DEFINE_integer("prompt_length", 25, "length of the prompts in the input sequence for soft prompt tuning.")
flags.DEFINE_string(
    "prompt_tuning_init_text", "Classify the text for me.", "What text to use to initialize the soft prompt embedding."
)
# Make sure we have some tokens defined for the llama, if not defined in the model.
_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "cls_token": "<CLS>",
}


def load_peft_model_and_tokenizer(
    use_mp: bool,
    use_fa: bool,
    adapter_name: str = "lora",
    is_trainable: bool = False,
) -> tuple[nn.Module, PreTrainedTokenizer]:
    """Load a trained PEFT adapter to the base model and return the PeftModel.

    Args:
    ----
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.
        adapter_name: e.g. lora
        is_trainable: train or inference mode

    Returns:
    -------
        The PEFT model and tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer(use_mp, use_fa, define_extra_tokens=True, load_in_4bit=True)
    if adapter_name == "lora":
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not is_trainable,
            r=FLAGS.r,
            init_lora_weights=True,
            lora_alpha=FLAGS.lora_alpha,
            lora_dropout=FLAGS.lora_dropout,
        )
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

    elif adapter_name == "soft_prompt_tuning":
        prompt_tuning_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=FLAGS.prompt_length,
            prompt_tuning_init_text=FLAGS.prompt_tuning_init_text,
            tokenizer_name_or_path=FLAGS.llama2_pretrained_model,
        )
        peft_model = get_peft_model(model, prompt_tuning_config)
        peft_model.print_trainable_parameters()

    return peft_model, tokenizer


def load_model_and_tokenizer(
    use_mp: bool,
    use_fa: bool,
    define_extra_tokens: Optional[bool] = True,
    load_in_4bit: Optional[bool] = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the model and tokenizer.

    Args:
    ----
        use_mp: Whether to use mixed-precision.
        use_fa: Whether to use Flash Attention 2.

    Returns:
    -------
        The model and tokenizer.
    """
    # load model
    model_args: Dict[str, Any] = {"use_cache": False}

    if use_mp:
        model_args["torch_dtype"] = torch.bfloat16
    if use_fa:
        if not use_mp:
            msg = "Use FA with bf16 (mixed precision)"
            raise ValueError(msg)
        model_args["attn_implementation"] = "flash_attention_2"

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_args["torch_dtype"],
            bnb_4bit_use_double_quant=False,
        )
        model_args["quantization_config"] = quant_config
    model = LlamaForCausalLM.from_pretrained(
        FLAGS.llama2_pretrained_model,
        **model_args,
    )

    # load tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(FLAGS.llama2_pretrained_model)
    if define_extra_tokens:
        tokenizer.add_special_tokens(_EXTRA_TOKENS)
    tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training

    tokenizer.model_max_length = FLAGS.maximum_sequence_length

    # extend embeddings to a multiple so we use Tensor cores
    multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
    model.resize_token_embeddings(
        len(tokenizer),
        pad_to_multiple_of=multiple,
    )

    # re-define token ids for the model.
    if define_extra_tokens:
        for extra_token_key, extra_token_val in _EXTRA_TOKENS.items():
            extra_token_id = tokenizer.convert_token_to_ids([extra_token_val])[0]
            model.config.__setattr__(f"{extra_token_key}_id", extra_token_id)

    return model, tokenizer


def fsdp_config(
    use_mp: bool,
    layer_to_wrap: nn.Module,
    strategy: str,
) -> dict[str, Any]:
    """Get FSDP config.

    Args:
    ----
        use_mp: Whether to use mixed-precision.
        layer_to_wrap: The layer we are wrapping using FSDP.
        strategy: The sharding strategy to use.

    Returns:
    -------
        A dictionary containing the configurations.
    """
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

    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_to_wrap},
    )
    sharding_strategy = getattr(ShardingStrategy, strategy)

    ret_dict["auto_wrap_policy"] = auto_wrap_policy
    ret_dict["sharding_strategy"] = sharding_strategy
    ret_dict["device_id"] = torch.cuda.current_device()
    return ret_dict


def shard_model(
    model: nn.Module,
    layer_to_wrap: nn.Module,
    use_mp: bool,
    use_activation_checkpointing: bool,
    strategy: str,
) -> nn.Module:
    """Shard the model to workers using FSDP.

    Args:
    ----
        model: The model to be sharded.
        layer_to_wrap: The layer we are wrapping using FSDP.
        use_mp: Whether to use mixed-precision.
        use_activation_checkpointing: Whether to use activation checkpointing.
        strategy: The sharding strategy to use.

    Returns:
    -------
        The sharded module with the requested configurations.
    """
    fsdp_cfg = fsdp_config(use_mp, layer_to_wrap, strategy)
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
    layer: nn.Module,
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

    def check_fn(submodule: nn.Module) -> bool:
        """Check if submodule is an instance of layer."""
        return isinstance(submodule, layer)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=check_fn,
    )
