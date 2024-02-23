"""Load Gemma with parameter efficiency."""

from typing import Any, Dict, Optional, Tuple

import torch
from absl import flags
from peft import (
    LoftQConfig,
    LoraConfig,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

FLAGS = flags.FLAGS

flags.DEFINE_string("pretrained_model", "google/gemma-7b-it", "initial pre-trained model to use as backbone LM.")

flags.DEFINE_integer("maximum_sequence_length", 1024, "The maximum sequence length.")
flags.DEFINE_integer("r", 16, "rank hyper-parameter for lora.")
flags.DEFINE_integer("lora_alpha", 8, "alpha hyper-parameter for lora.")
flags.DEFINE_float("lora_dropout", 0.05, "dropout rate hyper-parameter for lora.")
flags.DEFINE_integer("prompt_length", 25, "length of the prompts in the input sequence for soft prompt tuning.")
flags.DEFINE_string(
    "prompt_tuning_init_text", "Classify the text for me.", "What text to use to initialize the soft prompt embedding."
)
# Make sure we have some tokens defined for the gemma, if not defined in the model.
_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "cls_token": "<CLS>",
}


def load_peft_model_and_tokenizer(
    load_in_4bit: bool = True,
    adapter_name: str = "lora",
    is_trainable: bool = False,
) -> Tuple[torch.nn.Module, PreTrainedTokenizer]:
    """Load a trained PEFT adapter to the base model and return the PeftModel.

    Args:
    ----
        load_in_4bit: Whether to load the model in 4bit.
        adapter_name: e.g. lora
        is_trainable: train or inference mode

    Returns:
    -------
        The PEFT model and tokenizer.
    """
    model, tokenizer = load_model_and_tokenizer(load_in_4bit)
    if adapter_name == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not is_trainable,
            r=FLAGS.r,
            lora_alpha=FLAGS.lora_alpha,
            lora_dropout=FLAGS.lora_dropout,
            bias="none",
            init_lora_weights="loftq",
            loftq_config=LoftQConfig(loftq_bits=4),
        )

    elif adapter_name == "soft_prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=FLAGS.prompt_length,
            prompt_tuning_init_text=FLAGS.prompt_tuning_init_text,
            tokenizer_name_or_path=FLAGS.pretrained_model,
        )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model, tokenizer


def load_model_and_tokenizer(load_in_4bit: Optional[bool] = True) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
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
    model_args: Dict[str, Any] = {"use_cache": True, "torch_dtype": torch.bfloat16, "attn_implementation": "flash_attention_2"}
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_args["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
        model_args["quantization_config"] = quant_config
    model = AutoModelForCausalLM.from_pretrained(
        FLAGS.pretrained_model,
        **model_args,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_model)
    tokenizer.add_special_tokens(_EXTRA_TOKENS)
    tokenizer.model_max_length = FLAGS.maximum_sequence_length

    if torch.cuda.is_available():
        # extend embeddings to a multiple so we use Tensor cores
        multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
        model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=multiple)
    else:
        raise Exception("No CUDA Found!")

    # re-define token ids for the model.
    for extra_token_key, extra_token_val in _EXTRA_TOKENS.items():
        extra_token_id = tokenizer.convert_tokens_to_ids([extra_token_val])[0]
        model.config.__setattr__(f"{extra_token_key}_id", extra_token_id)

    return model, tokenizer
