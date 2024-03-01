"""Load LM efficiently."""

from typing import Any, Dict, List, Optional, Tuple

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
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("r", 16, "rank hyper-parameter for lora.")
flags.DEFINE_integer("lora_alpha", 8, "alpha hyper-parameter for lora.")
flags.DEFINE_float("lora_dropout", 0.05, "dropout rate hyper-parameter for lora.")
flags.DEFINE_integer("prompt_length", 25, "length of the prompts in the input sequence for soft prompt tuning.")
flags.DEFINE_string(
    "prompt_tuning_init_text", "Classify the text for me.", "What text to use to initialize the soft prompt embedding."
)

# Make sure we have some tokens defined for the LM, if not defined in the model.
_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "cls_token": "<CLS>",
}

target_modules = ["q_proj", "v_proj"]


def load_peft_model(
    model: PreTrainedModel,
    num_quantized_bits: int = 16,
    adapter_name: str = "lora",
    is_trainable: bool = False,
    model_type: str = "causal_lm",
    lora_target_modules: List[str] = target_modules,
) -> torch.nn.Module:
    """Load a trained PEFT adapter to the base model and return the PeftModel.

    Args:
    ----
        model: the main model.
        num_quantized_bits: number of bits in the loaded model.
        adapter_name: e.g. lora.
        is_trainable: train or inference mode.
        model_type: causal lm or seq-to-seq.
        lora_target_modules: which modules to train with lora.

    Returns:
    -------
        The PEFT model and tokenizer.
    """
    if model_type == "causal_lm":
        task_type = TaskType.CAUSAL_LM
    elif model_type == "seq_to_seq_lm":
        task_type = TaskType.SEQ_2_SEQ_LM

    if adapter_name == "lora":
        peft_config = LoraConfig(
            task_type=task_type,
            inference_mode=not is_trainable,
            r=FLAGS.r,
            lora_alpha=FLAGS.lora_alpha,
            lora_dropout=FLAGS.lora_dropout,
            bias="none",
            init_lora_weights="loftq",
            loftq_config=LoftQConfig(loftq_bits=num_quantized_bits),
            target_modules=lora_target_modules,
        )

    elif adapter_name == "soft_prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type=task_type,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=FLAGS.prompt_length,
            prompt_tuning_init_text=FLAGS.prompt_tuning_init_text,
            tokenizer_name_or_path=model.config._name_or_path,
        )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_model_and_tokenizer(
    model_id: str, model_type: str, model_dtype: torch.dtype, attn_implementation: str, load_in_4bit: Optional[bool] = True
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load the model and tokenizer.

    Args:
    ----
        model_id: the id for the pre-trained model.
        model_type: causal lm or seq_to_seq_lm.
        model_dtype: model data type.
        load_in_4bit: Whether to load in 4 bit quantization.

    Returns:
    -------
        The model and tokenizer.
    """
    # load model
    if model_type == "causal_lm":
        ModelClass = AutoModelForCausalLM
    elif model_type == "seq_to_seq_lm":
        ModelClass = AutoModelForSeq2SeqLM
    model_args: Dict[str, Any] = {"use_cache": False, "torch_dtype": model_dtype, "attn_implementation": attn_implementation}
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=model_args["torch_dtype"],
            bnb_4bit_use_double_quant=True,
        )
        model_args["quantization_config"] = quant_config
    model = ModelClass.from_pretrained(
        model_id,
        **model_args,
    )
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.add_special_tokens(_EXTRA_TOKENS)

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
