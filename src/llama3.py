"""The main module to load llama3 with the chat prompt."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from galore_torch import GaLoreAdamW8bit
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.base_lm import BaseLM
from src.model_utils import llama2_log_of_labels, lm_logits, mlm_log_of_labels

# Make sure we have some tokens defined for the LM, if not defined in the model.
# Specific for Llama3
_EXTRA_TOKENS = {
    "pad_token": "<|reserved_special_token_0|>",
}

target_modules = ["q_proj", "v_proj", "o_proj", "k_proj"]


def load_peft_model(
    model: PreTrainedModel,
    r: int = 16,
    lora_alpha: int = 8,
    lora_dropout: float = 0.05,
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
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            init_lora_weights=True,
            target_modules=lora_target_modules,
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
    model_args: Dict[str, Any] = {"use_cache": False, "attn_implementation": attn_implementation, "torch_dtype": model_dtype}
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

    # load tokenizer
    # padding is from left for the decoder only models.
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
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
        model.generation_config.__setattr__(f"{extra_token_key}_id", extra_token_id)

    return model, tokenizer


class LlamaQA(BaseLM):
    """Class to implement Llama3."""

    def __init__(
        self,
        device: str,
        seed: int = 42,
        lm_input_max_length: int = 1024 - 32,
        lm_output_max_length: int = 32,
        lm_top_p: float = 0.9,
        temperature: float = 0.6,
        learning_rate: float = 0.00005,
        galore_rank: int = 128,
        galore_update_proj_gap: int = 16,
        galore_scale: float = 0.25,
    ) -> None:
        super().__init__(device, "llama3", seed)
        self.device = device
        self.learning_rate = learning_rate
        self.lm_top_p = lm_top_p
        self.temperature = temperature
        self.lm_input_max_length = lm_input_max_length
        self.lm_output_max_length = lm_output_max_length
        # Chat templates for llama3.
        self.llama3_instruction_llama = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction}<|eot_id|>"
        self.llama3_input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input}<|eot_id|>"
        self.llama3_output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output}<|eot_id|>"

        model, tokenizer = load_model_and_tokenizer(
            model_id="/model-weights/Meta-Llama-3-8B-Instruct",
            model_type="causal_lm",
            model_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        # to train the main lm, we update all of its parameters.
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            print("enable GaLore for weights in module: ", module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [
            {"params": regular_params},
            {
                "params": galore_params,
                "rank": galore_rank,
                "update_proj_gap": galore_update_proj_gap,
                "scale": galore_scale,
                "proj_type": "std",
            },
        ]
        self.optimizer = GaLoreAdamW8bit(param_groups, lr=learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=learning_rate / 5.0)

        # required for llama3.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    def prepare_text_for_inference(self, texts: List[str], row_ids: List[str], gold_answers: List[str] = None) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for
        inference."""
        input_encodings_for_generation = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.lm_input_max_length,
            add_special_tokens=False,
        )
        data = {
            "lm_input_ids_for_generation": input_encodings_for_generation.input_ids,
            "lm_attention_mask_for_generation": input_encodings_for_generation.attention_mask,
            "row_ids": row_ids,
        }
        if gold_answers is not None:
            data["gold_answers"] = gold_answers
        return data

    def prepare_text_for_train(self, texts: List[str], output_texts: List[str], row_ids: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for training
        and inference."""
        inputs_for_training = [f"{texts[idx]}{output_texts[idx]}" for idx in range(len(texts))]
        input_encodings = self.tokenizer(
            inputs_for_training,
            truncation=True,
            padding=True,
            max_length=self.lm_input_max_length + self.lm_output_max_length,
            add_special_tokens=False,
        )

        inference_data = self.prepare_text_for_inference(texts, row_ids)
        train_data = {
            "row_ids": inference_data["row_ids"],
            "lm_input_ids_for_train": input_encodings.input_ids,
            "lm_attention_mask_for_train": input_encodings.attention_mask,
            "lm_input_ids_for_generation": inference_data["lm_input_ids_for_generation"],
            "lm_attention_mask_for_generation": inference_data["lm_attention_mask_for_generation"],
        }
        return train_data

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the Llama, run a forward computation over the batch, compute
        the log probability over the batch.

        This will be used for training.
        """
        self.train_mode_on()
        loaded_batch = self.data_to_device(
            batch, keys=["lm_input_ids_for_train", "lm_attention_mask_for_train", "lm_attention_mask_for_generation"]
        )
        input_ids = loaded_batch["lm_input_ids_for_train"]
        attention_mask = loaded_batch["lm_attention_mask_for_train"]
        original_len_without_answer = torch.sum(loaded_batch["lm_attention_mask_for_generation"], dim=1)
        with torch.set_grad_enabled(True):
            logits = lm_logits(
                model=self.model,
                input_ids=input_ids,
                input_mask=attention_mask,
            )
            batch_size, seq_len = input_ids.size()
            masked_labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            prompt_mask = torch.arange(seq_len, device=self.device).expand(
                batch_size, seq_len
            ) < original_len_without_answer.unsqueeze(1)
            masked_labels = masked_labels.masked_fill(prompt_mask == 1, -100)
            return llama2_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.loss_func)

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor, Any]:
        """Using the Llama, generate new text.

        This will be used for inference.
        """
        self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_generation", "lm_attention_mask_for_generation"])
        input_ids = loaded_batch["lm_input_ids_for_generation"]
        attention_mask = loaded_batch["lm_attention_mask_for_generation"]
        with torch.no_grad():
            # more look here:
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L130
            predictions_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=self.lm_top_p,
                temperature=self.temperature,
                max_length=self.lm_input_max_length + self.lm_output_max_length,
                num_return_sequences=1,
                output_logits=True,
                output_hidden_states=True,
                return_dict_in_generate=True,
                use_cache=True,
                renormalize_logits=True,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        prompt_len = input_ids.size()[1]
        selected_samples = predictions_output.sequences[:, prompt_len:]
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        logits_list = list(predictions_output.logits)
        logits = torch.stack(logits_list, dim=1)
        labels_to_consider = selected_samples.masked_fill(selected_samples == self.tokenizer.pad_token_id, -100)
        final_log_ps = mlm_log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens, predictions_output

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop."""
        answers, log_ps, _ = self.generation_pass(batch)
        log_ps = log_ps.cpu().detach().numpy()
        for idx, answer in enumerate(answers):
            output_row = {"potential_answer": answer, "prediction_score": log_ps[idx], "row_id": batch["row_ids"][idx]}
            if "gold_answers" in batch:
                output_row["gold_answer"] = batch["gold_answers"][idx]
            yield output_row
