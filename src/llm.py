"""The main module to train or make inference with llm."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from absl import flags, logging
from bitsandbytes.optim.adamw import AdamW8bit
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer

from src.utils.general_utils import clear_gpu_cache
from src.utils.model_utils import (
    decoder_only_log_of_labels,
    get_lora_model_from_base_model,
    lm_logits,
    load_model,
    log_of_labels,
    print_model_size,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_path", "/model-weights/gemma-2-9b-it", "/model-weights/Meta-Llama-3-8B-Instruct | /model-weights/gemma-2-9b-it"
)
flags.DEFINE_string("llm_name", "gemma2", "gemma2 | llama3")
flags.DEFINE_integer("t_0", 10, "number of epochs before resetting the learning rate with scheduler.")
flags.DEFINE_float("top_p", 0.9, "top_p value in nucleus sampling.", upper_bound=1.0, lower_bound=0.0)
flags.DEFINE_float("temperature", 0.01, "temperature value used in the softmax function.", lower_bound=0.0)
flags.DEFINE_integer("input_max_length", 1024, "max number of tokens for the input context.")
flags.DEFINE_integer("output_max_length", 256, "max number of tokens for the output context.")

flags.DEFINE_float("lr", 5e-5, "the initial learning rate.")
flags.DEFINE_float("lr_min", 1e-5, "the minimum learning rate for scheduler.")
flags.DEFINE_float("weight_decay", 0.001, "the weight decay for adam optimizer.")

# Make sure we have some tokens defined for the LM, if not defined in the model.
# Specific for Llama3
_LLAMA3_EXTRA_TOKENS = {
    "pad_token": "<|reserved_special_token_0|>",
}

_GPT2_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<s>",
    "eos_token": "</s>",
    "unk_token": "<unk>",
    "cls_token": "<cls>",
}


class LLM(torch.nn.Module):
    """Class to implement LLM."""

    def __init__(
        self,
        extra_tokens: Optional[Dict[str, str]] = None,
        local_rank: int = 0,
        rank: int = 0,
    ) -> None:
        super().__init__()

        self.rank = rank
        self.local_rank = local_rank
        self.terminators: List[int | List[int]] = []

        # We will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # Load the pre-trained model and setup its configuration

        model = load_model(FLAGS.model_path, local_rank=self.local_rank)

        # let fsdp handle this extra module to the devices.
        model.loss_func = loss_func

        # Load the tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_path, padding_side="left")

        if extra_tokens is not None:
            self.tokenizer.add_special_tokens(extra_tokens)

        # If there is a mismatch between tokenizer vocab size and embedding matrix,
        # throw a warning and then expand the embedding matrix
        if len(self.tokenizer) > model.get_input_embeddings().weight.shape[0]:
            logging.info("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            if torch.cuda.is_available():
                # extend embeddings to a multiple so we use Tensor cores
                multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
                model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=multiple)
            else:
                raise Exception("No CUDA Found!")

        if extra_tokens is not None:
            # re-define token ids for the model.
            for extra_token_key, extra_token_val in extra_tokens.items():
                extra_token_id = self.tokenizer.convert_tokens_to_ids([extra_token_val])[0]
                model.config.__setattr__(f"{extra_token_key}_id", extra_token_id)
                model.generation_config.__setattr__(f"{extra_token_key}_id", extra_token_id)

        print_model_size(model, FLAGS.model_path, self.rank)

        if FLAGS.use_peft:
            # Load the pre-trained peft model checkpoint and setup its configuration
            model = get_lora_model_from_base_model(model)
            self.peft_config = model.peft_config
            self.is_peft_adapter_restored = model.is_peft_adapter_restored

        model = model.to(torch.cuda.current_device())
        model = DDP(model, device_ids=[model.device])
        self.device = model.device
        self.model = model.module
        self.optimizer = AdamW8bit(self.model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=FLAGS.t_0, eta_min=FLAGS.lr_min)

    def prepare_text_for_inference(
        self, texts: List[str], row_ids: List[str], gold_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for
        inference."""
        input_encodings_for_generation = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=FLAGS.input_max_length,
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
            max_length=FLAGS.input_max_length + FLAGS.output_max_length,
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

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""
        clear_gpu_cache()
        self.model.eval()

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode!"""
        clear_gpu_cache()
        self.model.train()

    def data_to_device(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Move the batch tensors specified by keys into the gpu and return a
        dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the llm, run a forward computation over the batch, compute the
        log probability over the batch.

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
            return decoder_only_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.model.loss_func)

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """Using the llm, generate new text.

        This will be used for inference.
        """
        self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_generation", "lm_attention_mask_for_generation"])
        input_ids = loaded_batch["lm_input_ids_for_generation"]
        attention_mask = loaded_batch["lm_attention_mask_for_generation"]
        with torch.no_grad():
            predictions_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=FLAGS.top_p,
                temperature=FLAGS.temperature,
                max_length=FLAGS.input_max_length + FLAGS.output_max_length,
                num_return_sequences=1,
                output_logits=True,
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
        final_log_ps = log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.model.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Tuple[Dict[str, str], torch.Tensor]]:
        """The main prediction loop."""
        answers, log_ps = self.generation_pass(batch)
        loss = -torch.mean(log_ps, dim=0).detach().float()
        numpy_log_ps = log_ps.detach().float().cpu().numpy()
        for idx, answer in enumerate(answers):
            output_row = {
                "potential_answer": answer,
                "prediction_score": str(numpy_log_ps[idx]),
                "row_id": batch["row_ids"][idx],
            }
            if "gold_answers" in batch:
                # Somehow gold_answers becomes a tuple.
                output_row["gold_answer"] = batch["gold_answers"][idx]
            yield output_row, loss


class Llama3QA(LLM):
    """Class to implement Llama3."""

    def __init__(
        self,
        local_rank: int = 0,
        rank: int = 0,
    ) -> None:
        super().__init__(_LLAMA3_EXTRA_TOKENS, local_rank, rank)

        # Chat templates for llama3.
        self.instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
        self.input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
        self.output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output} <|eot_id|>"

        # required for llama3.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]


class Gemma2QA(LLM):
    """Class to implement Gemma2."""

    def __init__(
        self,
        local_rank: int = 0,
        rank: int = 0,
    ) -> None:
        super().__init__(None, local_rank, rank)

        # Chat templates for gemma2.
        self.instruction_template = "<bos><start_of_turn>user\n{instruction} <end_of_turn>"
        self.input_template = "<start_of_turn>user\n{input} <end_of_turn>"
        self.output_template = "<start_of_turn>model\n{output} <end_of_turn>"

        # required for gemma2.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]


class GPT2QA(LLM):
    """Class to implement GPT2."""

    def __init__(
        self,
        local_rank: int = 0,
        rank: int = 0,
    ) -> None:
        super().__init__(_GPT2_EXTRA_TOKENS, local_rank, rank)

        # Chat templates for gpt2.
        self.instruction_template = "<s>\ninstruction: {instruction} </s>"
        self.input_template = "<s>\nuser: {input} </s>"
        self.output_template = "<s>\nmodel: {output} </s>"

        # required for gemma2.
        self.terminators = [self.tokenizer.eos_token_id]
