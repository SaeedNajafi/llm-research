"""The main module to train or make inference with llm."""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from absl import flags, logging
from bitsandbytes.optim.adamw import AdamW8bit
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer

from src.utils.general_utils import clear_gpu_cache
from src.utils.model_utils import (
    decoder_only_log_of_labels,
    get_lora_model_from_base_model,
    get_submodule_by_pattern,
    lm_logits,
    load_model,
    log_of_labels,
    print_model_size,
    shard_model,
)

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "model_path", "/model-weights/gemma-2-9b-it", "/model-weights/Meta-Llama-3-8B-Instruct | /model-weights/gemma-2-9b-it"
)
flags.DEFINE_integer("t_0", 10, "number of epochs before resetting the learning rate with scheduler.")
flags.DEFINE_float("test_top_p", 0.9, "top_p value in nucleus sampling for inference.", upper_bound=1.0, lower_bound=0.0)
flags.DEFINE_float(
    "train_top_p", 0.95, "top_p value in nucleus sampling for training/sampling.", upper_bound=1.0, lower_bound=0.0
)
flags.DEFINE_float(
    "test_temperature", 0.0001, "temperature value used in the softmax function for prediction.", lower_bound=0.0
)
flags.DEFINE_float(
    "train_temperature",
    1.0,
    "temperature value used in the softmax function for drawing samples during training.",
    lower_bound=0.0,
)

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

# Make sure we have some tokens defined for the LM, if not defined in the model.
# Specific for Llama3.1
_LLAMA32_EXTRA_TOKENS = {
    "pad_token": "<|finetune_right_pad_id|>",
}


@dataclass
class LLMGenerationOutput:
    predictions_str: List[str] = field(default_factory=list)
    final_log_ps: torch.FloatTensor = None
    token_final_log_ps: torch.FloatTensor = None
    actual_lens: torch.LongTensor = None
    logits: torch.FloatTensor = None
    labels_to_consider: torch.FloatTensor = None
    partially_generated_sequences: List[List[str]] = field(default_factory=list)


class LLM(torch.nn.Module):
    """Class to implement LLM."""

    def __init__(self, extra_tokens: Optional[Dict[str, str]] = None, local_rank: int = 0, rank: int = 0) -> None:
        super().__init__()

        self.rank = rank
        self.local_rank = local_rank
        self.terminators: List[int | List[int]] = []

        # We will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # Load the pre-trained model and setup its configuration

        if FLAGS.ddp:
            self.distributed_strategy = "ddp"
        else:
            self.distributed_strategy = "fsdp"

        model = load_model(
            FLAGS.model_path,
            local_rank=self.local_rank,
            device=torch.cuda.current_device(),
            is_fsdp=self.distributed_strategy == "fsdp",
        )

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

        # Required while loading non-peft methods.
        self.is_peft_adapter_restored = False
        if FLAGS.use_peft:
            # Load the pre-trained peft model checkpoint and setup its configuration
            model = get_lora_model_from_base_model(model)
            self.peft_config = model.peft_config
            self.is_peft_adapter_restored = model.is_peft_adapter_restored

        if self.distributed_strategy == "ddp":
            model = DDP(model, device_ids=[model.device])
            self.model = model
            self.model.loss_func = self.model.module.loss_func
            self.model.device = self.model.module.device

        elif self.distributed_strategy == "fsdp":
            decoder_layer_module = get_submodule_by_pattern(model, r"DecoderLayer$")
            assert decoder_layer_module is not None, f"No DecoderLayer found in {model}"
            model = shard_model(model, decoder_layer_module, local_rank=local_rank)
            self.model = model

        self.device = self.model.device
        if self.distributed_strategy == "ddp":
            self.optimizer = AdamW8bit(self.model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)
        elif self.distributed_strategy == "fsdp":
            self.optimizer = AdamW(self.model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

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

    def prepare_text_for_train(
        self, texts: List[str], output_texts: List[str], row_ids: List[str], gold_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
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

        inference_data = self.prepare_text_for_inference(texts, row_ids, gold_answers=gold_answers)

        train_data = {
            "row_ids": inference_data.pop("row_ids"),
            "lm_input_ids_for_train": input_encodings.input_ids,
            "lm_attention_mask_for_train": input_encodings.attention_mask,
            "lm_input_ids_for_generation": inference_data.pop("lm_input_ids_for_generation"),
            "lm_attention_mask_for_generation": inference_data.pop("lm_attention_mask_for_generation"),
        }
        if gold_answers is not None:
            train_data["gold_answers"] = inference_data.pop("gold_answers")

        del inference_data
        return train_data

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""
        clear_gpu_cache()
        if self.distributed_strategy == "ddp":
            self.model.module.eval()
        elif self.distributed_strategy == "fsdp":
            self.model.eval()

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode!"""
        clear_gpu_cache()
        if self.distributed_strategy == "ddp":
            self.model.module.train()
        elif self.distributed_strategy == "fsdp":
            self.model.train()

    def data_to_device(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Move the batch tensors specified by keys into the gpu and return a
        dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    def train(self, batch: torch.utils.data.Dataset, per_step_scores: bool = False) -> torch.Tensor:
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
        original_len_without_answer = torch.sum(loaded_batch["lm_attention_mask_for_generation"], dim=1, keepdim=True)
        with torch.set_grad_enabled(True):
            logits = lm_logits(
                model=self.model,
                input_ids=input_ids,
                input_mask=attention_mask,
            )
            batch_size, seq_len = input_ids.size()
            masked_labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)

            if self.tokenizer.padding_side == "right":
                prompt_mask = (
                    torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < original_len_without_answer
                )

            elif self.tokenizer.padding_side == "left":
                left_pad_lens = torch.sum(torch.where(masked_labels == -100, 1, 0), dim=1, keepdim=True)
                original_len_with_pads = left_pad_lens + original_len_without_answer
                prompt_mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < original_len_with_pads

            masked_labels = masked_labels.masked_fill(prompt_mask == 1, -100)

            if per_step_scores:
                sequence_log_probs, token_log_probs = decoder_only_log_of_labels(
                    logits=logits, labels=masked_labels, loss_func=self.model.loss_func, per_step_scores=True
                )
                return sequence_log_probs, token_log_probs, logits, masked_labels
            else:
                sequence_log_probs = decoder_only_log_of_labels(
                    logits=logits, labels=masked_labels, loss_func=self.model.loss_func, per_step_scores=False
                )
                return sequence_log_probs

    def generation_pass(
        self,
        batch: torch.utils.data.Dataset,
        top_p: float = 0.9,
        temperature: float = 0.0001,
        num_return_sequences: int = 1,
        to_train: bool = False,
        use_cache: bool = True,
        per_step_scores: bool = False,
        iterative_rl_sampling: bool = False,
        generate_partial_sequences: bool = False,
    ) -> List[LLMGenerationOutput]:
        """Using the llm, generate new text.

        This will be used for inference.
        """
        if to_train:
            self.train_mode_on()
        else:
            self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_generation", "lm_attention_mask_for_generation"])
        input_ids = loaded_batch["lm_input_ids_for_generation"]
        attention_mask = loaded_batch["lm_attention_mask_for_generation"]
        with torch.set_grad_enabled(to_train):
            if self.distributed_strategy == "fsdp":
                # these weird line is necessary
                # https://github.com/pytorch/pytorch/issues/100069
                # with torch.no_grad():
                #    self.model.forward(input_ids=input_ids)
                with FSDP.summon_full_params(self.model, writeback=False, recurse=False):
                    results = list(
                        self.model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            do_sample=True,
                            top_p=top_p,
                            temperature=temperature,
                            max_length=FLAGS.input_max_length + FLAGS.output_max_length,
                            num_return_sequences=num_return_sequences,
                            output_logits=True,
                            return_dict_in_generate=True,
                            return_legacy_cache=use_cache,
                            use_cache=use_cache,
                            renormalize_logits=True,
                            eos_token_id=self.terminators,
                            pad_token_id=self.tokenizer.pad_token_id,
                            iterative_rl_sampling=iterative_rl_sampling,
                        )
                    )
            elif self.distributed_strategy == "ddp":
                results = list(
                    self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        do_sample=True,
                        top_p=top_p,
                        temperature=temperature,
                        max_length=FLAGS.input_max_length + FLAGS.output_max_length,
                        num_return_sequences=num_return_sequences,
                        output_logits=True,
                        return_dict_in_generate=True,
                        return_legacy_cache=use_cache,
                        use_cache=use_cache,
                        renormalize_logits=True,
                        eos_token_id=self.terminators,
                        pad_token_id=self.tokenizer.pad_token_id,
                        iterative_rl_sampling=iterative_rl_sampling,
                    )
                )

            outputs = []
            for result in results:
                predictions_output = result
                outputs.append(
                    self.find_log_information(predictions_output, input_ids, per_step_scores, generate_partial_sequences)
                )
            return outputs

    def find_log_information(
        self, predictions_output: Any, input_ids: torch.Tensor, per_step_scores: bool, generate_partial_sequences: bool
    ) -> LLMGenerationOutput:
        """Helper function to find generation logits and sequences."""
        prompt_len = input_ids.size()[1]
        selected_samples = predictions_output.sequences[:, prompt_len:]
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        if generate_partial_sequences:
            batch_size, seq_len = selected_samples.size()
            partial_sequences = []
            for b_index in range(batch_size):
                partial_sequences_per_batch = []
                for seq_index in range(seq_len):
                    if selected_samples[b_index, seq_index] == self.tokenizer.pad_token_id:
                        break
                    prefix = self.tokenizer.decode(selected_samples[b_index, 0 : seq_index + 1], skip_special_tokens=False)
                    partial_sequences_per_batch.append(prefix)
                partial_sequences.append(partial_sequences_per_batch)

        logits_list = list(predictions_output.logits)
        logits = torch.stack(logits_list, dim=1)
        labels_to_consider = selected_samples.masked_fill(selected_samples == self.tokenizer.pad_token_id, -100)
        if per_step_scores:
            final_log_ps, token_final_log_ps = log_of_labels(
                logits=logits, labels=labels_to_consider, loss_func=self.model.loss_func, per_step_scores=True
            )
            actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
            llm_generation_output = LLMGenerationOutput(
                predictions_str=predictions_str,
                final_log_ps=final_log_ps,
                token_final_log_ps=token_final_log_ps,
                actual_lens=actual_lens,
                logits=logits,
                labels_to_consider=labels_to_consider,
            )
        else:
            actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
            final_log_ps = log_of_labels(
                logits=logits, labels=labels_to_consider, loss_func=self.model.loss_func, per_step_scores=False
            )
            llm_generation_output = LLMGenerationOutput(
                predictions_str=predictions_str, final_log_ps=final_log_ps / actual_lens
            )

        if generate_partial_sequences:
            llm_generation_output.partially_generated_sequences = partial_sequences

        return llm_generation_output

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Tuple[Dict[str, str], torch.Tensor]]:
        """The main prediction loop."""
        llm_generation_outputs = self.generation_pass(
            batch,
            top_p=FLAGS.test_top_p,
            temperature=FLAGS.test_temperature,
            num_return_sequences=1,
            use_cache=True,
            per_step_scores=False,
        )
        answers = llm_generation_outputs[0].predictions_str
        log_ps = llm_generation_outputs[0].final_log_ps
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

    def __init__(self, local_rank: int = 0, rank: int = 0) -> None:
        super().__init__(_LLAMA3_EXTRA_TOKENS, local_rank, rank)

        # Chat templates for llama3.
        self.instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
        self.input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
        self.output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output} <|eot_id|>"

        # required for llama3.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]


class Llama32QA(LLM):
    """Class to implement Llama3.1."""

    def __init__(self, local_rank: int = 0, rank: int = 0) -> None:
        super().__init__(_LLAMA32_EXTRA_TOKENS, local_rank, rank)

        # Chat templates for llama3.2.
        self.instruction_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
        self.input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
        self.output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output} <|eot_id|>"

        # required for llama3.2
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]


class Gemma2QA(LLM):
    """Class to implement Gemma2."""

    def __init__(self, local_rank: int = 0, rank: int = 0) -> None:
        super().__init__(None, local_rank, rank)

        # Chat templates for gemma2.
        self.instruction_template = "<bos><start_of_turn>user\n{instruction}<end_of_turn>"
        self.input_template = "\n<start_of_turn>user\n{input}<end_of_turn>\n<start_of_turn>model"
        self.output_template = "\n{output} <end_of_turn>"

        # required for gemma2.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
