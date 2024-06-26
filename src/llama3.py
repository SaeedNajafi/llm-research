"""The main module to load llama3 with the chat prompt."""

from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch
from bitsandbytes.optim.adamw import AdamW8bit
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from src.configs import fsdp_config as FSDP_CONFIG
from src.configs import lora_config as LORA_CONFIG
from src.configs import train_config as TRAIN_CONFIG
from src.model_utils import llama_log_of_labels, lm_logits, log_of_labels
from src.policies import apply_fsdp_checkpointing
from src.utils import fsdp_auto_wrap_policy
from src.utils.fsdp_utils import hsdp_device_mesh
from src.utils.train_utils import clear_gpu_cache, freeze_transformer_layers, get_policies, print_model_size

# Make sure we have some tokens defined for the LM, if not defined in the model.
# Specific for Llama3
_EXTRA_TOKENS = {
    "pad_token": "<|reserved_special_token_0|>",
}


class Llama3(torch.nn.Module):
    """Class to implement Llama3."""

    def __init__(
        self,
        train_config: TRAIN_CONFIG,
        fsdp_config: FSDP_CONFIG,
        lora_config: LORA_CONFIG,
        local_rank: int = 0,
        rank: int = 0,
    ) -> None:
        super().__init__()

        self.train_config = train_config
        self.fsdp_config = fsdp_config
        self.lora_config = lora_config
        self.local_rank = local_rank
        self.rank = rank

        # Chat templates for llama3.
        self.llama3_instruction_llama = (
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{instruction} <|eot_id|>"
        )
        self.llama3_input_template = "<|start_header_id|>user<|end_header_id|>\n\n{input} <|eot_id|>"
        self.llama3_output_template = "<|start_header_id|>assistant<|end_header_id|>\n\n{output} <|eot_id|>"

        # We will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # Load the pre-trained model and setup its configuration
        use_cache = False if train_config.enable_fsdp else None
        if train_config.enable_fsdp and train_config.low_cpu_fsdp:
            """For FSDP, we can save cpu memory by loading pretrained model on
            rank0 only.

            this avoids cpu oom when loading large models like llama
            70B, in which case model alone would consume 2+TB cpu mem
            (70 * 4 * 8). This will add some comms overhead and
            currently requires latest nightly.
            """
            if self.rank == 0:
                model = LlamaForCausalLM.from_pretrained(
                    train_config.model_name,
                    load_in_8bit=True if train_config.quantization else None,
                    device_map="auto" if train_config.quantization else None,
                    use_cache=use_cache,
                    attn_implementation="flash_attention_2" if train_config.use_fast_kernels else None,
                )
            else:
                llama_config = LlamaConfig.from_pretrained(train_config.model_name)
                llama_config.use_cache = use_cache
                with torch.device("meta"):
                    model = LlamaForCausalLM(llama_config)

        else:
            model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_8bit=True if train_config.quantization else None,
                device_map="auto" if train_config.quantization else None,
                use_cache=use_cache,
                attn_implementation="flash_attention_2" if train_config.use_fast_kernels else None,
            )

        # Load the tokenizer and add special tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            train_config.model_name if train_config.tokenizer_name == "" else train_config.tokenizer_name, padding_side="left"
        )
        self.tokenizer.add_special_tokens(_EXTRA_TOKENS)

        # If there is a mismatch between tokenizer vocab size and embedding matrix,
        # throw a warning and then expand the embedding matrix
        if len(self.tokenizer) > model.get_input_embeddings().weight.shape[0]:
            print("WARNING: Resizing the embedding matrix to match the tokenizer vocab size.")
            if torch.cuda.is_available():
                # extend embeddings to a multiple so we use Tensor cores
                multiple = 64 if "A100" in torch.cuda.get_device_name() else 8
                model.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=multiple)
            else:
                raise Exception("No CUDA Found!")

        # re-define token ids for the model.
        for extra_token_key, extra_token_val in _EXTRA_TOKENS.items():
            extra_token_id = self.tokenizer.convert_tokens_to_ids([extra_token_val])[0]
            model.config.__setattr__(f"{extra_token_key}_id", extra_token_id)
            model.generation_config.__setattr__(f"{extra_token_key}_id", extra_token_id)

        # required for llama3.
        self.terminators = [self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]

        print_model_size(model, train_config, self.rank if train_config.enable_fsdp else 0)

        # Prepare the model for int8 training if quantization is enabled
        if train_config.quantization:
            model = prepare_model_for_kbit_training(model)

        # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
        if train_config.enable_fsdp and fsdp_config.pure_bf16:
            model.to(torch.bfloat16)

        if train_config.use_peft:
            # Load the pre-trained peft model checkpoint and setup its configuration
            if train_config.from_peft_checkpoint:
                model = PeftModel.from_pretrained(model, train_config.from_peft_checkpoint, is_trainable=True)
                self.peft_config = model.peft_config()
            # Generate the peft config and start fine-tuning from original model
            else:
                self.peft_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    r=lora_config.lora_r,
                    lora_alpha=lora_config.lora_alpha,
                    lora_dropout=lora_config.lora_dropout,
                    bias="none",
                    init_lora_weights=True,
                    target_modules=lora_config.lora_target_modules,
                )
                model = get_peft_model(model, self.peft_config)

            model.print_trainable_parameters()

        hsdp_device_mesh_plan = None
        if fsdp_config.hsdp and fsdp_config.sharding_strategy == ShardingStrategy.HYBRID_SHARD:
            hsdp_device_mesh_plan = hsdp_device_mesh(
                replica_group_size=fsdp_config.replica_group_size, sharding_group_size=fsdp_config.sharding_group_size
            )
            print("HSDP device mesh is ready")

        # Setting up FSDP if enable_fsdp is enabled
        if train_config.enable_fsdp:
            if not train_config.use_peft and train_config.freeze_layers:
                freeze_transformer_layers(model, train_config.num_freeze_layers)

            mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, self.rank)
            my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

            if torch.cuda.is_available():
                self.device_id = torch.cuda.current_device()

            self.loss_func.to(self.device_id)

            model = FSDP(
                model,
                auto_wrap_policy=my_auto_wrapping_policy if train_config.use_peft else wrapping_policy,
                cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
                mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
                sharding_strategy=fsdp_config.sharding_strategy,
                device_mesh=hsdp_device_mesh_plan,
                device_id=self.device_id,
                limit_all_gathers=True,
                sync_module_states=train_config.low_cpu_fsdp,
                param_init_fn=(
                    (lambda module: module.to_empty(device=torch.device("cuda"), recurse=False))
                    if train_config.low_cpu_fsdp and self.rank != 0
                    else None
                ),
            )
            if fsdp_config.fsdp_activation_checkpointing:
                apply_fsdp_checkpointing(model)

        elif not train_config.quantization and not train_config.enable_fsdp:
            if torch.cuda.is_available():
                model.to("cuda")
                self.loss_func.to("cuda")
                self.device_id = "cuda"
            else:
                self.device_id = "cpu"

        self.model = model
        self.optimizer = AdamW8bit(self.model.parameters(), lr=train_config.lr, weight_decay=train_config.weight_decay)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=train_config.T_0, eta_min=train_config.eta_min)

    def prepare_text_for_inference(
        self, texts: List[str], row_ids: List[str], gold_answers: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for
        inference."""
        input_encodings_for_generation = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.train_config.lm_input_max_length,
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
            max_length=self.train_config.lm_input_max_length + self.train_config.lm_output_max_length,
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
        clear_gpu_cache(self.local_rank)
        self.model.eval()

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode!"""
        clear_gpu_cache(self.local_rank)
        self.model.train()

    def data_to_device(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Move the batch tensors specified by keys into the gpu and return a
        dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device_id) for key in keys}

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
            return llama_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.loss_func)

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """Using the Llama, generate new text.

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
                top_p=self.train_config.lm_top_p,
                temperature=self.train_config.temperature,
                max_length=self.train_config.lm_input_max_length + self.train_config.lm_output_max_length,
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
        final_log_ps = log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Tuple[Dict[str, str], torch.Tensor]]:
        """The main prediction loop."""
        answers, log_ps = self.generation_pass(batch)
        loss = -torch.mean(log_ps, dim=0).detach().float()
        numpy_log_ps = log_ps.detach().cpu().numpy()
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
