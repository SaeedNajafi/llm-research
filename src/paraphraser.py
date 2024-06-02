"""This module will implement the paraphrase generator."""

from typing import Any, Dict, List, Tuple

import torch
from bitsandbytes.optim.adamw import PagedAdamW8bit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.base_lm import BaseLM
from src.cache import LruCache
from src.general_utils import white_space_fix
from src.model_utils import encoder_decoder_log_of_labels, mlm_log_of_labels

_PARAPHRASE_MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"


class Paraphraser(BaseLM):
    """Main class to load a paraphraser or train it."""

    def __init__(
        self,
        device: str,
        seed: int = 42,
        paraphrase_cache_capacity: int = 100000,
        diverse_beam_temperature: float = 0.7,
        diversity_penalty: float = 3.0,
        repetition_penalty: float = 10.0,
        paraphrase_top_p: float = 0.99,
        paraphrase_generation_max_length: int = 1024,
        no_repeat_ngram_size: int = 2,
        para_checkpoint_name: str = "last",
        para_model_path: str = "/tmp",
        paraphrase_learning_rate: float = 0.00005,
    ) -> None:
        super().__init__(device, model_name="paraphraser", seed=seed)

        self.paraphrase_cache_capacity = paraphrase_cache_capacity
        self.diverse_beam_temperature = diverse_beam_temperature
        self.diversity_penalty = diversity_penalty
        self.repetition_penalty = repetition_penalty
        self.paraphrase_top_p = paraphrase_top_p
        self.paraphrase_generation_max_length = paraphrase_generation_max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.para_checkpoint_name = para_checkpoint_name
        self.para_model_path = para_model_path
        self.paraphrase_learning_rate = paraphrase_learning_rate

        self.tokenizer = AutoTokenizer.from_pretrained(_PARAPHRASE_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(_PARAPHRASE_MODEL_NAME)
        self.cache: LruCache = LruCache(capacity=self.paraphrase_cache_capacity, filename=f"{self.para_model_path}/cache.bin")

        # to train the paraphraser, we update all of its parameters.
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=self.paraphrase_learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=self.paraphrase_learning_rate / 5.0)

    def save_to_checkpoint(self, para_model_path: str, para_checkpoint_name: str) -> None:
        """Save the model components to the disk."""
        super().save_to_checkpoint(para_model_path, para_checkpoint_name)
        # Save cache information.
        self.cache.save()

    def load_from_checkpoint(
        self, para_model_path: str, para_checkpoint_name: str, peft_load: bool = False, is_trainable: bool = False
    ) -> None:
        """Load the model components from the disk."""
        super().load_from_checkpoint(para_model_path, para_checkpoint_name, peft_load=peft_load)
        self.cache.filename = f"{para_model_path}/cache.bin"
        self.cache.load()

    def to_device(self) -> None:
        """Move the required modules to the gpu on the given device."""
        super().to_device()
        self.cache.load_to_device(self.device)

    def convert_to_ids(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Required format for the paraphrase generator model.

        Return ids and masks.
        """
        modified_texts = [white_space_fix(f"{text} </s>") for text in texts]
        encodings = self.tokenizer(
            modified_texts,
            truncation=True,
            padding=True,
            max_length=self.paraphrase_generation_max_length,
            add_special_tokens=False,
        )
        return encodings.input_ids, encodings.attention_mask

    def prepare_text_for_generation(self, texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for generation
        only."""
        ids, mask = self.convert_to_ids(texts)
        data = {"para_input_ids": ids, "para_attention_mask": mask, "para_input_texts": texts}
        return data

    def prepare_text_for_training(self, texts: List[str], output_texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for
        training."""
        input_ids, input_mask = self.convert_to_ids(texts)
        output_ids, output_mask = self.convert_to_ids(output_texts)
        data = {
            "para_input_texts": texts,
            "para_output_texts": output_texts,
            "para_input_ids": input_ids,
            "para_attention_mask": input_mask,
            "para_labels": output_ids,
            "para_target_attention_mask": output_mask,
        }
        return data

    def decode_paraphrases_atomic(
        self,
        para_input_ids: torch.Tensor,
        para_attention_mask: torch.Tensor,
        num_return_seq: int,
        decoding_technique: str,
        temperature: float = 1.0,
    ) -> Tuple[List[str], torch.Tensor]:
        """The main prediction loop to generate paraphrases."""
        self.predict_mode_on()
        if decoding_technique == "diverse_beam_search":
            predictions_output = self.model.generate(
                input_ids=para_input_ids,
                attention_mask=para_attention_mask,
                do_sample=False,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                num_beams=num_return_seq,
                num_beam_groups=num_return_seq,
                early_stopping=True,
                max_length=self.paraphrase_generation_max_length,
                num_return_sequences=num_return_seq,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                repetition_penalty=self.repetition_penalty,
                diversity_penalty=self.diversity_penalty,
                temperature=self.diverse_beam_temperature,
                renormalize_logits=True,
            )

        elif decoding_technique == "top_p":
            predictions_output = self.model.generate(
                input_ids=para_input_ids,
                attention_mask=para_attention_mask,
                do_sample=True,
                top_p=self.paraphrase_top_p,
                temperature=temperature,
                max_length=self.paraphrase_generation_max_length,
                num_return_sequences=num_return_seq,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                renormalize_logits=True,
            )

        selected_samples = predictions_output.sequences
        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        predictions_str = [pred.lstrip('"').lstrip("'").rstrip("'").rstrip('"').strip() for pred in predictions_str]

        logits_list = list(predictions_output.logits)
        logits = torch.stack(logits_list, dim=1)
        ignore_first_token_samples = selected_samples[:, 1:]
        labels_to_consider = ignore_first_token_samples.masked_fill(
            ignore_first_token_samples == self.tokenizer.pad_token_id, -100
        )
        final_log_ps = mlm_log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens

    def decode_paraphrases(
        self,
        para_input_ids: torch.Tensor,
        para_attention_mask: torch.Tensor,
        num_return_seq: int,
        decoding_technique: str,
        temperature: float = 1.0,
    ) -> Tuple[List[str], torch.Tensor]:
        """The main prediction loop to generate paraphrases."""
        if decoding_technique in ["diverse_beam_search", "top_p"]:
            return self.decode_paraphrases_atomic(
                para_input_ids=para_input_ids,
                para_attention_mask=para_attention_mask,
                num_return_seq=num_return_seq,
                decoding_technique=decoding_technique,
                temperature=temperature,
            )
        elif decoding_technique == "mixed":
            if num_return_seq == 1:
                raise Exception("Mixed decoding requires num_return_seq > 1.")
            beam_paraphrases, beam_log_ps = self.decode_paraphrases_atomic(
                para_input_ids=para_input_ids,
                para_attention_mask=para_attention_mask,
                num_return_seq=num_return_seq,
                decoding_technique="diverse_beam_search",
                temperature=temperature,
            )
            top_p_paraphrases, top_p_log_ps = self.decode_paraphrases_atomic(
                para_input_ids=para_input_ids,
                para_attention_mask=para_attention_mask,
                num_return_seq=num_return_seq,
                decoding_technique="top_p",
                temperature=temperature,
            )

            batch_size = len(beam_paraphrases) // num_return_seq

            # the array to hold the mixed samples from beam-search and top-p sampling.
            mixed_paraphrases = []
            mixed_log_ps = []
            for idx in range(batch_size):
                top_p_arr = top_p_paraphrases[idx * num_return_seq : (idx + 1) * num_return_seq]
                beam_arr = beam_paraphrases[idx * num_return_seq : (idx + 1) * num_return_seq]
                top_p_log_ps_arr = top_p_log_ps[idx * num_return_seq : (idx + 1) * num_return_seq]
                beam_log_ps_arr = beam_log_ps[idx * num_return_seq : (idx + 1) * num_return_seq]
                mixed_paraphrases.extend(top_p_arr[: num_return_seq // 2])
                mixed_paraphrases.extend(beam_arr[: num_return_seq // 2])
                mixed_log_ps.extend(top_p_log_ps_arr[: num_return_seq // 2])
                mixed_log_ps.extend(beam_log_ps_arr[: num_return_seq // 2])
            return mixed_paraphrases, torch.stack(mixed_log_ps, dim=0)
        return [], None

    def generate_paraphrases(
        self,
        batch: torch.utils.data.Dataset,
        num_return_seq: int,
        decoding_technique: str,
        temperature: float = 1.0,
        use_internal_cache: bool = False,
    ) -> Tuple[List[str], torch.Tensor]:
        """Generate paraphrases.

        Use cache if found and if requested.
        """
        loaded_batch = self.data_to_device(batch, keys=["para_input_ids", "para_attention_mask"])
        if use_internal_cache:
            batch_size = batch["para_input_ids"].size()[0]
            paraphrases_indices: Dict[int, Tuple[List[str], torch.Tensor]] = {}
            missed_indices = []
            for idx, para_input_text in enumerate(batch["para_input_texts"]):
                value = self.cache.get(para_input_text)
                if value is not None:
                    # This is a hit.
                    paraphrases_indices[idx] = value
                else:
                    missed_indices.append(idx)
            if len(missed_indices) > 0:
                missed_indices_tensor = torch.tensor(missed_indices, device=self.device)
                missed_para_input_ids = torch.index_select(loaded_batch["para_input_ids"], 0, missed_indices_tensor)
                missed_para_attention_mask = torch.index_select(loaded_batch["para_attention_mask"], 0, missed_indices_tensor)
                missed_paraphrases, missed_log_ps = self.decode_paraphrases(
                    para_input_ids=missed_para_input_ids,
                    para_attention_mask=missed_para_attention_mask,
                    num_return_seq=num_return_seq,
                    decoding_technique=decoding_technique,
                    temperature=temperature,
                )

                # Insert into cache.
                for missed_idx in missed_indices:
                    new_paraphrases = missed_paraphrases[missed_idx * num_return_seq : (missed_idx + 1) * num_return_seq]
                    new_log_p = missed_log_ps[missed_idx * num_return_seq : (missed_idx + 1) * num_return_seq]
                    paraphrases_indices[missed_idx] = (new_paraphrases, new_log_p)
                    self.cache.insert(key=batch["para_input_texts"][missed_idx], value=paraphrases_indices[missed_idx])

            paraphrases = []
            log_ps = []
            for idx in range(batch_size):
                paraphrases.extend(paraphrases_indices[idx][0])
                log_ps.extend(paraphrases_indices[idx][1])

            return paraphrases, torch.stack(log_ps, dim=0)

        else:
            return self.decode_paraphrases(
                para_input_ids=loaded_batch["para_input_ids"],
                para_attention_mask=loaded_batch["para_attention_mask"],
                num_return_seq=num_return_seq,
                decoding_technique=decoding_technique,
                temperature=temperature,
            )

    def paraphrase_forward_pass(self, batch: torch.utils.data.Dataset, train: bool = False) -> torch.Tensor:
        """Run a forward computation over the batch, compute the log
        probability over the batch.

        This function is used while training the paraphraser.
        """
        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        loaded_batch = self.data_to_device(
            batch, keys=["para_input_ids", "para_attention_mask", "para_target_attention_mask", "para_labels"]
        )
        # keep an internal link to the loaded batch on gpu or cpu.
        self.loaded_batch = loaded_batch

        # we have to make sure that the PAD token is ignored.
        # huggingface ignores a pad token if the token is -100!
        orig_labels = loaded_batch["para_labels"]
        labels = orig_labels.masked_fill(orig_labels == self.tokenizer.pad_token_id, -100)

        with torch.set_grad_enabled(train):
            class_log_p = encoder_decoder_log_of_labels(
                model=self.model,
                input_ids=loaded_batch["para_input_ids"],
                input_mask=loaded_batch["para_attention_mask"],
                decoder_mask=loaded_batch["para_target_attention_mask"],
                labels=labels,
                loss_func=self.loss_func,
            )

        return class_log_p
