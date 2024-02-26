"""This module will implement the paraphrase generator.

This module implements different ideas for fine-tuning a backbone LM on
some downstream NLP datasets.
"""

from typing import Dict, List

import torch
from absl import flags
from bitsandbytes.optim.adamw import PagedAdamW8bit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model_utils import clear_cache, encoder_decoder_log_of_labels, set_random_seed

FLAGS = flags.FLAGS


flags.DEFINE_float("paraphrase_learning_rate", 0.00001, "The learning rate used to train the paraphrase model", lower_bound=0.0)
flags.DEFINE_string("para_model_path", "/tmp/", "The main directory to save or load the paraphrase model from.")
flags.DEFINE_string("para_checkpoint_name", "last", "The checkpoint name to load the paraphrase from.")
flags.DEFINE_integer("no_repeat_ngram_size", 2, "Related to generation with beam search.")
flags.DEFINE_integer("paraphrase_generation_max_length", 1024, "Maximum Length to use for paraphrase generation.")
flags.DEFINE_float("top_p", 0.99, "The top_p value used in nucleus sampling.")
flags.DEFINE_float("repetition_penalty", 10.0, "The penalty for repeating sequences in the diverse beam search algorithm.")
flags.DEFINE_float("diversity_penalty", 3.0, "The diversity penalty used in the diverse beam search algorithm.")
flags.DEFINE_float("diverse_beam_temperature", 0.7, "The temperature value used in diverse beam search.")
flags.DEFINE_integer("use_paraphrase_cache", 1, "Whether to use cache for the generated paraphrase samples.")

_PARAPHRASE_MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"


class Paraphraser(torch.nn.Module):
    """Main class to load a paraphraser or train it."""

    def __init__(self, seed: int, device: int, mode: str, fixed: bool = False) -> None:
        super().__init__(seed, device)

        set_random_seed(FLAGS.seed)
        self.device = f"cuda:{device}"
        self.tokenizer = AutoTokenizer.from_pretrained(_PARAPHRASE_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(_PARAPHRASE_MODEL_NAME, device=self.device)
        self.fixed = fixed

        # for some subclasses, we will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        self.loss_func = self.loss_func.to(self.device)

        if not self.fixed and mode == "train":
            # to train the paraphraser, we update all of its parameters.
            self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=FLAGS.paraphrase_learning_rate)
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=FLAGS.paraphrase_learning_rate / 10.0)

        elif not self.fixed and mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load_from_checkpoint(FLAGS.para_model_path, FLAGS.para_checkpoint_name)

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""
        clear_cache()
        # turn on eval mode which disables dropout.
        self.model.eval()

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode!"""
        clear_cache()
        # turn on training mode which enables dropout.
        self.model.train()

    def move_to_gpu(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """If gpu flag is set, move the batch tensors specified by keys into
        the gpu and return a dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    def generate_paraphrases(
        self, batch: torch.utils.data.Dataset, num_return_seq: int, decoding_technique: str, temperature: float = 1.0
    ) -> List[str]:
        """The main prediction loop to generate paraphrases."""
        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["para_input_ids", "para_attention_mask"])

        if decoding_technique == "diverse_beam_search":
            predictions_output = self.model.generate(
                input_ids=loaded_batch["para_input_ids"],
                attention_mask=loaded_batch["para_attention_mask"],
                no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
                num_beams=num_return_seq,
                num_beam_groups=num_return_seq,
                early_stopping=True,
                max_length=FLAGS.paraphrase_generation_max_length,
                num_return_sequences=num_return_seq,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
                repetition_penalty=FLAGS.repetition_penalty,
                diversity_penalty=FLAGS.diversity_penalty,
                temperature=FLAGS.diverse_beam_temperature,
            )

        elif decoding_technique == "top_p":
            predictions_output = self.model.generate(
                input_ids=loaded_batch["para_input_ids"],
                attention_mask=loaded_batch["para_attention_mask"],
                no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
                do_sample=True,
                top_p=FLAGS.top_p,
                temperature=temperature,
                max_length=FLAGS.paraphrase_generation_max_length,
                num_return_sequences=num_return_seq,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
            )

        selected_samples = predictions_output.sequences
        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        predictions_str = [pred.lstrip('"').lstrip("'").rstrip("'").rstrip('"').strip() for pred in predictions_str]
        return predictions_str

    def paraphrase_forward_pass(self, batch: torch.utils.data.Dataset, train: bool = False) -> torch.Tensor:
        """Run a forward computation over the batch, compute the log
        probability over the batch.

        This function is used while training the paraphraser.
        """
        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        loaded_batch = self.move_to_gpu(
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
