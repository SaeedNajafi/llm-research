"""This module will implement the paraphrase generator."""

from typing import Any, Dict, List, Tuple

import torch
from absl import app, flags, logging
from bitsandbytes.optim.adamw import PagedAdamW8bit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.checkpoint_utils import model_load, model_save
from src.general_utils import DictDataset, white_space_fix
from src.model_utils import clear_cache, encoder_decoder_log_of_labels, mlm_log_of_labels, optimizer_to, set_random_seed

FLAGS = flags.FLAGS


flags.DEFINE_float("paraphrase_learning_rate", 0.00005, "The learning rate used to train the paraphrase model", lower_bound=0.0)
flags.DEFINE_string("para_model_path", "/tmp/", "The main directory to save or load the paraphrase model from.")
flags.DEFINE_string("para_checkpoint_name", "last", "The checkpoint name to load the paraphrase from.")
flags.DEFINE_integer("no_repeat_ngram_size", 2, "Related to generation with beam search.")
flags.DEFINE_integer("paraphrase_generation_max_length", 1024, "Maximum length to use for paraphrase generation.")
flags.DEFINE_float("top_p", 0.99, "The top_p value used in nucleus sampling.")
flags.DEFINE_float("repetition_penalty", 10.0, "The penalty for repeating sequences in the diverse beam search algorithm.")
flags.DEFINE_float("diversity_penalty", 3.0, "The diversity penalty used in the diverse beam search algorithm.")
flags.DEFINE_float("diverse_beam_temperature", 0.7, "The temperature value used in diverse beam search.")
flags.DEFINE_integer("use_paraphrase_cache", 1, "Whether to use cache for the generated paraphrase samples.")

_PARAPHRASE_MODEL_NAME = "humarin/chatgpt_paraphraser_on_T5_base"


class Paraphraser(torch.nn.Module):
    """Main class to load a paraphraser or train it."""

    def __init__(self, device: str, fixed: bool = False) -> None:
        super().__init__()

        if not torch.cuda.is_available():
            raise Exception("CUDA is not available. Code has been tested for cuda GPUs.")

        set_random_seed(FLAGS.seed)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(_PARAPHRASE_MODEL_NAME)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(_PARAPHRASE_MODEL_NAME)
        self.fixed = fixed

        # for some subclasses, we will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

        # to train the paraphraser, we update all of its parameters.
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=FLAGS.paraphrase_learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=FLAGS.paraphrase_learning_rate / 5.0)

    def save_to_checkpoint(self, para_model_path: str, para_checkpoint_name: str) -> None:
        """Save the model components to the disk."""
        model_save(
            model=self.model,
            model_path=para_model_path,
            checkpoint_name=f"_paraphraser_{para_checkpoint_name}",
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def load_from_checkpoint(self, para_model_path: str, para_checkpoint_name: str) -> None:
        """Load the model components from the disk."""
        model, optimizer, scheduler = model_load(
            model=self.model,
            model_path=para_model_path,
            checkpoint_name=f"_paraphraser_{para_checkpoint_name}",
            peft_load=False,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

    def to_device(self) -> None:
        """Move the required modules to the gpu on the given device."""
        self.model.to(self.device)
        self.loss_func.to(self.device)
        optimizer_to(self.optimizer, self.device)

    def convert_to_ids(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Required format for the paraphrase generator model.

        Return ids and masks.
        """
        modified_texts = [white_space_fix(f"{text} </s>") for text in texts]
        encodings = self.tokenizer(
            modified_texts,
            truncation=True,
            padding="max_length",
            max_length=FLAGS.paraphrase_generation_max_length,
            add_special_tokens=False,
        )
        return encodings.input_ids, encodings.attention_mask

    def prepare_text_for_generation(self, texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for generation
        only."""
        ids, mask = self.convert_to_ids(texts)
        return {"para_input_ids": ids, "para_attention_mask": mask}

    def prepare_text_for_training(self, texts: List[str], output_texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for
        training."""
        input_ids, input_mask = self.convert_to_ids(texts)
        output_ids, output_mask = self.convert_to_ids(output_texts)
        data = {
            "para_input_ids": input_ids,
            "para_attention_mask": input_mask,
            "para_labels": output_ids,
            "para_target_attention_mask": output_mask,
        }
        return data

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

    def data_to_device(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """Move the batch tensors specified by keys into the gpu and return a
        dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    def generate_paraphrases(
        self, batch: torch.utils.data.Dataset, num_return_seq: int, decoding_technique: str, temperature: float = 1.0
    ) -> Tuple[List[str], torch.Tensor]:
        """The main prediction loop to generate paraphrases."""
        self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["para_input_ids", "para_attention_mask"])

        if decoding_technique == "diverse_beam_search":
            predictions_output = self.model.generate(
                input_ids=loaded_batch["para_input_ids"],
                attention_mask=loaded_batch["para_attention_mask"],
                do_sample=False,
                no_repeat_ngram_size=FLAGS.no_repeat_ngram_size,
                num_beams=num_return_seq,
                num_beam_groups=num_return_seq,
                early_stopping=True,
                max_length=FLAGS.paraphrase_generation_max_length,
                num_return_sequences=num_return_seq,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                repetition_penalty=FLAGS.repetition_penalty,
                diversity_penalty=FLAGS.diversity_penalty,
                temperature=FLAGS.diverse_beam_temperature,
                renormalize_logits=True,
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


def example_test_train_loop(model: Paraphraser) -> None:
    """Do a complete test of the model."""
    text = ["Today seems to be a rainy day in Toronto, and I like it!", "I hate you bro."]
    data = model.prepare_text_for_generation(text)
    dataloader = DataLoader(DictDataset(data), batch_size=len(text), shuffle=False)
    for data in dataloader:
        logging.info(data)

        logging.info("diverse beam search testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)

        logging.info("top_p testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)

    # Test the training loop.
    output_text = ["Today seems to be a rainy day in Toronto", "I hate you!"]
    data = model.prepare_text_for_training(text, output_text)
    dataloader = DataLoader(DictDataset(data), batch_size=len(text), shuffle=True)
    epochs = 20
    for e in range(epochs):
        for data in dataloader:
            log_ps = model.paraphrase_forward_pass(data, train=True)
            loss = -torch.mean(log_ps, dim=0)
            model.optimizer.zero_grad()
            loss.backward()
            logging.info(f"epoch_{e} loss_value:{loss.item()}")
            model.optimizer.step()
            model.scheduler.step(e)

    for data in dataloader:
        logging.info("top_p testing.")
        paraphrases, log_ps = model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)

    logging.info(model.scheduler.get_last_lr())
    model.save_to_checkpoint("/tmp", "testing_stage")

    # Create a different model and load it.
    FLAGS.para_model_path = "/tmp"
    FLAGS.para_checkpoint_name = "testing_stage"
    new_model = Paraphraser(device=model.device)
    new_model.load_from_checkpoint(FLAGS.para_model_path, FLAGS.para_checkpoint_name)
    new_model.to_device()
    logging.info(new_model.scheduler.get_last_lr())

    for data in dataloader:
        logging.info("top_p testing.")
        paraphrases, log_ps = new_model.generate_paraphrases(data, num_return_seq=2, decoding_technique="diverse_beam_search")
        logging.info(paraphrases)
        logging.info(log_ps)

    for p1, p2 in zip(model.model.parameters(), new_model.model.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            logging.info(False)
            raise Exception("Not same models after saving and loading!")

    logging.info("Same models after saving and loading!")

    for e in range(10):
        for data in dataloader:
            log_ps = new_model.paraphrase_forward_pass(data, train=True)
            loss = -torch.mean(log_ps, dim=0)
            new_model.optimizer.zero_grad()
            loss.backward()
            logging.info(f"epoch_{e} loss_value:{loss.item()}")
            new_model.optimizer.step()
            new_model.scheduler.step(e + epochs)


def main(argv: Any) -> None:
    """Example function to launch the train and generate functions."""
    del argv

    logging.info("Testing the model on gpu!")
    model = Paraphraser(device="cuda:0")
    model.to_device()
    example_test_train_loop(model)
    del model


if __name__ == "__main__":
    app.run(main)
