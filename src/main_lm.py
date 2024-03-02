"""This module implements the backbone LM and the paraphrasing module that
helps to train LM better."""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from absl import app, flags, logging
from bitsandbytes.optim.adamw import PagedAdamW8bit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader

from src.base_lm import BaseLM
from src.general_utils import DictDataset
from src.load_lm import load_model_and_tokenizer, load_peft_model
from src.model_utils import llama2_log_of_labels, lm_logits, mlm_log_of_labels
from src.paraphraser import Paraphraser

FLAGS = flags.FLAGS

flags.DEFINE_integer("lm_input_max_length", 1024, "Maximum length to use for lm on the input side only.")
flags.DEFINE_integer("lm_output_max_length", 128, "Maximum length to use for lm on the output side only.")

flags.DEFINE_string("pretrained_model", "meta-llama/Llama-2-13b-chat-hf", "initial pre-trained model to use as backbone LM.")
flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")
flags.DEFINE_string("checkpoint", None, "checkpoint name to load from.")

# details about the training process.
flags.DEFINE_string("ensemble_type", "no_ensemble", "ensemble type with the paraphraser.")
flags.DEFINE_string("paraphrase_loss", "pg_z_score", "the training objective used to train the paraphrase model.")
flags.DEFINE_string(
    "rl_sampling_method",
    "on_policy",
    "Whether to do on-policy sampling using the paraphrase model \
        or off-policy sampling using a separate paraphrase model, or include PPO KL while sampling.",
)
flags.DEFINE_float("learning_rate", 0.001, "The learning rate used to train the main model", lower_bound=0.0)
flags.DEFINE_float(
    "kl_penalty_coefficient",
    0.1,
    "What is the coefficient for the KL penalty used in the ppo algorithm?",
)

# Decoding hyper-parameter.
flags.DEFINE_float(
    "temperature",
    0.6,
    "Temperature used for the softmax to smooth or sharpen the token probabilities.",
)
flags.DEFINE_float("lm_top_p", 0.90, "The top_p value used in nucleus sampling for the main lm.")


class MainLM(BaseLM):
    """Wrapper class around the LM Model to experiment with different prompting
    ideas along with paraphrasing the inputs or training the paraphrase model
    with the feedback of the LM."""

    def __init__(
        self,
        device: str,
        enable_para_augmentation: int,
        enable_paraphrase_training: int,
        paraphrase_model: Optional[Paraphraser] = None,
        fixed_paraphrase_model: Optional[Paraphraser] = None,
        lm_type: str = "llama2",
        training_type: str = "lora_finetune",
    ) -> None:
        super().__init__(device, "main_lm")

        self.lm = lm_type
        self.training_type = training_type
        self.device = device
        self.paraphrase_model = paraphrase_model
        self.fixed_paraphrase_model = fixed_paraphrase_model

        if self.lm == "llama2":
            model, tokenizer = load_model_and_tokenizer(
                model_id=FLAGS.pretrained_model,
                model_type="causal_lm",
                model_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                load_in_4bit=True,
            )
            if self.training_type == "soft_prompt_finetune":
                peft_model = load_peft_model(
                    model=model,
                    adapter_name="soft_prompt_tuning",
                    is_trainable=FLAGS.mode == "train",
                    model_type="causal_lm",
                )

            if self.training_type == "lora_finetune":
                print(FLAGS.mode == "train")
                peft_model = load_peft_model(
                    model=model,
                    adapter_name="lora",
                    is_trainable=FLAGS.mode == "train",
                    model_type="causal_lm",
                )
        else:
            raise Exception(f"The following lm type {self.lm} has not been implemented!")

        self.model = peft_model
        self.tokenizer = tokenizer

        # to train the main lm, we update all of its parameters.
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=FLAGS.learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=FLAGS.learning_rate / 5.0)

        self.enable_para_augmentation = enable_para_augmentation
        self.enable_paraphrase_training = enable_paraphrase_training

        if FLAGS.mode == "train" and self.enable_paraphrase_training == 1:
            # for training with the paraphraser, we need average ensembling prediction
            # while evaluating the checkpoints on the dev data.
            FLAGS.ensemble_type = "paraphrase_predict"

    def prepare_text(self, texts: List[str], output_texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for training
        and inference."""
        inputs_for_training = [f"{texts[idx]} {output_texts[idx]}" for idx in range(len(texts))]
        input_encodings = self.tokenizer(
            inputs_for_training,
            truncation=True,
            padding=True,
            max_length=FLAGS.lm_input_max_length + FLAGS.lm_output_max_length,
            add_special_tokens=False,
        )
        input_encodings_for_generation = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=FLAGS.lm_input_max_length,
            add_special_tokens=False,
        )
        data = {
            "lm_input_texts": texts,
            "lm_output_texts": output_texts,
            "lm_input_ids_for_train": input_encodings.input_ids,
            "lm_attention_mask_for_train": input_encodings.attention_mask,
            "lm_input_ids_for_generation": input_encodings_for_generation.input_ids,
            "lm_attention_mask_for_generation": input_encodings_for_generation.attention_mask,
        }
        return data

    def llama2_train_pass(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the Llama2, run a forward computation over the batch, compute
        the log probability over the batch.

        This will be used for training.
        """
        self.train_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_train", "lm_attention_mask_for_train"])
        input_ids = loaded_batch["lm_input_ids_for_train"]
        attention_mask = loaded_batch["lm_attention_mask_for_train"]
        with torch.set_grad_enabled(True):
            logits = lm_logits(
                model=self.model,
                input_ids=input_ids,
                input_mask=attention_mask,
            )
            masked_labels = input_ids.masked_fill(input_ids == self.tokenizer.pad_token_id, -100)
            return llama2_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.loss_func)

    def llama2_generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """Using the Llama2, generate new text.

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
                top_p=FLAGS.lm_top_p,
                temperature=FLAGS.temperature,
                max_length=FLAGS.lm_input_max_length + FLAGS.lm_output_max_length,
                num_return_sequences=1,
                output_logits=True,
                return_dict_in_generate=True,
                use_cache=True,
                renormalize_logits=True,
            )

        prompt_len = input_ids.size()[1]
        selected_samples = predictions_output.sequences[:, prompt_len:]
        # all special tokens will be removed.
        predictions_str = self.tokenizer.batch_decode(selected_samples, skip_special_tokens=True)
        predictions_str = [pred.lstrip('"').lstrip("'").rstrip("'").rstrip('"').strip() for pred in predictions_str]

        logits_list = list(predictions_output.logits)
        logits = torch.stack(logits_list, dim=1)
        labels_to_consider = selected_samples.masked_fill(selected_samples == self.tokenizer.pad_token_id, -100)
        final_log_ps = mlm_log_of_labels(logits=logits, labels=labels_to_consider, loss_func=self.loss_func)
        actual_lens = torch.sum(torch.where(labels_to_consider > 0, 1, 0), dim=1)
        # Average log probs per token (length normalization).
        return predictions_str, final_log_ps / actual_lens


def example_test_loop(model: MainLM) -> None:
    """Do a complete test of the model."""
    instruction = "In this task, you are given a context and question. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not know the answer from the context, generate '<no_result>' in the output. \
            Do not repeat the question in the output."
    template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
    input_texts = [
        "Question: What grade did Saeed get in the exam? Context: Saeed scores 80 in the math exam.",
        "Question: Who is the president of the USA? Context: Saeed is the president of the United States of America.",
        "Question: Who got the score A? Context: Saeed is the president of the United States of America.",
    ]
    input_texts = [template.format(instruction=instruction, input_text=txt) for txt in input_texts]
    output_texts = ["Saeed got 80 in the math exam. </s>", "The president of the USA is Saeed. </s>", "I don't know </s>"]
    data = model.prepare_text(input_texts, output_texts)
    dataloader = DataLoader(DictDataset(data), batch_size=len(input_texts), shuffle=False)
    start_time = time.time()
    for data in dataloader:
        logging.info(data)

        logging.info("inference")
        answers, log_ps = model.llama2_generation_pass(data)
        logging.info(answers)
        logging.info(log_ps)

    end_time = time.time()
    logging.info(f"Time took: {end_time-start_time}")


def example_train_loop(model: MainLM) -> None:
    """Do a complete train of the model."""
    instruction = "In this task, you are given a context and question. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not know the answer from the context, generate '<no_result>' in the output. \
            Do not repeat the question in the output."
    template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
    input_texts = [
        "Question: What grade did Saeed get in the exam? Context: Saeed scores 80 in the math exam.",
        "Question: Who is the president of the USA? Context: Saeed is the president of the United States of America.",
        "Question: Who got the score A? Context: Saeed is the president of the United States of America.",
    ]
    input_texts = [template.format(instruction=instruction, input_text=txt) for txt in input_texts]
    output_texts = ["Saeed got 80 in the math exam. </s>", "The president of the USA is Saeed. </s>", "I don't know </s>"]
    data = model.prepare_text(input_texts, output_texts)
    dataloader = DataLoader(DictDataset(data), batch_size=len(input_texts), shuffle=True)

    epochs = 100
    for e in range(epochs):
        for data in dataloader:
            log_ps = model.llama2_train_pass(data)
            loss = -torch.mean(log_ps, dim=0)
            model.optimizer.zero_grad()
            loss.backward()
            logging.info(f"epoch_{e} loss_value:{loss.item()}")
            model.optimizer.step()
            model.scheduler.step(e)

    logging.info(f"last_lr before saving: {model.scheduler.get_last_lr()}")
    model.save_to_checkpoint("/tmp", "testing_stage")

    del model

    FLAGS.mode = "test"
    model = MainLM(device="cuda:0", enable_para_augmentation=0, enable_paraphrase_training=0)
    model.load_from_checkpoint("/tmp", "testing_stage", peft_load=True)
    model.to_device()
    for data in dataloader:
        logging.info(data)

        logging.info("inference")
        answers, log_ps = model.llama2_generation_pass(data)
        logging.info(answers)
        logging.info(log_ps)

    del model


def main(argv: Any) -> None:
    """Example function to launch the train and generate functions."""
    del argv

    logging.info("Testing the model on gpu!")
    FLAGS.mode = "test"
    model = MainLM(device="cuda:0", enable_para_augmentation=0, enable_paraphrase_training=0)
    model.to_device()
    example_test_loop(model)
    del model

    logging.info("Training the model on gpu!")
    FLAGS.mode = "train"
    model = MainLM(device="cuda:0", enable_para_augmentation=0, enable_paraphrase_training=0)
    model.to_device()
    example_train_loop(model)


if __name__ == "__main__":
    app.run(main)
