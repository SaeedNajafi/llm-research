"""This module implements the backbone LM and the paraphrasing module that
helps to train LM better."""

from typing import Any, Dict, List, Optional

import torch
from absl import flags
from bitsandbytes.optim.adamw import PagedAdamW8bit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.base_lm import BaseLM
from src.load_lm import load_model_and_tokenizer, load_peft_model
from src.paraphraser import Paraphraser

FLAGS = flags.FLAGS

flags.DEFINE_integer("lm_input_max_length", 1024, "Maximum length to use for lm on the input side only.")
flags.DEFINE_integer("lm_output_max_length", 128, "Maximum length to use for lm on the output side only.")

flags.DEFINE_string("pretrained_model", "/model-weights/Llama-2-7b-chat-hf", "initial pre-trained model to use as backbone LM.")
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
flags.DEFINE_float("learning_rate", 0.0001, "The learning rate used to train the main model", lower_bound=0.0)
flags.DEFINE_float(
    "kl_penalty_coefficient",
    0.1,
    "What is the coefficient for the KL penalty used in the ppo algorithm?",
)


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
    ) -> None:
        super().__init__(device, "main_lm")

        self.lm = FLAGS.lm_type
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
            if FLAGS.exp_type == "soft_prompt_finetune":
                peft_model = load_peft_model(
                    model=model,
                    num_quantized_bits=4,
                    adapter_name="soft_prompt_tuning",
                    is_trainable=FLAGS.mode == "train",
                    model_type="causal_lm",
                )

            if FLAGS.exp_type == "lora_finetune":
                peft_model = load_peft_model(
                    model=model,
                    num_quantized_bits=4,
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
            padding="max_length",
            max_length=FLAGS.lm_input_max_length + FLAGS.lm_output_max_length,
            add_special_tokens=False,
        )
        input_encodings_for_generation = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
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
