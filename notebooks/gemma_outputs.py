import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from bitsandbytes.optim.adamw import PagedAdamW8bit
from src.galore_torch import GaLoreAdamW8bit
from peft import LoraConfig, TaskType, get_peft_model
from sentence_transformers import SentenceTransformer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
import os, json, csv
import pandas as pd
from src.base_lm import BaseLM
from src.general_utils import DictDataset, test_loop, train_loop
from src.model_utils import clear_cache, llama2_log_of_labels, lm_logits, mlm_log_of_labels, set_random_seed
from src.general_utils import white_space_fix
import time
from absl import app, flags, logging

FLAGS = flags.FLAGS

flags.DEFINE_string("model_path", "/tmp", "The main directory to save or load the model or its artifacts.")
flags.DEFINE_string("input_csv", "a name", "the name of file to read data from.")
flags.DEFINE_string("output_csv", "a name", "the name of file to read data to.")


train_batch_size = 3
eval_batch_size = 3
lm_input_max_length = 1536
lm_output_max_length = 512
lm_top_p = 0.9
temperature = 0.5
metric_device = "cuda:1"
metric_batch_size = 8
learning_rate = 0.00005

# folder to store models and predictions.
model_path = "/scratch/ssd004/scratch/snajafi/checkpoints/gemma-prompt-recovery"

# related to lora
r = 16
lora_alpha = 8
lora_dropout = 0.05


"""Load LM efficiently."""

# Make sure we have some tokens defined for the LM, if not defined in the model.
_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "cls_token": "<CLS>",
}

target_modules = ["q_proj", "v_proj", "o_proj", "k_proj"]


def load_peft_model(
    model: PreTrainedModel,
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
    model_id: str, model_type: str,
    model_dtype: torch.dtype, attn_implementation: str, load_in_4bit: Optional[bool] = True
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

class Gemma(BaseLM):
    """Class to implement Gemma for generation tasks."""

    def __init__(
        self,
        mode: str,
        device: str,
        seed: int = 42,
    ) -> None:
        super().__init__(device, "main_lm", seed)
        self.device = device
        model, tokenizer = load_model_and_tokenizer(
            model_id="/model-weights/gemma-7b-it",
            model_type="causal_lm",
            model_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=False,
        )
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=learning_rate / 5.0)

    def prepare_text(self, texts: List[str], output_texts: List[str], instructions: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for training
        and inference."""
        input_texts = [f"{instructions[idx]} {texts[idx]}" for idx in range(len(instructions))]
        template = "<bos><start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model"
        inputs_for_training = [
            f"{template.format(user_input=input_texts[idx])}\n{output_texts[idx]}<end_of_turn>" for idx in range(len(input_texts))
        ]
        inputs_for_generation = [
            template.format(user_input=input_texts[idx]) for idx in range(len(input_texts))
        ]

        input_encodings = self.tokenizer(
            inputs_for_training,
            truncation=True,
            padding="max_length",
            max_length=lm_input_max_length + lm_output_max_length,
            add_special_tokens=False,
        )
        input_encodings_for_generation = self.tokenizer(
            inputs_for_generation,
            truncation=True,
            padding="max_length",
            max_length=lm_input_max_length,
            add_special_tokens=False,
        )
        data = {
            "lm_input_ids_for_train": input_encodings.input_ids,
            "lm_attention_mask_for_train": input_encodings.attention_mask,
            "lm_input_ids_for_generation": input_encodings_for_generation.input_ids,
            "lm_attention_mask_for_generation": input_encodings_for_generation.attention_mask,
        }
        return data

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the Gemma, run a forward computation over the batch, compute
        the log probability over the batch.

        This will be used for training.
        """
        self.train_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["lm_input_ids_for_train", "lm_attention_mask_for_train",
                                                        "lm_attention_mask_for_generation"])
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
            prompt_mask = torch.arange(seq_len, device=self.device).expand(batch_size, seq_len) < original_len_without_answer.unsqueeze(1)
            masked_labels = masked_labels.masked_fill(prompt_mask == 1, -100)
            return llama2_log_of_labels(logits=logits, labels=masked_labels, loss_func=self.loss_func)

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """Using the Gemma, generate new text.

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
                top_p=lm_top_p,
                temperature=temperature,
                max_length=lm_input_max_length + lm_output_max_length,
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

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop."""
        outputs, log_ps = self.generation_pass(batch)
        log_ps = log_ps.cpu().detach().numpy()
        for idx, output in enumerate(outputs):
            output_row = {
                "gemma_output": output,
                "gemma_logit": log_ps[idx],
            }
            yield output_row


def inference_main(argv: Any) -> None:
    """Example function to launch the train and generate functions."""
    del argv

    logging.info("Building the model on gpu!")
    # Create model and start training.
    set_random_seed(42)

    model = Gemma(mode="test", device="cuda:0", seed=42)
    model.to_device()

    # create the dataset for predicting new output from Gemma.
    dataframe = pd.read_csv(os.path.join(FLAGS.model_path, FLAGS.input_csv), sep=",")
    instructions = dataframe.instruction.tolist()
    inputs = dataframe.input.tolist()
    outputs = dataframe.output.tolist()

    data = model.prepare_text(inputs, outputs, instructions)
    dataset = DictDataset(data)
    test_dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    start_time = time.time()
    test_loop(
        model=model,
        mode="test",
        model_path=FLAGS.model_path,
        prediction_file_name=FLAGS.output_csv,
        test_dataloader=test_dataloader,
        metric=None,
    )
    end_time = time.time()
    logging(f"processed in {end_time - start_time}.")

    del model


def similarity_main(argv: Any) -> None:
    """Example function to find differences between text A and text B."""
    del argv

    logging.info("Building the model on gpu!")
    # Create model and start training.
    set_random_seed(42)

    model = Gemma(mode="test", device="cuda:0", seed=42)
    model.to_device()

    # create the dataset for predicting new output from Gemma.
    dataframe = pd.read_csv(os.path.join(FLAGS.model_path, FLAGS.input_csv), sep=",")
    inputs = dataframe.input.tolist()
    outputs = dataframe.output.tolist()
    instruction = "I will give you two pieces of text. Text B has been changed based on the text A and an instruction. I like to describe the changes in text B and then summarize those changes as an instruction. Generate a general instruction based on the changes in text B."
    inputs = [f"\nText A: {inputs[idx]}\nText B: {outputs[idx]}" for idx in range(len(inputs))]
    data = model.prepare_text(inputs, ["no output"] * len(inputs), [instruction] * len(inputs))
    dataset = DictDataset(data)
    test_dataloader = DataLoader(dataset, batch_size=eval_batch_size, shuffle=False)

    start_time = time.time()
    test_loop(
        model=model,
        mode="test",
        model_path=FLAGS.model_path,
        prediction_file_name=FLAGS.output_csv,
        test_dataloader=test_dataloader,
        metric=None,
    )
    end_time = time.time()
    logging(f"processed in {end_time - start_time}.")

    del model




if __name__ == "__main__":
    # app.run(inference_main)
    app.run(similarity_main)



