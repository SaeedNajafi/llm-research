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

from src.base_lm import BaseLM
from src.general_utils import DictDataset, test_loop, train_loop
from src.model_utils import clear_cache, llama2_log_of_labels, lm_logits, mlm_log_of_labels, set_random_seed
from src.general_utils import white_space_fix

train_batch_size = 4
eval_batch_size = 4
lm_input_max_length = 1024
lm_output_max_length = 128
lm_top_p = 0.9
temperature = 0.6
metric_device = "cuda:1"
metric_batch_size = 8
learning_rate = 0.00005
train_file_name = "128-shot-datasets/squad/128-42-train.tsv"
dev_file_name = "128-shot-datasets/squad/128-42-dev.tsv"
test_file_name = "128-shot-datasets/squad/test.tsv"

# folder to store models and predictions.
model_path = "/scratch/ssd004/scratch/snajafi/checkpoints/llama2-lora"

# related to lora
r = 16
lora_alpha = 8
lora_dropout = 0.05

"""Load LM efficiently."""

# Make sure we have some tokens defined for the LM, if not defined in the model.
_EXTRA_TOKENS = {
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<s>",
    "eos_token": "</s>",
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

class Llama2QA(BaseLM):
    """Class to implement Llama2 for QA task."""

    def __init__(
        self,
        mode: str,
        device: str,
        seed: int = 42,
    ) -> None:
        super().__init__(device, "main_lm", seed)
        self.device = device
        model, tokenizer = load_model_and_tokenizer(
            model_id="/model-weights/Llama-2-7b-chat-hf",
            model_type="causal_lm",
            model_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            load_in_4bit=True,
        )
        peft_model = load_peft_model(
            model=model,
            adapter_name="lora",
            is_trainable=mode == "train",
            model_type="causal_lm",
        )
        self.model = peft_model
        self.tokenizer = tokenizer
        '''
        # to train the main lm, we update all of its parameters.
        galore_params = []
        target_modules_list = ["attn", "mlp"]
        for module_name, module in self.model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            if not any(target_key in module_name for target_key in target_modules_list):
                continue
            print('enable GaLore for weights in module: ', module_name)
            galore_params.append(module.weight)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in self.model.parameters() if id(p) not in id_galore_params]
        # then call galore_adamw
        param_groups = [{'params': regular_params}, 
                        {'params': galore_params, 'rank': 128, 'update_proj_gap': 16, 'scale': 0.25, 'proj_type': 'std'}]
        self.optimizer = GaLoreAdamW8bit(param_groups, lr=learning_rate)
        '''
        self.optimizer = PagedAdamW8bit(self.model.parameters(), lr=learning_rate)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=learning_rate / 5.0)

    def prepare_text(self, texts: List[str], output_texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for training
        and inference."""
        instruction = "In this task, you are given a context and question. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not know the answer from the context, generate 'no_answer' in the output. \
            Do not repeat the question in the output."
        template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
        # sample of the answers if possible.
        sampled_answers = [random.choice(text.split("[<@>]")) for text in output_texts]
        answers = [f"Answer: {answer}" for answer in sampled_answers]

        inputs_for_training = [
            white_space_fix(f"{template.format(instruction=instruction, input_text=texts[idx])} {answers[idx]}") for idx in range(len(texts))
        ]
        inputs_for_generation = [white_space_fix(template.format(instruction=instruction, input_text=texts[idx])) for idx in range(len(texts))]

        input_encodings = self.tokenizer(
            inputs_for_training,
            truncation=True,
            padding=True,
            max_length=lm_input_max_length + lm_output_max_length,
            add_special_tokens=False,
        )
        input_encodings_for_generation = self.tokenizer(
            inputs_for_generation,
            truncation=True,
            padding=True,
            max_length=lm_input_max_length,
            add_special_tokens=False,
        )
        data = {
            "output_texts": output_texts,
            "lm_input_ids_for_train": input_encodings.input_ids,
            "lm_attention_mask_for_train": input_encodings.attention_mask,
            "lm_input_ids_for_generation": input_encodings_for_generation.input_ids,
            "lm_attention_mask_for_generation": input_encodings_for_generation.attention_mask,
        }
        return data

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the Llama2, run a forward computation over the batch, compute
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
        answers, log_ps = self.generation_pass(batch)
        log_ps = log_ps.cpu().detach().numpy()
        for idx, answer in enumerate(answers):
            output_row = {
                "potential_answer": answer,
                "prediction_score": log_ps[idx],
                "gold_answer": batch["output_texts"][idx],
            }
            yield output_row
            
def read_gen_fewshot_file(file_path: str) -> Tuple[List[str], List[str]]:
    """Load the fewshot files for QA task."""
    df = pd.read_csv(file_path, sep="\t")
    input_texts = df.article.tolist()
    output_texts = df.answer.tolist()
    return input_texts, output_texts


def create_dataloader(
    model: Llama2QA,
    train_file_name: Optional[str] = None,
    dev_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
) -> DataLoader:
    """Function to create the required dataloader to train the LM models."""
    if train_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(train_file_name)
        shuffle = True
        batch_size = train_batch_size

    if dev_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(dev_file_name)
        shuffle = False
        batch_size = eval_batch_size

    if test_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(test_file_name)
        shuffle = False
        batch_size = eval_batch_size

    data = model.prepare_text(input_texts, output_texts)
    dataset = DictDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def read_gen_fewshot_file(file_path: str) -> Tuple[List[str], List[str]]:
    """Load the fewshot files for QA task."""
    df = pd.read_csv(file_path, sep="\t")
    input_texts = df.article.tolist()
    output_texts = df.answer.tolist()
    return input_texts, output_texts


def create_dataloader(
    model: Llama2QA,
    train_file_name: Optional[str] = None,
    dev_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
) -> DataLoader:
    """Function to create the required dataloader to train the LM models."""
    if train_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(train_file_name)
        shuffle = True
        batch_size = train_batch_size

    if dev_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(dev_file_name)
        shuffle = False
        batch_size = eval_batch_size

    if test_file_name is not None:
        input_texts, output_texts = read_gen_fewshot_file(test_file_name)
        shuffle = False
        batch_size = eval_batch_size

    data = model.prepare_text(input_texts, output_texts)
    dataset = DictDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

class QAMetricModel:
    """Load and cache a model used for evaluating generative text
    generation."""

    model_id = "sentence-transformers/sentence-t5-xxl"

    def __init__(self, device: str = "cuda:0", batch_size: int = 16) -> None:
        """Save the gpu device and construct the model and cache it."""
        self.device = device
        self.batch_size = batch_size
        self.metric_model = SentenceTransformer(self.model_id, device=self.device).eval()

    def compute_metric(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute the metric for the given predictions and multiple
        references."""
        average_score = torch.tensor(0.0, device=self.device)
        num_chunks = max(len(predictions) // self.batch_size, 1)
        for chunk_i in range(num_chunks):
            clear_cache()

            if (chunk_i + 1) * self.batch_size <= len(predictions):
                predictions_sub_arr = predictions[chunk_i * self.batch_size : (chunk_i + 1) * self.batch_size]
                references_sub_arr = references[chunk_i * self.batch_size : (chunk_i + 1) * self.batch_size]
            else:
                predictions_sub_arr = predictions[chunk_i * self.batch_size :]
                references_sub_arr = references[chunk_i * self.batch_size :]

            # need to track multiple references.
            ref_sub_arr_len = [len(ref_sub_arr) for ref_sub_arr in references_sub_arr]
            references_sub_arr_flattened = []
            for ref_sub_arr in references_sub_arr:
                references_sub_arr_flattened.extend(ref_sub_arr)

            prediction_embeddings = self.metric_model.encode(
                predictions_sub_arr,
                show_progress_bar=False,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )

            references_embeddings = self.metric_model.encode(
                references_sub_arr_flattened,
                show_progress_bar=False,
                batch_size=self.batch_size,
                device=self.device,
                normalize_embeddings=True,
                convert_to_tensor=True,
            )
            dot_products = torch.matmul(prediction_embeddings, references_embeddings.t())
            score_collector = torch.zeros_like(dot_products)
            i = 0
            j = 0
            while i < len(predictions_sub_arr):
                j_len = ref_sub_arr_len[i]
                score_collector[i][j : j + j_len] = 1.0 / j_len
                i += 1
                j += j_len

            average_score += torch.sum(dot_products * score_collector)
        return (average_score / len(predictions)).item()


qa_metric_model = None


def postprocess_qa(label: str) -> str:
    label = str(label)
    label = label.lower()
    label = label.replace("\n", " ")
    label = label.removesuffix("</s>")
    label = label.removeprefix("<s>")
    label = label.removeprefix("\n")
    label = label.removesuffix("\n")
    label = label.removeprefix(".")
    label = label.removesuffix(".")
    label = label.removeprefix("answer:")
    label = label.removeprefix(",")
    label = label.strip()
    if "no answer" in label or "no_answer" in label:
        label = "no answer"
    return label


def qa_metric(prediction_file: str) -> Dict[str, float]:
    """Compute the metric for the qa task."""
    global qa_metric_model
    if qa_metric_model is None:
        qa_metric_model = QAMetricModel(device=metric_device, batch_size=metric_batch_size)

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_answers = [postprocess_qa(label) for label in df["gold_answer"].tolist()]

    multiple_gold_answers = []
    for answer in gold_answers:
        multiple_gold_answers.append(answer.split("[<@>]"))

    return_metrics: Dict[str, float] = {}
    metrics = {
        "potential_answer": "qa_score",
    }

    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            predictions = [postprocess_qa(pred) for pred in df[metric_column].tolist()]
            score = qa_metric_model.compute_metric(predictions, multiple_gold_answers)
            return_metrics[metric] = score

    return return_metrics

# Create model and start training.
set_random_seed(42)

model = Llama2QA(mode="train", device="cuda:0", seed=42)
model.to_device()
train_dataloader = create_dataloader(model, train_file_name=train_file_name)
dev_dataloader = create_dataloader(model, dev_file_name=dev_file_name)

"""
train_loop(
    model=model,
    mode="train",
    model_path=model_path,
    metric_to_save="qa_score",
    max_epochs=10,
    training_steps=100000,  # not important
    steps_per_checkpoint=16,
    metric=qa_metric,
    train_dataloader=train_dataloader,
    eval_dataloader=dev_dataloader,
)
"""

# Run on the Test Data.
model.load_from_checkpoint(model_path, "best_step", peft_load=True, is_trainable=False)
model.to_device()
test_dataloader = create_dataloader(model, test_file_name=test_file_name)
test_loop(
    model=model,
    mode="test",
    model_path=model_path,
    prediction_file_name="test.predicted.tsv",
    test_dataloader=test_dataloader,
    metric=qa_metric,
)