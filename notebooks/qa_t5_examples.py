import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.base_lm import BaseLM
from src.general_utils import DictDataset, test_loop, train_loop
from src.model_utils import clear_cache, encoder_decoder_log_of_labels, mlm_log_of_labels, set_random_seed

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
model_path = "/scratch/ssd004/scratch/snajafi/checkpoints/t5-base"

"""QA Model based on T5 base and without any optimizations for large-scale training."""


class T5BaseQA(BaseLM):
    """Class to implement T5-base for QA task."""

    def __init__(self, device: str, seed: int) -> None:
        super().__init__(device=device, model_name="t5_base", seed=seed)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("google/t5-base-lm-adapt")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/t5-base-lm-adapt")
        # to train the main lm, we update all of its parameters.
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.001)
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=10, eta_min=learning_rate / 10.0)

    def prepare_text(self, texts: List[str], output_texts: List[str]) -> Dict[str, Any]:
        """Convert texts to ids and return the dataset required for training
        and inference."""
        instruction = "In this task, you are given a context and question. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not know the answer from the context, generate 'no_answer' in the output. \
            Do not repeat the question in the output."
        inputs = [f"{instruction} {text}" for text in texts]
        # sample of the answers if possible.
        sampled_answers = [random.choice(text.split("[<@>]")) for text in output_texts]
        answers = [f"Answer: {answer}" for answer in sampled_answers]
        input_encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding=True,
            max_length=lm_input_max_length,
            add_special_tokens=False,
        )
        answer_encodings = self.tokenizer(
            answers,
            truncation=True,
            padding=True,
            max_length=lm_output_max_length,
            add_special_tokens=False,
        )
        data = {
            "input_ids": input_encodings.input_ids,
            "attention_mask": input_encodings.attention_mask,
            "labels": answer_encodings.input_ids,
            "target_attention_mask": answer_encodings.attention_mask,
            "input_texts": texts,
            "output_texts": output_texts,
        }
        return data

    def train(self, batch: torch.utils.data.Dataset) -> torch.Tensor:
        """Using the T5-base, run a forward computation over the batch, compute
        the log probability over the batch.

        This will be used for training.
        """
        self.train_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["input_ids", "attention_mask", "target_attention_mask", "labels"])
        orig_labels = loaded_batch["labels"]
        labels = orig_labels.masked_fill(orig_labels == self.tokenizer.pad_token_id, -100)
        with torch.set_grad_enabled(True):
            class_log_p = encoder_decoder_log_of_labels(
                model=self.model,
                input_ids=loaded_batch["input_ids"],
                input_mask=loaded_batch["attention_mask"],
                decoder_mask=loaded_batch["target_attention_mask"],
                labels=labels,
                loss_func=self.loss_func,
            )
        return class_log_p

    def generation_pass(self, batch: torch.utils.data.Dataset) -> Tuple[List[str], torch.Tensor]:
        """This will be used for inference."""
        self.predict_mode_on()
        loaded_batch = self.data_to_device(batch, keys=["input_ids", "attention_mask"])
        input_ids = loaded_batch["input_ids"]
        attention_mask = loaded_batch["attention_mask"]
        with torch.no_grad():
            # more look here:
            # https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L130
            predictions_output = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=True,
                top_p=lm_top_p,
                temperature=temperature,
                max_length=lm_output_max_length + lm_input_max_length,
                num_return_sequences=1,
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
    model: T5BaseQA,
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

model = T5BaseQA(device="cuda:0", seed=42)
model.to_device()
train_dataloader = create_dataloader(model, train_file_name=train_file_name)
dev_dataloader = create_dataloader(model, dev_file_name=dev_file_name)

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

# Run on the Test Data.
model.load_from_checkpoint(model_path, "best_step")
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