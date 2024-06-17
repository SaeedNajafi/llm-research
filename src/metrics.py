"""This module implements different metrics used to evaluate the
predictions."""

from typing import Dict, List

import collections
import json
import os
import re
import string
import sys

import numpy as np
import pandas as pd
import torch
from absl import flags
from sentence_transformers import SentenceTransformer

from src.model_utils import clear_cache

FLAGS = flags.FLAGS
flags.DEFINE_string("metric_device", "cuda:0", "The device per node to calculate the metric.")
flags.DEFINE_integer("metric_batch_size", 16, "batch size used for the metric model.")


class QAMetricModel:
    """Load and cache a model used for evaluating generative text
    generation."""

    model_id = "sentence-transformers/sentence-t5-xxl"

    def __init__(
        self,
        device: str = "cuda:0",
        batch_size: int = 16,
    ) -> None:
        """Save the gpu device and construct the model and cache it."""
        self.device = device
        self.batch_size = batch_size
        self.metric_model = SentenceTransformer(self.model_id, device=self.device).eval()

    def compute_metric(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute the metric for the given predictions and multiple
        references."""
        all_scores = []
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

            all_scores.append(dot_products * score_collector)
        return (
            torch.stack(all_scores, dim=0)
            .squeeze()
            .reshape(
                -1,
            )
        )


qa_metric_model = None


def postprocess_label(label: str) -> str:
    label = label.removesuffix("</s>")
    label = label.removeprefix("<s>")
    label = label.strip()
    return label

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
    return label

def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

def white_space_fix(text):
    return " ".join(text.split())

def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

def lower(text):
    return text.lower()

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(postprocess_qa(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1_precision_recall(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return [int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)]
    if num_same == 0:
        return [0, 0, 0]
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return [f1, precision, recall]

def qa_metric(prediction_file: str) -> Dict[str, float]:
    """Compute the metric for the qa task."""
    global qa_metric_model
    if qa_metric_model is None:
        qa_metric_model = QAMetricModel(device=FLAGS.metric_device, batch_size=FLAGS.metric_batch_size)

    df = pd.read_csv(prediction_file, delimiter=",")

    gold_answers = [postprocess_qa(label) for label in df["gold_class"].tolist()]

    multiple_gold_answers = []
    for answer in gold_answers:
        multiple_gold_answers.append(answer.split("[<@>]"))

    return_metrics: Dict[str, float] = {}
    metrics = {
        "potential_class": "best_paraphrase_score",
        "original_potential_class": "original_input_score",
        "all_potential_class": "best_paraphrase+original_input_score",
    }

    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            predictions = [postprocess_qa(pred) for pred in df[metric_column].tolist()]
            scores = qa_metric_model.compute_metric(predictions, multiple_gold_answers)
            avg_score = torch.mean(scores, dim=0).item()
            return_metrics[metric] = avg_score

    return return_metrics


def compute_dev_results(predictions, gold_answers):
    # Read gold-data
    exact_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    for idx, answers in enumerate(gold_answers):
        per_row_answers = [a for a in answers if normalize_answer(a)]
        if per_row_answers == ["<no_answer>"]:
            per_row_answers = [""]
        a_pred = predictions[idx]
        if a_pred == "<no_answer>":
            a_pred = ""

        # Take max over all gold answers
        exact_scores.append(max(compute_exact(a, a_pred) for a in gold_answers))
        f1_scores.append(max(compute_f1_precision_recall(a, a_pred)[0] for a in gold_answers))
        precision_scores.append(max(compute_f1_precision_recall(a, a_pred)[1] for a in gold_answers))
        recall_scores.append(max(compute_f1_precision_recall(a, a_pred)[2] for a in gold_answers))

    total = len(exact_scores)
    scores = collections.OrderedDict(
        [
            ("exact", 100.0 * sum(exact_scores.values()) / total),
            ("f1", 100.0 * sum(f1_scores.values()) / total),
            ("precision", 100.0 * sum(precision_scores.values()) / total),
            ("recall", 100.0 * sum(recall_scores.values()) / total),
            ("total", total),
        ]
    )
    return scores


def squad_llama3_metric(prediction_file: str) -> Dict[str, float]:
    df = pd.read_csv(prediction_file, delimiter=",")
    gold_answers = [gold_answer.split("_@_") for gold_answer in df["gold_answer"].tolist()]
    predictions = df["potential_answer"].tolist()
    predictions_new = {}
    for pred in predictions:
        a_pred = pred
        a_pred = a_pred.removeprefix("assistant\n\n")
        a_pred = a_pred.removeprefix("The shortest continuous text span from the passage that serves as an answer to the given question is:\n\n")
        a_pred = a_pred.removeprefix("The shortest continuous text span from the passage that serves as an answer to the question is:\n\n")
        a_pred = a_pred.removeprefix("The shortest continuous text span that serves as an answer to the given question is:\n\n")
        a_pred = a_pred.removeprefix("Based on the passage, the correct answer is")
        a_pred = a_pred.removeprefix("The correct answer is")
        a_pred = a_pred.removeprefix("According to the passage,")
        a_pred = a_pred.removeprefix(":")
        a_pred = a_pred.removeprefix("Here is the answer:")
        a_pred = a_pred.removeprefix("the correct answer is")
        a_pred = white_space_fix(a_pred)
        try:
            a_pred = a_pred.split("Final Answer_11: ")[1]
        except:
            try:
                a_pred = a_pred.split("Answer: ")[1]
            except:
                a_pred = a_pred

        if "<no_answer>" in a_pred:
            a_pred = "<no_answer>"

        predictions_new.append(a_pred)
    