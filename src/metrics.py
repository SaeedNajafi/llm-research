"""This module implements different metrics used to evaluate the predictions
for the downstream tasks."""

from typing import Dict, List

import pandas as pd
import torch
from absl import flags
from sentence_transformers import SentenceTransformer

from src.model_utils import clear_cache, set_random_seed

FLAGS = flags.FLAGS
flags.DEFINE_integer("metric_device_id", 3, "GPU device id per node to calculate the metric.")
flags.DEFINE_integer("metric_batch_size", 16, "batch size used for the metric model.")


class QAMetricModel:
    """Load and cache a model used for evaluating generative text
    generation."""

    model_id = "sentence-transformers/sentence-t5-xxl"

    def __init__(self, device_id: int = 0, batch_size: int = 16) -> None:
        """Save the gpu device and construct the model and cache it."""
        self.old_random_seed = FLAGS.seed
        self.old_rng = torch.random.get_rng_state()
        # set a unique random seed for the metric model.
        set_random_seed(len("QAMetricModel"))
        self.device = f"cuda:{device_id}"
        self.batch_size = batch_size
        self.metric_model = SentenceTransformer(self.model_id, device=self.device).eval()

    def compute_metric(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute the metric for the given predictions and multiple
        references."""
        average_score = torch.tensor(0.0)
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
    if "no answer" in label or "no_answer" in label:
        label = "no answer"
    return label


def qa_metric(prediction_file: str) -> Dict[str, float]:
    """Compute the metric for the qa task."""
    global qa_metric_model
    if qa_metric_model is None:
        qa_metric_model = QAMetricModel(device_id=FLAGS.metric_device_id, batch_size=FLAGS.metric_batch_size)

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
            score = qa_metric_model.compute_metric(predictions, multiple_gold_answers)
            return_metrics[metric] = score

    # Set back the random seed for the rest of models.
    set_random_seed(FLAGS.seed)
    torch.random.set_rng_state(qa_metric_model.old_rng)
    return return_metrics
