"""This module implements different metrics used to evaluate the
predictions.
Usage:
python src/metrics.py --metric_device=cuda:0 \
--metric_type llm2vec \
--input_file /scratch/ssd004/scratch/snajafi/gemma2-9b-predictions/squadv2_predictions_original_validation.normal_no_icl.csv
"""

import collections
import math
import re
import string
from typing import Any, Dict, List

import pandas as pd
import torch
from absl import app, flags

# from llm2vec import LLM2Vec
# from peft import PeftModel
from sentence_transformers import SentenceTransformer

from src.utils.general_utils import clear_gpu_cache

# from transformers import AutoConfig, AutoModel, AutoTokenizer


FLAGS = flags.FLAGS
flags.DEFINE_string("metric_device", "cuda:0", "The device per node to calculate the metric.")
flags.DEFINE_integer("metric_batch_size", 32, "batch size used for the metric model.")
flags.DEFINE_string("metric_type", "llm2vec", "llm2vec or sentence-t5 model?")
flags.DEFINE_string("input_file", "/path/filename", "absolute path and name of the file.")


# def load_llm2vec(cuda_device: str) -> LLM2Vec:
#     # Loading base llama-3-8b model, along with custom code that enables bidirectional connections in decoder-only
#     # LLMs. MNTP LoRA weights are merged into the base model.
#     tokenizer = AutoTokenizer.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
#     config = AutoConfig.from_pretrained("McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp", trust_remote_code=True)
#     model = AutoModel.from_pretrained(
#         "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
#         trust_remote_code=True,
#         config=config,
#         torch_dtype=torch.bfloat16,
#         device_map=cuda_device if torch.cuda.is_available() else "cpu",
#     )
#     model = PeftModel.from_pretrained(
#         model,
#         "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
#     )
#     model = model.merge_and_unload()  # This can take several minutes on cpu

#     # Loading supervised model. This loads the trained LoRA weights on top of MNTP model.
#     # Hence the final weights are -- Base model + MNTP (LoRA) + supervised (LoRA).
#     model = PeftModel.from_pretrained(model, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")

#     model = model.eval()

#     # Wrapper for encoding and pooling operations
#     l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=8192)

#     return l2v


class QAMetricModel:
    """Load and cache a model used for evaluating generative text
    generation."""

    sentence_t5_model_id = "sentence-transformers/sentence-t5-xxl"

    def __init__(self, device: str = "cuda:0", batch_size: int = 16, metric_type: str = "llm2vec") -> None:
        """Save the gpu device and construct the model and cache it."""
        self.device = device
        self.batch_size = batch_size
        self.metric_type = metric_type
        self.instruction = "Retrieve Wikipedia passages that answer the question."
        # if self.metric_type == "llm2vec":
        #    self.metric_model = load_llm2vec(self.device)
        # elif self.metric_type == "sentence_t5":
        if self.metric_type == "sentence_t5":
            self.metric_model = SentenceTransformer(self.sentence_t5_model_id, device=self.device).eval()

    def compute_metric(self, predictions: List[str], references: List[List[str]]) -> float:
        """Compute the metric for the given predictions and multiple
        references."""
        all_scores = []
        num_chunks = math.ceil(len(predictions) / self.batch_size)
        for chunk_i in range(num_chunks):
            clear_gpu_cache()

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

            if self.metric_type == "sentence_t5":
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
            elif self.metric_type == "llm2vec":
                predictions_sub_arr_with_queries = [[self.instruction, pred] for pred in predictions_sub_arr]
                prediction_embeddings_unnormal = self.metric_model.encode(predictions_sub_arr_with_queries)

                ref_sub_arr_flattened_with_queries = [[self.instruction, ref] for ref in references_sub_arr_flattened]
                references_embeddings_unnormal = self.metric_model.encode(ref_sub_arr_flattened_with_queries)
                prediction_embeddings = torch.nn.functional.normalize(prediction_embeddings_unnormal, p=2, dim=1)
                references_embeddings = torch.nn.functional.normalize(references_embeddings_unnormal, p=2, dim=1)

            dot_products = torch.matmul(prediction_embeddings, references_embeddings.t())
            score_collector = torch.zeros_like(dot_products)
            i = 0
            j = 0
            while i < len(predictions_sub_arr):
                j_len = ref_sub_arr_len[i]
                score_collector[i][j : j + j_len] = 1.0 / j_len
                i += 1
                j += j_len

            all_scores.append(torch.sum(dot_products * score_collector, dim=1))

        return torch.cat(tuple(all_scores), dim=0).squeeze()


qa_metric_model = None

regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)


def remove_articles(text: str) -> str:
    return re.sub(regex, " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


exclude = set(string.punctuation)


def remove_punc(text: str) -> str:
    return "".join(ch for ch in text if ch not in exclude)


def lower(text: str) -> str:
    return text.lower()


prefixes_txt_to_remove = [
    "assistant",
    "here are the answers based on the provided passages:",
    "the shortest continuous text span from the passage that serves as an answer to the given question is:",
    "the shortest continuous text span from the passage that serves as an answer to the question is:",
    "the shortest continuous text span that serves as an answer to the given question is:",
    "based on the passage, the correct answer is",
    "the correct answer is",
    "according to the passage,",
    "here is the answer:",
    "the correct answer is",
]

replace_with_space = [
    "final answer",
    "from passage it can be inferred that",
    "from the passage it can be inferred that",
    "let me know if you have any other passages you'd like me to analyze",
    "let me know if you have any other passages you would like me to analyze",
]

no_answer_indications = [
    "<no_answer>",
    "no_answer",
    "noanswer",
    "passage does not mention",
    "there is no mention",
    "there is no information",
    "passage does not include",
    "not explicitly mentioned in passage",
    "passage does not provide",
    "passage does not specify",
    "there is no specific information",
    "is not mentioned in passage",
    "passage does not indicate",
    "none are mentioned in passage",
]


def postprocess_qa(txt: str) -> str:
    txt = str(txt)
    txt = txt.lower()

    for indication in no_answer_indications:
        if indication in txt:
            txt = "this question is not answerable"
            return txt

    for prefix in prefixes_txt_to_remove:
        txt = txt.removeprefix(prefix)

    try:
        txt = txt.split("final answer:")[1]
    except Exception:
        try:
            txt = txt.split("answer:")[1]
        except Exception:
            try:
                txt = txt.split("**final answer:**")[1]
            except Exception:
                pass

    for to_replace in replace_with_space:
        txt = txt.replace(to_replace, "")
    txt = txt.strip()
    return txt


def normalize_answer(text: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    return white_space_fix(remove_articles(remove_punc(postprocess_qa(text))))


def get_tokens(text: str) -> List[str]:
    if not text:
        return []
    return normalize_answer(text).split()


def compute_exact(a_gold: str, a_pred: str) -> int:
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1_precision_recall(a_gold: str, a_pred: str) -> List[float]:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if gold_toks == ["this", "question", "is", "not", "answerable"] or pred_toks == [
        "this",
        "question",
        "is",
        "not",
        "answerable",
    ]:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return [int(gold_toks == pred_toks), int(gold_toks == pred_toks), int(gold_toks == pred_toks)]
    if num_same == 0:
        return [0, 0, 0]
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return [f1, precision, recall]


def qa_metric_sentence_similarity(prediction_file: str) -> Dict[str, float]:
    """Compute the metric for the qa task."""
    global qa_metric_model
    if qa_metric_model is None:
        qa_metric_model = QAMetricModel(
            device=FLAGS.metric_device, batch_size=FLAGS.metric_batch_size, metric_type=FLAGS.metric_type
        )

    df = pd.read_csv(prediction_file, delimiter=",")
    gold_answers = [[normalize_answer(str(ans)) for ans in str(answers).split("_@_")] for answers in df["gold_answer"].tolist()]
    return_metrics: Dict[str, float] = {}
    metrics = {"potential_answer": "sentence_similarity"}
    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            predictions = [normalize_answer(pred) for pred in df[metric_column].tolist()]
            scores = qa_metric_model.compute_metric(predictions, gold_answers)
            avg_score = torch.mean(scores, dim=0).item()
            return_metrics[metric] = avg_score

    return return_metrics


def qa_metric_squadv2_metrics(prediction_file: str) -> Dict[str, float]:
    # Read gold-data
    df = pd.read_csv(prediction_file, delimiter=",")
    gold_answers = [str(answers).split("_@_") for answers in df["gold_answer"].tolist()]
    exact_scores = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    return_metrics: Dict[str, float] = {}
    metrics = {"potential_answer": "squadv2_metrics"}
    for metric_column, metric in metrics.items():
        if metric_column in df.columns:
            predictions = df[metric_column].tolist()
            for idx, prediction in enumerate(predictions):
                gold_answer = gold_answers[idx]
                # Take max over all gold answers
                exact_scores.append(max(compute_exact(a, prediction) for a in gold_answer))
                max_f1 = 0.0
                max_p = 0.0
                max_r = 0.0
                for a in gold_answer:
                    f1, p, r = compute_f1_precision_recall(a, prediction)
                    max_f1 = max([f1, max_f1])
                    max_p = max([p, max_p])
                    max_r = max([r, max_r])

                f1_scores.append(max_f1)
                precision_scores.append(max_p)
                recall_scores.append(max_r)

            total = len(exact_scores)
            return_metrics[f"{metric}_exact"] = 100.0 * sum(exact_scores) / total
            return_metrics[f"{metric}_f1"] = 100.0 * sum(f1_scores) / total
            return_metrics[f"{metric}_precision"] = 100.0 * sum(precision_scores) / total
            return_metrics[f"{metric}_recall"] = 100.0 * sum(recall_scores) / total
    return return_metrics


def qa_metric(prediction_file: str) -> Dict[str, float]:
    """Combine and use all the metrics for the qa task."""
    scores = qa_metric_sentence_similarity(prediction_file)
    other_scores = qa_metric_squadv2_metrics(prediction_file)
    scores.update(other_scores)
    return scores


class RewardCalculator:
    """This class will be used to compute rewards to train text generators."""

    def __init__(self, reward_name: str):
        self.reward_name = reward_name

    def compute_rewards(self, sample_gold_answers: List[List[str]], sample_outputs: List[List[str]]) -> List[List[float]]:
        """Depending on the reward function, call the necessary functions."""
        sample_rewards = []
        if self.reward_name in [
            "squadv2_metrics_f1",
            "squadv2_metrics_recall",
            "squadv2_metrics_precision",
            "squadv2_metrics_exact",
        ]:
            for batch_idx in range(len(sample_gold_answers)):
                per_example_rewards = []
                for sample_idx in range(len(sample_gold_answers[batch_idx])):
                    gold_answer_string = sample_gold_answers[batch_idx][sample_idx]
                    references = str(gold_answer_string).split("_@_")
                    prediction = sample_outputs[batch_idx][sample_idx]
                    if self.reward_name == "squadv2_metrics_f1":
                        per_example_rewards.append(max(compute_f1_precision_recall(ref, prediction)[0] for ref in references))
                    elif self.reward_name == "squadv2_metrics_recall":
                        per_example_rewards.append(max(compute_f1_precision_recall(ref, prediction)[2] for ref in references))
                    elif self.reward_name == "squadv2_metrics_precision":
                        per_example_rewards.append(max(compute_f1_precision_recall(ref, prediction)[1] for ref in references))
                    elif self.reward_name == "squadv2_metrics_exact":
                        per_example_rewards.append(max(compute_exact(ref, prediction) for ref in references))
                sample_rewards.append(per_example_rewards)
        return sample_rewards


def main(argv: Any) -> None:
    """Test the metrics."""
    del argv

    print(f"prediction results for the file: {FLAGS.input_file}")
    scores = qa_metric(FLAGS.input_file)
    for key, value in scores.items():
        print(key, round(value, 4))


if __name__ == "__main__":
    app.run(main)
