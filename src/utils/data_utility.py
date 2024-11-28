"""This module implements the functions for preprocessing the data files into
pytorch datasets and eventually to create a dataloader."""

import ast
import random
from typing import List, Tuple

import pandas as pd
from absl import flags
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.llm import LLM
from src.squadv2_instructions import contexts, explanation_instruction, explanations
from src.squadv2_instructions import gold_answers as context_gold_answers
from src.squadv2_instructions import normal_instruction, questions
from src.utils.general_utils import DictDataset, white_space_fix

FLAGS = flags.FLAGS

flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string("test_file", "/tmp/test.csv", "the path/name of the test file.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name of the prediction file.")


def process_squadv2_dataset(
    file_name: str, experiment_type: str, instruction_template: str, input_template: str, output_template: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Read and pre-process the squadv2 dataset for my application."""

    dataset = pd.read_csv(file_name, sep="\t").to_dict(orient="records")

    if experiment_type in ["normal_with_icl", "normal_no_icl"]:
        instruction = normal_instruction

    elif experiment_type in ["explanation_with_icl", "explanation_no_icl"]:
        instruction = explanation_instruction

    formed_instruction = instruction_template.format(instruction=instruction)

    icl_messages = []
    for icl_idx, context_example in enumerate(contexts[2:6]):
        input = f"Passage: {white_space_fix(context_example)}"
        input += f"\nQuestion: {white_space_fix(questions[icl_idx])}"
        if experiment_type == "normal_with_icl":
            input += "\nFinal Answer: "
            output = f"Final Answer: {white_space_fix(context_gold_answers[icl_idx])}"
            user_message = input_template.format(input=input)
            assistant_message = output_template.format(output=output)
            icl_messages.append(user_message)
            icl_messages.append(assistant_message)
        elif experiment_type == "explanation_with_icl":
            input += "\nExplanations and Thought Process and Final Answer: Let's think step by step."
            output = f"Explanations and Thought Process:\n{white_space_fix(explanations[icl_idx])}"
            output += f"\nFinal Answer: {white_space_fix(context_gold_answers[icl_idx])}"
            user_message = input_template.format(input=input)
            assistant_message = output_template.format(output=output)
            icl_messages.append(user_message)
            icl_messages.append(assistant_message)

    squad_inputs = []
    squad_outputs = []
    gold_outputs = []
    squad_ids = []
    idx = 0
    for row in dataset:
        idx += 1
        context = row["context"]
        question = row["question"]
        gold_answers = ast.literal_eval(row["answers"])
        context = white_space_fix(context)
        question = white_space_fix(question)
        if experiment_type in ["normal_no_icl", "normal_with_icl"]:
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nFinal Answer: "
        elif experiment_type in ["explanation_no_icl", "explanation_with_icl"]:
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nExplanations and Thought Process and Final Answer: Let's think step by step."

        formed_input = input_template.format(input=user_final_message)
        squad_input = f"{formed_instruction}{''.join(icl_messages)}{formed_input}"
        squad_inputs.append(squad_input)
        squad_ids.append(str(idx))

        gold_outputs.append("_@_".join(gold_answers))
        gold_answer = random.choice(gold_answers)
        if experiment_type == "normal_no_icl":
            # Right now, this is only valid for fine-tuning.
            squad_output = output_template.format(output=f"Final Answer: {gold_answer}")
            squad_outputs.append(squad_output)

    return squad_inputs, squad_ids, squad_outputs, gold_outputs


def create_squadv2_dataloader(
    model: LLM,
    fold_name: str,
    experiment_type: str,
    batch_size: int = 1,
    world_size: int = 1,
    rank: int = 0,
) -> DataLoader:
    """Function to create the required dataloader to train the LM models."""

    if fold_name == "train":
        file_name = FLAGS.train_file
    elif fold_name == "dev":
        file_name = FLAGS.dev_file
    elif fold_name == "test":
        file_name = FLAGS.test_file

    squad_inputs, squad_ids, squad_outputs, gold_outputs = process_squadv2_dataset(
        file_name, experiment_type, model.instruction_template, model.input_template, model.output_template
    )

    if fold_name == "train":
        data = model.prepare_text_for_train(squad_inputs, squad_outputs, squad_ids, gold_answers=gold_outputs)
        dataset = DictDataset(data)
        shuffle = True

    elif fold_name == "dev":
        # To compute the correct validation loss, we need squad_outputs.
        data = model.prepare_text_for_train(squad_inputs, squad_outputs, squad_ids, gold_answers=gold_outputs)
        dataset = DictDataset(data)
        shuffle = False

    elif fold_name == "test":
        data = model.prepare_text_for_inference(squad_inputs, squad_ids, gold_answers=gold_outputs)
        dataset = DictDataset(data)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset, shuffle=shuffle, num_replicas=world_size, rank=rank),
        num_workers=4,
    )
    return dataloader
