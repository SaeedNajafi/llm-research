"""This module implements the functions for preprocessing the data files into
pytorch datasets and eventually to create a dataloader.

We consider distributed training in clusters where there could be
preemption of the jobs. Therefore, we save the dataloader status along
with other components (optimizer, model, etc.)
"""

import ast
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from absl import flags
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from src.llm import LLM
from src.squadv2_instructions import explanation_icl_input, explanation_instruction, normal_icl_input, normal_instruction
from src.utils.general_utils import white_space_fix

FLAGS = flags.FLAGS

flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string("test_file", "/tmp/test.csv", "the path/name of the test file.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name of the prediction file.")


class DictDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text
    tasks.

    May need to return actual text instead of ids for better processing
    in the code.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """Store the reference to the tokenized data."""
        self.data = data
        self.keys = list(self.data.keys())

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values or just strings."""
        ret = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str) or isinstance(val[idx], torch.Tensor):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data[self.keys[0]])


def process_squadv2_dataset(
    file_name: str, experiment_type: str, instruction_template: str, input_template: str, output_template: str
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Read and pre-process the squadv2 dataset for my application."""

    dataset = pd.read_csv(file_name, sep="\t").to_dict(orient="records")

    if experiment_type == "normal_icl":
        instruction = normal_icl_input

    elif experiment_type == "normal_no_icl":
        instruction = normal_instruction

    elif experiment_type == "explanation_icl":
        instruction = explanation_icl_input

    elif experiment_type == "explanation_no_icl":
        instruction = explanation_instruction

    formed_instruction = instruction_template.format(instruction=instruction)

    next_example_number = 11 if "_no_" not in experiment_type else -1
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
        if experiment_type == "normal_icl":
            user_final_message = f"Passage_{next_example_number}: {context}"
            user_final_message += f"\nQuestion_{next_example_number}: {question}"
            user_final_message += f"\nFinal Answer_{next_example_number}: "
        elif experiment_type == "normal_no_icl":
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nFinal Answer: "
        elif experiment_type == "explanation_icl":
            user_final_message = f"Passage_{next_example_number}: {context}"
            user_final_message += f"\nQuestion_{next_example_number}: {question}"
            user_final_message += f"\nExplanations and Thought Process_{next_example_number}: "
        elif experiment_type == "explanation_no_icl":
            user_final_message = f"Passage: {context}"
            user_final_message += f"\nQuestion: {question}"
            user_final_message += "\nExplanations and Thought Process and Final Answer: "

        formed_input = input_template.format(input=user_final_message)
        squad_input = f"{formed_instruction}{formed_input}"
        squad_inputs.append(squad_input)
        squad_ids.append(str(idx))

        gold_outputs.append("_@_".join(gold_answers))
        gold_answer = random.choice(gold_answers)

        if experiment_type == "normal_icl":
            squad_output = output_template.format(output=f"Final Answer_{next_example_number}: {gold_answer}")
            squad_outputs.append(squad_output)
        elif experiment_type == "normal_no_icl":
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
        data = model.prepare_text_for_train(squad_inputs, squad_outputs, squad_ids)
        dataset = DictDataset(data)
        shuffle = True

    elif fold_name in ["dev", "test"]:
        data = model.prepare_text_for_inference(squad_inputs, squad_ids, gold_answers=gold_outputs)
        dataset = DictDataset(data)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        sampler=DistributedSampler(dataset, shuffle=shuffle, num_replicas=world_size, rank=rank),
        num_workers=8,
    )
    return dataloader
