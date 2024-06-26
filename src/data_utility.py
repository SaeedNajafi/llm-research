"""This module implements the functions for preprocessing the data files into
pytorch datasets and eventually to create a dataloader.

We consider distributed training in clusters where there could be
preemption of the jobs. Therefore, we save the dataloader status along
with other components (optimizer, model, etc.)
"""

import ast
import random
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler

from src.general_utils import DictDataset, white_space_fix
from src.llama3 import Llama3
from src.squadv2_llama3_instructions import explanation_icl_input, explanation_instruction, normal_icl_input, normal_instruction


class DistributedSaveableSampler(DistributedSampler):
    """Just like with the case with
    torch.utils.data.distributed.DistributedSampler you *MUST* call
    self.set_epoch(epoch:int) to ensure all replicates use the same random
    shuffling within each epoch if shuffle is True."""

    def __init__(
        self,
        dataset: torch.utils.data.dataset.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        """
        Arguments:
            force_synchronization (boolean, optional): If it's true then after
                each yield we will force a synchronization so each process'
                _curr_idx will be the same, this guarantees correctness of the
                save in case there is no synchronization during training, but
                comes at a performance cost
            For the rest of the arguments please see:
                https://pytorch.org/docs/1.7.1/data.html?highlight=distributed%20sampler#torch.utils.data.distributed.DistributedSampler

        """
        super().__init__(dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)

        self._curr_idx = 0
        self.force_synchronization = True

    def __iter__(self) -> Iterator[int]:
        """Logic modified from
        https://pytorch.org/docs/1.7.1/_modules/torch/utils/data/distributed.html#DistributedSampler
        """
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore
        else:
            indices = list(range(len(self.dataset)))  # type: ignore

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            indices += indices[: (self.total_size - len(indices))]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        while self._curr_idx + self.rank < self.total_size:
            to_yield = self.rank + self._curr_idx

            # we need to increment this before the yield because
            # there might be a save or preemption while we are yielding
            # so we must increment it before to save the right index
            self._curr_idx += self.num_replicas

            yield to_yield

            if self.force_synchronization:
                dist.barrier()
        self._curr_idx = 0

    def state_dict(self, dataloader_iter: Optional[_BaseDataLoaderIter] = None) -> Dict[str, int]:
        """Create a state dict for the DataSampler."""
        prefetched_num = 0
        # in the case of multiworker dataloader, the helper worker could be
        # pre-fetching the data that is not consumed by the main dataloader.
        # we need to subtract the unconsumed part .
        if dataloader_iter is not None:
            if dataloader_iter._num_workers > 0:
                batch_size = dataloader_iter._index_sampler.batch_size
                prefetched_num = (dataloader_iter._send_idx - dataloader_iter._rcvd_idx) * batch_size

        return {
            "index": self._curr_idx - (prefetched_num * self.num_replicas),
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict: Dict[str, int]) -> None:
        """Load the DataSampler's state."""
        self._curr_idx = state_dict["index"]
        self.epoch = state_dict["epoch"]


def process_squad_dataset(
    file_name: str, experiment_type: str, llama3_instruction_llama: str, llama3_input_template: str, llama3_output_template: str
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

    llama3_instruction = llama3_instruction_llama.format(instruction=instruction)

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

        llama3_input = llama3_input_template.format(input=user_final_message)
        squad_input = f"{llama3_instruction}{llama3_input}"
        squad_inputs.append(squad_input)
        squad_ids.append(str(idx))

        gold_outputs.append("_@_".join(gold_answers))
        gold_answer = random.choice(gold_answers)

        if experiment_type == "normal_icl":
            squad_output = llama3_output_template.format(output=f"Final Answer_{next_example_number}: {gold_answer}")
            squad_outputs.append(squad_output)
        elif experiment_type == "normal_no_icl":
            squad_output = llama3_output_template.format(output=f"Final Answer: {gold_answer}")
            squad_outputs.append(squad_output)
    return squad_inputs, squad_ids, squad_outputs, gold_outputs


def create_squadv2_dataloader(
    model: Llama3, file_name: str, fold_name: str, experiment_type: str, batch_size: int = 1, world_size: int = 1, rank: int = 0
) -> DataLoader:
    """Function to create the required dataloader to train the LM models."""

    squad_inputs, squad_ids, squad_outputs, gold_outputs = process_squad_dataset(
        file_name, experiment_type, model.llama3_instruction_llama, model.llama3_input_template, model.llama3_output_template
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
        sampler=DistributedSaveableSampler(dataset, shuffle=shuffle, num_replicas=world_size, rank=rank),
        num_workers=0,
    )
    return dataloader
