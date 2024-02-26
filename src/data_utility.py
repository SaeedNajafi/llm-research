"""This module implements the functions for preprocessing the data files into
pytorch datasets and eventually to create a dataloader.

We consider distributed training in clusters where there could be
preemption of the jobs. Therefore, we save the dataloader status along
with other components (optimizer, model, etc.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pandas as pd
import torch
import torch.distributed as dist
from absl import flags
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import _BaseDataLoaderIter
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("train_batch_size", 16, "The batch size used for training.")
flags.DEFINE_integer("eval_batch_size", 2048, "The batch size used for inference on the test or validation data.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_string("instruction_type", "qa", "The instruction type to format the input sentences.")


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


@dataclass
class GenRawData:
    """Input/Outputs for generative raw data."""

    inputs: List[str]
    gold_outputs: List[str]
    paraphrase_inputs: List[str]


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def return_gen_instruction() -> Tuple[str, str]:
    """Return the instruction type for generative QA tasks."""
    instruction = ""
    # these are llama 2 chat format.
    if "_squad_" in FLAGS.instruction_type:
        instruction = "In this task, you are given a context and question. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not have the complete information, generate 'no_answer' in the output. \
            Do not repeat the question in the output."
        template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
    elif "_race_" in FLAGS.instruction_type:
        instruction = "In this task, you are given a context. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not have the complete information, generate 'no_answer' in the output. \
            Do not repeat the question in the output."
        template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
    elif "_narrativeqa_" in FLAGS.instruction_type:
        instruction = "In this task, you are given a context. \
            Provide a short phrase as the answer for the given question using only the information from the context. \
            If you do not have the complete information, generate 'no_answer' in the output. \
            Do not repeat the question in the output."
        template = "<s> [INST] <<SYS>> {instruction} <</SYS>> {input_text} [/INST]"
    return white_space_fix(instruction), white_space_fix(template)


def tokenize_samples(batch: torch.utils.data.Dataset, samples: List[str], tokenizer: AutoTokenizer) -> None:
    samples = [white_space_fix(f"{sample} </s>") for sample in samples]
    output_encodings = tokenizer(
        samples,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    batch["para_target_attention_mask"] = torch.tensor(output_encodings.attention_mask)
    batch["para_labels"] = torch.tensor(output_encodings.input_ids)


def gen_augment_batch(
    batch: torch.utils.data.Dataset,
    paraphrases: List[str],
    tokenizer: AutoTokenizer,
    num_return_seq: int,
) -> None:
    """Augment the batch with paraphrases for generative tasks."""
    batch_size = len(batch["raw_input_ids"])

    inputs = []
    instruction, template = return_gen_instruction()
    for index in range(batch_size):
        par_inputs = []
        for par_index in range(num_return_seq):
            par_base_index = index * num_return_seq
            paraphrase = paraphrases[par_base_index + par_index : par_base_index + par_index + 1][0]
            generated_paraphrase = paraphrase.removesuffix(" </s>")
            original_question = batch["raw_input_ids"][index].split("Context:")[0]
            input_str = f"{original_question} Context: {generated_paraphrase} [/INST] "
            par_inputs.append(input_str)
        inputs.append(par_inputs)

    par_input_ids = []
    for index in range(batch_size):
        par_input_ids.append(batch["raw_input_ids"][index])
        for each in inputs[index]:
            par_input_ids.append(each)

    batch["raw_input_ids"] = par_input_ids


def gen_template_data(input_texts: List[str], output_texts: List[str]) -> GenRawData:
    """Helper function to format the data for the models in the generative
    tasks."""
    instruction, template = return_gen_instruction()
    input_sentences = [template.format(instruction=instruction, input_text=txt.removesuffix(" </s>")) for txt in input_texts]
    gold_outputs = [f"{txt}" for txt in output_texts]

    # only paraphrase the context, and not the question.
    paraphrase_inputs = [white_space_fix(f"{txt.removesuffix(' </s>').split('Context:')[1]} </s>") for txt in input_texts]
    return GenRawData(
        inputs=input_sentences,
        gold_outputs=gold_outputs,
        paraphrase_inputs=paraphrase_inputs,
    )


def read_gen_fewshot_file(file_path: str) -> GenRawData:
    """Load the fewshot files for QA task."""
    df = pd.read_csv(file_path, sep="\t")
    input_texts = df.article.tolist()
    output_texts = df.answer.tolist()
    return gen_template_data(input_texts, output_texts)


class GenDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the text
    tasks.

    May need to return actual text instead of ids for better processing
    in the code.
    """

    def __init__(self, data: Dict[str, Any]) -> None:
        """Store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        ret = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["raw_input_ids"])


def gen_tokenize_data(
    rawdata: GenRawData, tokenizer: AutoTokenizer, para_tokenizer: Optional[AutoTokenizer] = None
) -> GenDataset:
    """Tokenize data into a dataset if needed."""

    data = {"raw_input_ids": rawdata.inputs, "raw_answer_ids": rawdata.gold_outputs}

    if para_tokenizer is not None:
        para_input_encodings = para_tokenizer(
            rawdata.paraphrase_inputs,
            truncation=True,
            padding="max_length",
            max_length=FLAGS.source_max_length,
            add_special_tokens=False,
        )
        data["para_input_ids"] = para_input_encodings.input_ids
        data["para_attention_mask"] = para_input_encodings.attention_mask

    return GenDataset(data)


def create_dataloader(
    tokenizer: AutoTokenizer,
    train_file_name: Optional[str] = None,
    dev_file_name: Optional[str] = None,
    test_file_name: Optional[str] = None,
    task_name: Optional[str] = None,
    para_tokenizer: Optional[AutoTokenizer] = None,
) -> DataLoader:
    """Function to create the required dataloader to train the LM models."""
    if task_name in ["squad", "narrativeqa", "race"]:
        if train_file_name is not None:
            gen_rawdata = read_gen_fewshot_file(train_file_name)
            shuffle = True
            batch_size = FLAGS.train_batch_size

        if dev_file_name is not None:
            gen_rawdata = read_gen_fewshot_file(dev_file_name)
            shuffle = False
            batch_size = FLAGS.eval_batch_size

        if test_file_name is not None:
            gen_rawdata = read_gen_fewshot_file(test_file_name)
            shuffle = False
            batch_size = FLAGS.eval_batch_size

        dataset = gen_tokenize_data(gen_rawdata, tokenizer, para_tokenizer)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=batch_size,
            sampler=DistributedSaveableSampler(dataset, shuffle=shuffle),
            num_workers=0,
        )
    return dataloader
