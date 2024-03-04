"""Module for general utils."""

import csv
import io
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

from src.base_lm import BaseLM


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def start_predicting(model: BaseLM, dataloader: torch.utils.data.DataLoader, prediction_file: str) -> None:
    """Read batches from the dataloader and predict the outputs from the model
    for the correct experiment and save the results in the prediction_file as
    csv format row by row."""
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        header_written = False
        step = 0
        for batch in dataloader:
            for ret_row in model.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))
            step += 1
            print(f"Prediction Step: {step}.")

    return


def start_training(model: BaseLM, dataloader: torch.utils.data.DataLoader) -> Iterator[Tuple[int, torch.Tensor]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss = model.train(batch)
        step += 1
        yield step, loss


def train_loop(
    model: BaseLM,
    mode: str,
    model_path: str,
    metric_to_save: str,
    max_epochs: int,
    training_steps: int,
    steps_per_checkpoint: int,
    metric: Callable[[str], Dict[str, float]],
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run the model on input data for training."""
    if mode == "train":
        start_time = time.time()
        writer = SummaryWriter(model_path)
        epoch = 0
        global_step = 0
        total_loss = []
        eval_file = os.path.join(model_path, "temp_eval.csv")
        start_predicting(model, eval_dataloader, eval_file)
        scores = metric(eval_file)
        for score_name, score_val in scores.items():
            writer.add_scalar(f"{score_name}/dev", score_val, 0)
            if score_name == metric_to_save:
                best_score = score_val
                model.save_to_checkpoint(model_path, "best_step")

        while epoch < max_epochs and global_step < training_steps:
            print("\nEpoch:{0}\n".format(epoch))
            epoch_loss = []
            for step, loss in start_training(model, train_dataloader):
                global_step += 1

                # do optimization stuff here.
                loss = -torch.mean(loss, dim=0)
                model.optimizer.zero_grad()
                loss.backward()
                model.optimizer.step()
                model.scheduler.step(epoch)

                loss_value = loss.item()
                total_loss.append(loss_value)
                epoch_loss.append(loss_value)
                mean_total_loss = np.mean(total_loss)
                mean_epoch_loss = np.mean(epoch_loss)
                print(
                    f"\rEpoch: {epoch} | Batch: {step} | Mean Loss: {mean_total_loss} | "
                    f"Epoch Loss: {mean_epoch_loss} | Loss: {loss_value}\n"
                )
                writer.add_scalar("GPU_Usage_train", torch.cuda.max_memory_allocated(device=None) / (1e9), global_step)
                if global_step % steps_per_checkpoint == 0:
                    start_predicting(model, eval_dataloader, eval_file)
                    scores = metric(eval_file)
                    for score_name, score_val in scores.items():
                        writer.add_scalar(f"{score_name}/dev", score_val, global_step)
                        if score_name == metric_to_save:
                            if score_val > best_score:
                                best_score = score_val
                                model.save_to_checkpoint(model_path, "best_step")

                writer.add_scalar("Mean_Total_Loss/train", mean_total_loss, global_step)
                writer.add_scalar("Mean_Epoch_Loss/train", mean_epoch_loss, global_step)
                writer.flush()
                if global_step == training_steps:
                    # stop training in this epoch.
                    break

            epoch += 1

        writer.close()

        # delete the eval_file
        os.remove(eval_file)
        end_time = time.time()
        print(f"Training finished in {end_time - start_time} seconds!")

    else:
        raise Exception(f"the mode {mode} is not for training.")


def test_loop(
    model: BaseLM,
    mode: str,
    model_path: str,
    prediction_file_name: str,
    test_dataloader: torch.utils.data.DataLoader,
    metric: Optional[Callable[[str], Dict[str, float]]] = None,
) -> None:
    writer = SummaryWriter(model_path)
    if mode in ["test", "inference", "eval"]:
        ("Predicting...")
        prediction_file = os.path.join(model_path, prediction_file_name)
        start_predicting(model, test_dataloader, prediction_file)
        if metric is not None:
            scores = metric(prediction_file)
            for score_name, score_val in scores.items():
                writer.add_scalar(f"{score_name}/test", score_val, 0)
    else:
        raise Exception(f"the mode {mode} is not for testing.")


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
