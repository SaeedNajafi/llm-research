# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import csv
import gc
import io
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Tuple, Union

import torch
import torch.distributed as dist
import yaml
from accelerate.utils import is_ccl_available
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm

from src.configs import FsdpConfig, TrainConfig
from src.model_checkpointing import save_model_and_optimizer_sharded, save_model_checkpoint, save_optimizer_checkpoint
from src.utils.memory_utils import MemoryTrace


@contextlib.contextmanager
def profile(cfg: TrainConfig) -> Iterator[torch.profiler.profile]:
    """Initialize the profile context manager."""
    use_profiler: bool = cfg.use_profiler
    if use_profiler:
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if cfg.max_train_step > 0 and cfg.max_train_step < min_step:
            message = f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage,"
            message += f" {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step,"
            message += f" please increase the max_train_step, current max_train_step {cfg.max_train_step}"
            raise ValueError(message)
        print(f"pytorch profiling is activated and results will be saved in {cfg.profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(cfg.profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield torch_profiler


def save(model: Any, rank: int, train_config: TrainConfig, fsdp_config: FsdpConfig, epoch: int) -> None:
    """Save the fsdp model during training."""
    dist.barrier()
    if train_config.use_peft:
        if rank == 0:
            print("we are about to save the PEFT modules")
        model.model.save_pretrained(train_config.output_dir)
        if rank == 0:
            print(f"PEFT modules are saved in {train_config.output_dir} directory")

    else:
        if not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
            save_model_checkpoint(model.model, rank, train_config, epoch=epoch)
        elif not train_config.use_peft and fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
            print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
            print("=====================================================")

            save_model_and_optimizer_sharded(model.model, rank, train_config)
            if train_config.save_optimizer:
                save_model_and_optimizer_sharded(model.model, rank, train_config, optim=model.optimizer)
                print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                print("=====================================================")

        if not train_config.use_peft and train_config.save_optimizer:
            save_optimizer_checkpoint(model.model, model.optimizer, rank, train_config, epoch=epoch)
            print(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
            print("=====================================================")

    dist.barrier()


def train(
    model: Any,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    train_config: TrainConfig,
    fsdp_config: FsdpConfig,
    rank: int,
    world_size: int,
    wandb_run: Any,
    metric: Callable[[str], Dict[str, float]],
) -> Dict[str, Union[float, str]]:
    """Trains the model on the given dataloader.

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        train_config: The training configuration
        eval_dataloader: The dataloader containing the eval data

    Returns: results dictionary containing average training and validation perplexity and loss
    """

    train_prep: List[float] = []
    train_loss: List[float] = []
    val_prep: List[float] = []
    val_loss: List[float] = []
    val_scores: List[Dict[str, float]] = []

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)

        now_date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        metrics_filename = f"{train_config.output_dir}/metrics_data_{rank}-{now_date}.json"
        train_step_perplexity: List[float] = []
        train_step_loss: List[float] = []
        val_step_loss: List[float] = []
        val_step_perplexity: List[float] = []

    epoch_times: List[float] = []
    checkpoint_times: List[float] = []
    results: Dict[str, Union[float, str]] = {}
    best_val_score = -float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached
    # Start the training loop
    for epoch in range(train_config.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            total_loss = 0.0
            total_length = len(train_dataloader) // train_config.gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config) as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if rank == 0:
                            print(
                                "max training steps reached, stopping training, total train steps finished: ",
                                total_train_steps - 1,
                            )
                        break

                    loss = model.train(batch)
                    loss = -torch.mean(loss, dim=0)
                    loss = loss / train_config.gradient_accumulation_steps
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    total_loss += loss.detach().float()

                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % train_config.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                            model.model.clip_grad_norm_(train_config.gradient_clipping_threshold)
                        model.optimizer.step()
                        model.optimizer.zero_grad()
                        pbar.update(1)
                    if train_config.use_profiler:
                        profile_context.step()
                    if wandb_run:
                        if rank == 0:
                            wandb_run.log(
                                {
                                    "train/epoch": epoch + 1,
                                    "train/step": epoch * len(train_dataloader) + step,
                                    "train/loss": loss.detach().float(),
                                }
                            )

                    message = f"Training Epoch: {epoch+1}/{train_config.num_epochs},"
                    message += f" step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float()})"
                    pbar.set_description(message)

                    if train_config.save_metrics:
                        save_to_json(
                            metrics_filename,
                            train_step_loss,
                            train_loss,
                            train_step_perplexity,
                            train_prep,
                            val_step_loss,
                            val_loss,
                            val_step_perplexity,
                            val_prep,
                            val_scores,
                        )
                pbar.close()

        epoch_end_time = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if rank == 0:
            memtrace.print_stats()

        # Update the learning rate as needed
        model.scheduler.step()
        if train_config.run_validation:
            eval_ppl, eval_epoch_loss, temp_val_loss, temp_step_perplexity, eval_scores = evaluation(
                model, train_config, eval_dataloader, train_config.prediction_file_name, rank, world_size, wandb_run, metric
            )
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
                val_scores.append(eval_scores)

            checkpoint_start_time = time.perf_counter()
            if train_config.save_model and train_config.checkpoint_on_metric == "loss":
                if -eval_epoch_loss > best_val_score:
                    save(model, rank, train_config, fsdp_config, epoch)

            elif train_config.save_model and train_config.checkpoint_on_metric != "loss":
                for score_name, score_val in eval_scores.items():
                    if score_name == train_config.checkpoint_on_metric:
                        if score_val > best_val_score:
                            save(model, rank, train_config, fsdp_config, epoch)

            checkpoint_end_time = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_end_time)

            if train_config.checkpoint_on_metric == "loss":
                if -eval_epoch_loss > best_val_score:
                    best_val_score = -eval_epoch_loss
                    if rank == 0:
                        print(f"best eval loss on epoch {epoch+1} is {-best_val_score}")
                val_loss.append(float(-best_val_score))
                val_prep.append(float(eval_ppl))

            elif train_config.checkpoint_on_metric != "loss":
                for score_name, score_val in eval_scores.items():
                    if score_name == train_config.checkpoint_on_metric:
                        if score_val > best_val_score:
                            best_val_score = score_val
                            if rank == 0:
                                print(f"best eval {score_name} on epoch {epoch+1} is {best_val_score}")
                val_loss.append(float(-best_val_score))
                val_prep.append(float(eval_ppl))

        message = f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f},"
        message += f" train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
        if rank == 0:
            print(message)

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep,
                val_scores,
            )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep) / len(train_prep)
    avg_train_loss = sum(train_loss) / len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep) / len(val_prep)
        avg_eval_loss = sum(val_loss) / len(val_loss)

    results["avg_train_prep"] = avg_train_prep
    results["avg_train_loss"] = avg_train_loss
    if train_config.run_validation:
        results["avg_eval_prep"] = avg_eval_prep
        results["avg_eval_loss"] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # for metrics.
    temp_dict: Dict[str, float] = {score_name: 0.0 for score_name in val_scores[-1].keys()}
    for row in val_scores:
        for score_name, score_val in row.items():
            temp_dict[score_name] += score_val
    avg_scores_dict = {key: value / len(val_scores) for key, value in temp_dict.items()}
    results.update(avg_scores_dict)

    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    # saving the training params including fsdp setting for reference.
    if not train_config.use_peft and rank == 0:
        save_train_params(train_config, fsdp_config, rank)

    return results


def evaluation(
    model: Any,
    train_config: TrainConfig,
    eval_dataloader: torch.utils.data.DataLoader,
    prediction_file_name: str,
    rank: int,
    world_size: int,
    wandb_run: Any,
    metric: Callable[[str], Dict[str, float]],
) -> Tuple[float, float, List[float], List[float], Dict[str, float]]:
    """Evaluates the model on the given dataloader.

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data

    Returns: eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity
    """
    val_step_loss: List[float] = []
    val_step_perplexity: List[float] = []
    val_scores: Dict[str, torch.Tensor] = {}
    eval_loss: float = 0.0  # Initialize evaluation loss
    total_eval_steps: float = 0
    with MemoryTrace() as memtrace:
        with io.open(f"{prediction_file_name.rstrip('.csv')}_rank_{rank}.csv", mode="w", encoding="utf-8") as out_fp:
            writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
            header_written = False
            for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
                print(f"--> Prediction step {step+1} for rank {rank}")
                total_eval_steps += 1
                # stop when the maximum number of eval steps is reached
                if train_config.max_eval_step > 0 and total_eval_steps > train_config.max_eval_step:
                    if rank == 0:
                        print("max eval steps reached, stopping evaluation, total_eval_steps: ", total_eval_steps - 1)
                    break

                for ret_row, ret_loss in model.predict(batch):
                    if not header_written:
                        headers = ret_row.keys()
                        writer.writerow(headers)
                        header_written = True
                    writer.writerow(list(ret_row.values()))

                if train_config.save_metrics:
                    val_step_loss.append(ret_loss.item())
                    val_step_perplexity.append(float(torch.exp(ret_loss)))

                eval_loss += ret_loss

    if rank == 0:
        memtrace.print_stats()

    scores = metric(f"{prediction_file_name.rstrip('.csv')}_rank_{rank}.csv")
    for score_name, score_val in scores.items():
        val_scores[score_name] = torch.tensor(score_val, dtype=torch.float64, device=model.device)

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if world_size > 1:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        for score_name, score_val in val_scores.items():
            dist.all_reduce(score_val, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    eval_epoch_loss = eval_epoch_loss / world_size
    for score_name, score_val in val_scores.items():
        val_scores[score_name] = torch.div(score_val, world_size)

    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics

    float_val_socres: Dict[str, float] = {}
    for score_name, score_val in val_scores.items():
        float_val_socres[score_name] = float(score_val)

    if rank == 0:
        message = f"eval_ppl={eval_ppl} eval_epoch_loss={eval_epoch_loss}"
        for score_name, score_val in float_val_socres.items():
            message += f" {score_name}={score_val}"
        print(message)

        if wandb_run:
            log_data = {
                "eval/perplexity": eval_ppl,
                "eval/loss": eval_epoch_loss,
            }
            for score_name, score_val in float_val_socres.items():
                log_data[f"eval/{score_name}"] = score_val
            wandb_run.log(log_data, commit=False)

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity, float_val_socres


def setup() -> None:
    """Initialize the process group for distributed training."""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl")


def setup_environ_flags(rank: int) -> None:
    """Set environment flags for debugging purposes."""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        print("--> Running with torch dist debug set to detail")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()


def clear_gpu_cache() -> None:
    """Clear the GPU cache for all ranks."""
    torch.cuda.empty_cache()
    gc.collect()


def get_parameter_dtypes(model: torch.nn.Module) -> Dict[str, torch.dtype]:
    """Get the data types of model parameters."""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes


def print_model_size(model: torch.nn.Module, config: TrainConfig, rank: int = 0) -> None:
    """Print model name, the number of trainable parameters and initialization
    time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")


def save_train_params(train_config: TrainConfig, fsdp_config: FsdpConfig, rank: int) -> None:
    """This function saves the train_config and FSDP config into a
    train_params.yaml.

    This will be used by converter script in the inference folder to
    fetch the HF model name or path. It also would be helpful as a log
    for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith("__")}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith("__")}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
        train_config.dist_checkpoint_root_folder + "/" + train_config.dist_checkpoint_folder + "-" + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir, "train_params.yaml")

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        print(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, "w") as f:
            f.write(config_yaml)
        if rank == 0:
            print(f"training params are saved in {file_name}")


def save_to_json(
    output_filename: str,
    train_step_loss: List[float],
    train_epoch_loss: List[float],
    train_step_ppl: List[float],
    train_epoch_ppl: List[float],
    val_step_loss: List[float],
    val_epoch_loss: List[float],
    val_step_ppl: List[float],
    val_epoch_ppl: List[float],
    val_scores: List[Dict[str, float]],
) -> None:
    """Save some metrics to json."""
    metrics_data = {
        "train_step_loss": train_step_loss,
        "train_epoch_loss": train_epoch_loss,
        "train_step_perplexity": train_step_ppl,
        "train_epoch_perplexity": train_epoch_ppl,
        "val_step_loss": val_step_loss,
        "val_epoch_loss": val_epoch_loss,
        "val_step_perplexity": val_step_ppl,
        "val_epoch_perplexity": val_epoch_ppl,
        "val_scores": val_scores,
    }
    with open(output_filename, "w") as f:
        json.dump(metrics_data, f)
