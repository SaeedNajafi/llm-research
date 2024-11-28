# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import contextlib
import csv
import datetime
import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

import torch
import torch.distributed as dist
from absl import flags, logging
from accelerate.utils import is_ccl_available
from tqdm import tqdm

from src.trainers import LossCalculator
from src.utils.memory_utils import MemoryTrace
from src.utils.save_utils import find_checkpoint, save_checkpoint, save_to_json

FLAGS = flags.FLAGS

flags.DEFINE_boolean("use_profiler", False, "use profiler or not.")
flags.DEFINE_integer("max_train_step", 0, "maximum number of steps for training.")
flags.DEFINE_integer("max_eval_step", 0, "maximum number of steps for evaluating.")
flags.DEFINE_integer("num_epochs", 10, "number of training epochs.")
flags.DEFINE_integer("steps_before_evaluation", 100, "number of training steps before evaluation.")

flags.DEFINE_integer("gradient_accumulation_steps", 1, "number of training steps for gradient accumulation.")
flags.DEFINE_boolean("gradient_clipping", True, "use gradient clipping or not?")
flags.DEFINE_float("gradient_clipping_threshold", 1.0, "threshold for gradient clipping.")
flags.DEFINE_boolean("run_validation", True, "run validation and compute metrics.")
flags.DEFINE_string("checkpoint_on_metric", "loss", "loss | squadv2_metrics_f1")
flags.DEFINE_boolean("disable_scheduler", False, "Whether to disable scheduler or not.")


@dataclass
class EvaluationOutput:
    eval_prediction_ppl: float = 0.0
    eval_epoch_prediction_loss: float = 0.0
    val_step_prediction_losses: List[float] = field(default_factory=list)
    val_step_prediction_perplexities: List[float] = field(default_factory=list)
    eval_true_ppl: float = 0.0
    eval_epoch_true_loss: float = 0.0
    val_step_true_losses: List[float] = field(default_factory=list)
    val_step_true_perplexities: List[float] = field(default_factory=list)
    float_val_scores: Dict[str, float] = field(default_factory=dict)


@contextlib.contextmanager
def profile() -> Iterator[torch.profiler.profile]:
    """Initialize the profile context manager."""
    if FLAGS.use_profiler:
        profiler_dir = os.path.join(FLAGS.checkpoint_folder, "profiler")
        os.makedirs(profiler_dir, exist_ok=True)
        # profiler needs a warmup stage to get the accurate profiling results
        wait_step, warmup_step, active_step = 1, 2, 3
        min_step = wait_step + warmup_step + active_step + 1
        if FLAGS.max_train_step > 0 and FLAGS.max_train_step < min_step:
            msg = f"pytorch profiler requires at least {min_step} train steps to finish the warm-up and recording stage,"
            msg += f" {wait_step} for wait_step, {warmup_step} for warmup_step, {active_step} for profiling step,"
            msg += f" please increase the max_train_step, current max_train_step {FLAGS.max_train_step}"
            raise ValueError(msg)
        logging.info(f"pytorch profiling is activated and results will be saved in {profiler_dir}")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=wait_step, warmup=warmup_step, active=active_step, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_dir),
            profile_memory=True,
            with_stack=False,
            with_flops=True,
            record_shapes=True,
        ) as torch_profiler:
            yield torch_profiler
    else:
        torch_profiler = contextlib.nullcontext()
        yield torch_profiler


def train(
    model: Any,
    loss_calculator: LossCalculator,
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
    rank: int,
    world_size: int,
    wandb_run: Any,
    metrics: List[Callable[[str], Dict[str, float]]],
) -> Dict[str, Union[float, str]]:
    """Trains the model on the given dataloader."""

    train_perplexities: List[float] = []
    train_losses: List[float] = []
    val_prediction_perplexities: List[float] = []
    val_prediction_losses: List[float] = []
    val_true_perplexities: List[float] = []
    val_true_losses: List[float] = []
    val_scores: List[Dict[str, float]] = []

    if not os.path.exists(FLAGS.checkpoint_folder):
        os.makedirs(FLAGS.checkpoint_folder, exist_ok=True)

    now_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    metric_filename = f"{FLAGS.checkpoint_folder}/metrics_data_{rank}-{now_date}.json"
    train_step_perplexities: List[float] = []
    train_step_losses: List[float] = []
    val_step_prediction_losses: List[float] = []
    val_step_prediction_perplexities: List[float] = []
    val_step_true_losses: List[float] = []
    val_step_true_perplexities: List[float] = []

    epoch_times: List[float] = []
    checkpoint_times: List[float] = []
    results: Dict[str, Union[float, str]] = {}
    total_train_steps = 0
    grad_norm = 0.0
    max_steps_reached = False  # Flag to indicate max training steps reached
    best_val_score = -float("inf")

    # Checkpoint check. Always call before training.
    # If no checkpoint, it returns 0.
    _, checkpointed_epoch = find_checkpoint(model)

    # Evaluate the pre-trained model.
    evaluation(model, "dev", eval_dataloader, FLAGS.prediction_file, rank, world_size, wandb_run, metrics, loss_calculator)

    # Start the training loop
    for epoch in range(checkpointed_epoch, FLAGS.num_epochs):
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            total_loss = 0.0
            total_length = len(train_dataloader) // FLAGS.gradient_accumulation_steps
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile() as profile_context:
                for step, batch in enumerate(train_dataloader):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if FLAGS.max_train_step > 0 and total_train_steps > FLAGS.max_train_step:
                        max_steps_reached = True
                        if rank == 0:
                            msg = "max training steps reached, stopping training,"
                            msg += f" total train steps finished: {total_train_steps - 1}"
                            logging.info(msg)
                        break

                    if (step + 1) % FLAGS.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        # next forward / backward pass will be synced
                        dist.barrier()
                        loss = loss_calculator.train(batch)
                        loss = loss / FLAGS.gradient_accumulation_steps
                        loss_value = loss.detach().float()
                        train_step_losses.append(loss_value.item())
                        train_step_perplexities.append(float(torch.exp(loss_value)))
                        total_loss += loss_value

                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if FLAGS.gradient_clipping and FLAGS.gradient_clipping_threshold > 0.0:
                            if model.distributed_strategy == "fsdp":
                                grad_norm = model.model.clip_grad_norm_(FLAGS.gradient_clipping_threshold)
                            elif model.distributed_strategy == "ddp":
                                grad_norm = torch.nn.utils.clip_grad_norm_(
                                    model.model.parameters(), FLAGS.gradient_clipping_threshold
                                )
                        model.optimizer.step()
                        model.optimizer.zero_grad()
                        pbar.update(1)

                    else:
                        # no need to sync while accumulating gradients
                        with model.model.no_sync():
                            loss = loss_calculator.train(batch)
                            loss = loss / FLAGS.gradient_accumulation_steps
                            loss_value = loss.detach().float()
                            train_step_losses.append(loss_value.item())
                            train_step_perplexities.append(float(torch.exp(loss_value)))
                            total_loss += loss_value
                            # regular backpropagation when fp16 is not used
                            loss.backward()

                    if FLAGS.use_profiler:
                        profile_context.step()

                    if wandb_run:
                        wandb_run.log(
                            {
                                f"train/rank_{rank}/epoch": epoch + 1,
                                f"train/rank_{rank}/step": epoch * len(train_dataloader) + step,
                                f"train/rank_{rank}/loss": loss_value,
                                f"train/rank_{rank}/grad_norm": grad_norm,
                            }
                        )

                    msg = f"Rank: {rank} --> Training Epoch: {epoch + 1}/{FLAGS.num_epochs},"
                    msg += f" step {step + 1}/{len(train_dataloader)} completed (loss: {loss_value})"
                    pbar.set_description(msg)

                    if FLAGS.run_validation:
                        if (step + 1) % FLAGS.steps_before_evaluation == 0:
                            evaluation_output = evaluation(
                                model,
                                "dev",
                                eval_dataloader,
                                FLAGS.prediction_file,
                                rank,
                                world_size,
                                wandb_run,
                                metrics,
                                loss_calculator,
                            )
                            val_step_prediction_losses.extend(evaluation_output.val_step_prediction_losses)
                            val_step_prediction_perplexities.extend(evaluation_output.val_step_prediction_perplexities)
                            val_scores.append(evaluation_output.float_val_scores)
                            val_prediction_losses.append(float(evaluation_output.eval_epoch_prediction_loss))
                            val_prediction_perplexities.append(float(evaluation_output.eval_prediction_ppl))
                            if FLAGS.compute_true_validation_loss:
                                val_step_true_losses.extend(evaluation_output.val_step_true_losses)
                                val_step_true_perplexities.extend(evaluation_output.val_step_true_perplexities)
                                val_true_losses.append(float(evaluation_output.eval_epoch_true_loss))
                                val_true_perplexities.append(float(evaluation_output.eval_true_ppl))

                            checkpoint_start_time = time.perf_counter()
                            if FLAGS.checkpoint_on_metric == "validation_loss":
                                if -evaluation_output.eval_epoch_true_loss > best_val_score:
                                    save_checkpoint(model, step + 1, epoch + 1)
                                    best_val_score = -evaluation_output.eval_epoch_true_loss
                                    if rank == 0:
                                        logging.info(
                                            f"best eval loss on epoch {epoch + 1}, step {step + 1} is {-best_val_score}."
                                        )

                            elif FLAGS.checkpoint_on_metric != "validation_loss":
                                for score_name, score_val in evaluation_output.float_val_scores.items():
                                    if score_name == FLAGS.checkpoint_on_metric:
                                        if score_val > best_val_score:
                                            save_checkpoint(model, step + 1, epoch + 1)
                                            best_val_score = score_val
                                            if rank == 0:
                                                msg = f"best eval {score_name} on epoch {epoch + 1},"
                                                msg += f" step {step + 1} is {best_val_score}."
                                                logging.info(msg)

                            checkpoint_duration = time.perf_counter() - checkpoint_start_time
                            checkpoint_times.append(checkpoint_duration)

                    save_to_json(
                        metric_filename,
                        train_perplexities,
                        train_losses,
                        train_step_perplexities,
                        train_step_losses,
                        val_step_prediction_perplexities,
                        val_step_prediction_losses,
                        val_step_true_perplexities,
                        val_step_true_losses,
                        val_prediction_perplexities,
                        val_prediction_losses,
                        val_true_perplexities,
                        val_true_losses,
                        val_scores,
                    )

                pbar.close()

        epoch_duration = time.perf_counter() - epoch_start_time
        epoch_times.append(epoch_duration)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if world_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_loss = train_epoch_loss / world_size
        train_perplexity = torch.exp(train_epoch_loss)
        train_perplexities.append(float(train_perplexity))
        train_losses.append(float(train_epoch_loss))
        msg = f"Epoch {epoch + 1}: train_perplexity={train_perplexity:.4f},"
        msg += f" train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_duration}s"
        if rank == 0:
            logging.info(msg)
            memtrace.print_stats()

        # Update the learning rate as needed.
        if not FLAGS.disable_scheduler:
            model.scheduler.step()

        # End of epoch validation.
        if FLAGS.run_validation:
            evaluation_output = evaluation(
                model,
                "dev",
                eval_dataloader,
                FLAGS.prediction_file,
                rank,
                world_size,
                wandb_run,
                metrics,
                loss_calculator,
            )
            val_step_prediction_losses.extend(evaluation_output.val_step_prediction_losses)
            val_step_prediction_perplexities.extend(evaluation_output.val_step_prediction_perplexities)
            val_scores.append(evaluation_output.float_val_scores)
            val_prediction_losses.append(float(evaluation_output.eval_epoch_prediction_loss))
            val_prediction_perplexities.append(float(evaluation_output.eval_prediction_ppl))
            if FLAGS.compute_true_validation_loss:
                val_step_true_losses.extend(evaluation_output.val_step_true_losses)
                val_step_true_perplexities.extend(evaluation_output.val_step_true_perplexities)
                val_true_losses.append(float(evaluation_output.eval_epoch_true_loss))
                val_true_perplexities.append(float(evaluation_output.eval_true_ppl))

            checkpoint_start_time = time.perf_counter()
            if FLAGS.checkpoint_on_metric == "validation_loss":
                if -evaluation_output.eval_epoch_true_loss > best_val_score:
                    save_checkpoint(model, step + 1, epoch + 1)
                    best_val_score = -evaluation_output.eval_epoch_true_loss
                    if rank == 0:
                        logging.info(f"best eval loss on epoch {epoch + 1} is {-best_val_score}.")

            elif FLAGS.checkpoint_on_metric != "validation_loss":
                for score_name, score_val in evaluation_output.float_val_scores.items():
                    if score_name == FLAGS.checkpoint_on_metric:
                        if score_val > best_val_score:
                            save_checkpoint(model, step + 1, epoch + 1)
                            best_val_score = score_val
                            if rank == 0:
                                msg = f"best eval {score_name} on epoch {epoch + 1} is {best_val_score}."
                                logging.info(msg)

            checkpoint_duration = time.perf_counter() - checkpoint_start_time
            checkpoint_times.append(checkpoint_duration)

        # Saving the results to plot later
        save_to_json(
            metric_filename,
            train_perplexities,
            train_losses,
            train_step_perplexities,
            train_step_losses,
            val_step_prediction_perplexities,
            val_step_prediction_losses,
            val_step_true_perplexities,
            val_step_true_losses,
            val_prediction_perplexities,
            val_prediction_losses,
            val_true_perplexities,
            val_true_losses,
            val_scores,
        )

    avg_epoch_time = sum(epoch_times) / len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times) / len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_perplexity = sum(train_perplexities) / len(train_perplexities)
    avg_train_loss = sum(train_losses) / len(train_losses)
    if FLAGS.run_validation:
        avg_eval_prediction_perplexity = sum(val_prediction_perplexities) / len(val_prediction_perplexities)
        avg_eval_prediction_loss = sum(val_prediction_losses) / len(val_prediction_losses)
        if FLAGS.compute_true_validation_loss:
            avg_eval_true_perplexity = sum(val_true_perplexities) / len(val_true_perplexities)
            avg_eval_true_loss = sum(val_true_losses) / len(val_true_losses)

    results["avg_train_perplexity"] = avg_train_perplexity
    results["avg_train_loss"] = avg_train_loss
    if FLAGS.run_validation:
        results["avg_eval_prediction_perplexity"] = avg_eval_prediction_perplexity
        results["avg_eval_prediction_loss"] = avg_eval_prediction_loss
        if FLAGS.compute_true_validation_loss:
            results["avg_eval_true_perplexity"] = avg_eval_true_perplexity
            results["avg_eval_true_loss"] = avg_eval_true_loss

    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time

    # For metrics.
    temp_dict: Dict[str, float] = {score_name: 0.0 for score_name in val_scores[-1].keys()}
    for row in val_scores:
        for score_name, score_val in row.items():
            temp_dict[score_name] += score_val
    avg_scores_dict = {key: value / len(val_scores) for key, value in temp_dict.items()}
    results.update(avg_scores_dict)

    results["metric_filename"] = metric_filename

    return results


def evaluation(
    model: Any,
    eval_type: str,
    eval_dataloader: torch.utils.data.DataLoader,
    prediction_file_name: str,
    rank: int,
    world_size: int,
    wandb_run: Any,
    metrics: Optional[List[Callable[[str], Dict[str, float]]]] = None,
    loss_calculator: Optional[LossCalculator] = None,
) -> Any:
    """Evaluates the model on the given dataloader."""
    val_step_prediction_losses: List[float] = []
    val_step_prediction_perplexities: List[float] = []
    val_step_true_losses: List[float] = []
    val_step_true_perplexities: List[float] = []
    val_scores: Dict[str, torch.Tensor] = {}
    eval_prediction_loss: float = 0.0  # Initialize evaluation loss
    eval_true_loss: float = 0.0  # Initialize evaluation true loss
    total_eval_steps: float = 0
    with MemoryTrace() as memtrace:
        with io.open(
            f"{prediction_file_name.rstrip('.csv')}_{eval_type}_rank_{rank}.csv", mode="w", encoding="utf-8"
        ) as out_fp:
            writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
            header_written = False
            for step, batch in enumerate(tqdm(eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True)):
                logging.info(f"--> Prediction step {step+1} for rank {rank}")
                total_eval_steps += 1
                # stop when the maximum number of eval steps is reached
                if FLAGS.max_eval_step > 0 and total_eval_steps > FLAGS.max_eval_step:
                    if rank == 0:
                        msg = f"max eval steps reached, stopping evaluation, total_eval_steps: {total_eval_steps - 1}"
                        logging.info(msg)
                    break

                for ret_row, prediction_loss in model.predict(batch):
                    if not header_written:
                        headers = ret_row.keys()
                        writer.writerow(headers)
                        header_written = True
                    writer.writerow(list(ret_row.values()))

                # model.predic returns the batch prediction loss on each yield.
                batch_prediction_loss = prediction_loss
                val_step_prediction_losses.append(batch_prediction_loss.item())
                val_step_prediction_perplexities.append(float(torch.exp(batch_prediction_loss)))
                eval_prediction_loss += batch_prediction_loss

                if FLAGS.compute_true_validation_loss:
                    assert loss_calculator is not None
                    validation_true_loss = loss_calculator.compute_true_validation_loss(batch)
                    val_step_true_losses.append(validation_true_loss.item())
                    val_step_true_perplexities.append(float(torch.exp(validation_true_loss)))
                    eval_true_loss += validation_true_loss

    dist.barrier()
    if rank == 0:
        memtrace.print_stats()

    scores: Dict[str, float] = {}
    if metrics is not None:
        for metric in metrics:
            metric_scores = metric(f"{prediction_file_name.rstrip('.csv')}_{eval_type}_rank_{rank}.csv")
            scores.update(metric_scores)

    for score_name, score_val in scores.items():
        val_scores[score_name] = torch.tensor(score_val, dtype=torch.float64, device=model.device)

    # If there's more than one CUDA device, reduce evaluation loss across all devices.
    if world_size > 1:
        dist.all_reduce(eval_prediction_loss, op=dist.ReduceOp.SUM)
        if FLAGS.compute_true_validation_loss:
            dist.all_reduce(eval_true_loss, op=dist.ReduceOp.SUM)
        for score_name, score_val in val_scores.items():
            dist.all_reduce(score_val, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_prediction_loss = eval_prediction_loss / len(eval_dataloader)
    eval_epoch_prediction_loss = eval_epoch_prediction_loss / world_size
    if FLAGS.compute_true_validation_loss:
        eval_epoch_true_loss = eval_true_loss / len(eval_dataloader)
        eval_epoch_true_loss = eval_epoch_true_loss / world_size
    for score_name, score_val in val_scores.items():
        val_scores[score_name] = torch.div(score_val, world_size)

    eval_prediction_ppl = torch.exp(eval_epoch_prediction_loss)
    if FLAGS.compute_true_validation_loss:
        eval_true_ppl = torch.exp(eval_epoch_true_loss)

    # Print evaluation metrics

    float_val_scores: Dict[str, float] = {}
    for score_name, score_val in val_scores.items():
        float_val_scores[score_name] = float(score_val)

    if rank == 0:
        message = (
            f"{eval_type}_prediction_ppl={eval_prediction_ppl}\n{eval_type}_epoch_prediction_loss={eval_epoch_prediction_loss}"
        )
        if FLAGS.compute_true_validation_loss:
            message += f"\n{eval_type}_true_ppl={eval_true_ppl}\n{eval_type}_epoch_true_loss={eval_epoch_true_loss}"
        for score_name, score_val in float_val_scores.items():
            message += f"\n{score_name}={score_val}"
        logging.info(message)

        if wandb_run:
            log_data = {
                f"{eval_type}/prediction_perplexity": eval_prediction_ppl,
                f"{eval_type}/prediction_loss": eval_epoch_prediction_loss,
            }
            if FLAGS.compute_true_validation_loss:
                log_data[f"{eval_type}/true_perplexity"] = eval_true_ppl
                log_data[f"{eval_type}/true_loss"] = eval_epoch_true_loss
            for score_name, score_val in float_val_scores.items():
                log_data[f"{eval_type}/{score_name}"] = score_val
            wandb_run.log(log_data, commit=False)

    dist.barrier()
    return EvaluationOutput(
        eval_prediction_ppl=eval_prediction_ppl,
        eval_epoch_prediction_loss=eval_epoch_prediction_loss,
        val_step_prediction_losses=val_step_prediction_losses,
        val_step_prediction_perplexities=val_step_prediction_perplexities,
        eval_true_ppl=eval_true_ppl,
        eval_epoch_true_loss=eval_epoch_true_loss,
        val_step_true_losses=val_step_true_losses,
        val_step_true_perplexities=val_step_true_perplexities,
        float_val_scores=float_val_scores,
    )


def setup() -> None:
    """Initialize the process group for distributed training."""
    if is_ccl_available():
        # distributed training on xpus
        dist.init_process_group("ccl")
    else:
        dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=60000))


def setup_environ_flags(rank: int) -> None:
    """Set environment flags for debugging purposes."""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        logging.info("--> Running with torch dist debug set to detail")


def cleanup() -> None:
    """Clean up the process group after training."""
    dist.destroy_process_group()
