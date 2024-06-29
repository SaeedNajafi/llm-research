from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional

import wandb
from peft import TaskType
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType


@dataclass
class TrainConfig:
    model_name: str = "/model-weights/Meta-Llama-3-8B-Instruct"
    llm_name: str = "llama3"
    # model_name: str = "google/gemma-2-9b-it"
    # llm_name: str = "gemma2"
    mode: str = "train"
    experiment_type: str = "normal_no_icl"
    train_file_name: str = "./notebooks/1024-shot-datasets/squad/1024-13-train.tsv"
    dev_file_name: str = "./notebooks/1024-shot-datasets/squad/1024-13-dev.tsv"
    test_file_name: str = "./notebooks/1024-shot-datasets/squad/original_validation.tsv"
    prediction_file_name: str = "/scratch/ssd004/scratch/snajafi/checkpoints/gemma2-1024-13/squadv2.results.csv"
    output_dir: str = "/scratch/ssd004/scratch/snajafi/checkpoints/gemma2-1024-13"

    # will be used if using FSDP
    dist_checkpoint_root_folder: str = "/scratch/ssd004/scratch/snajafi/checkpoints/gemma2-1024-13"
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP

    checkpoint_on_metric: str = "squadv2_metrics_f1"
    run_validation: bool = True
    batch_size_training: int = 8
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = True
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 50
    T_0: int = 10
    max_train_step: int = 0
    max_eval_step: int = 0
    low_cpu_mem_usage: bool = True
    lm_top_p: float = 0.9
    temperature: float = 0.01
    lm_input_max_length: int = 1024
    lm_output_max_length: int = 256
    num_workers_dataloader: int = 1
    lr: float = 5e-5
    eta_min: float = 1e-5
    weight_decay: float = 0.001
    seed: int = 13
    val_batch_size: int = 8
    peft_method: str = "lora"
    use_peft: bool = True
    from_peft_checkpoint: str = ""
    save_model: bool = True

    save_optimizer: bool = True  # will be used if using FSDP
    attn_implementation: str = "eager"  # "flash_attention_2"
    use_wandb: bool = True  # Enable wandb for experient tracking
    save_metrics: bool = True  # saves training metrics to a json file for later plotting
    use_profiler: bool = False
    profiler_dir: str = "/scratch/ssd004/scratch/snajafi/checkpoints/gemma2-1024-13/profiler"  # will be used if using profiler


@dataclass
class LoraConfig:
    r: int = 512
    lora_alpha: int = 512
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj", "k_proj"])
    lora_dropout: float = 0.3
    task_type: TaskType = TaskType.CAUSAL_LM
    inference_mode: bool = False


@dataclass
class FsdpConfig:
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    # HYBRID_SHARD "Full Shard within a node DDP cross Nodes",
    # SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    sharding_strategy: str = "NO_SHARD"
    # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT


@dataclass
class WandbConfig:
    project: str = "llm_research"  # wandb project name
    entity: Optional[str] = None  # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = "squadv2-experiments"
    notes: Optional[str] = "tuning weights"
    mode: Optional[str] = None


def update_config(config: Any, **kwargs: Any) -> None:
    """Update the config classes using the kwargs."""
    if isinstance(config, (tuple, list)):
        for c in config:
            update_config(c, **kwargs)
    else:
        for k, v in kwargs.items():
            if hasattr(config, k):
                setattr(config, k, v)
            elif "." in k:
                # allow --some_config.some_param=True
                config_name, param_name = k.split(".")
                if type(config).__name__ == config_name:
                    if hasattr(config, param_name):
                        setattr(config, param_name, v)
                    else:
                        # In case of specialized config we can warn user
                        print(f"Warning: {config_name} does not accept parameter: {k}")
            elif isinstance(config, TrainConfig):
                print(f"Warning: unknown parameter {k}")


def setup_wandb(train_config: TrainConfig, fsdp_config: FsdpConfig, lora_config: LoraConfig, **kwargs: Any) -> Any:
    """Setup the wandb account."""
    wandb_config = WandbConfig()
    update_config(wandb_config, **kwargs)
    init_dict = asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    run.config.update(lora_config, allow_val_change=True)
    return run
