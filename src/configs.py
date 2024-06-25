from dataclasses import dataclass, field
from typing import List, Optional, Any

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

@dataclass
class train_config:
    model_name: str = "PATH/to/Model"
    tokenizer_name: str = ""
    enable_fsdp: bool = False
    low_cpu_fsdp: bool = False
    run_validation: bool = True
    batch_size_training: int = 4
    batching_strategy: str = "packing"  # alternative: padding
    context_length: int = 4096
    gradient_accumulation_steps: int = 1
    gradient_clipping: bool = False
    gradient_clipping_threshold: float = 1.0
    num_epochs: int = 3
    T_0: int = 3
    max_train_step: int = 0
    max_eval_step: int = 0
    lm_top_p: float = 0.9
    temperature: float = 0.6
    lm_input_max_length: int = 1024
    lm_output_max_length: int = 32
    num_workers_dataloader: int = 1
    lr: float = 1e-4
    eta_min: float = 1e-5
    weight_decay: float = 0.0
    seed: int = 42
    use_fp16: bool = False
    mixed_precision: bool = True
    val_batch_size: int = 1
    dataset = "samsum_dataset"
    peft_method: str = "lora"
    use_peft: bool = False
    from_peft_checkpoint: str = ""
    output_dir: str = "PATH/to/save/PEFT/model"
    freeze_layers: bool = False
    num_freeze_layers: int = 1
    quantization: bool = False
    one_gpu: bool = False
    save_model: bool = True
    dist_checkpoint_root_folder: str = "PATH/to/save/FSDP/model"  # will be used if using FSDP
    dist_checkpoint_folder: str = "fine-tuned"  # will be used if using FSDP
    save_optimizer: bool = False  # will be used if using FSDP
    use_fast_kernels: bool = False
    use_wandb: bool = False  # Enable wandb for experient tracking
    save_metrics: bool = False  # saves training metrics to a json file for later plotting
    checkpoint_on_metric: str = "loss"
    use_profiler: bool = False
    profiler_dir: str = "PATH/to/save/profiler/results"  # will be used if using profiler


@dataclass
class lora_config:
    lora_r: int = 512
    lora_alpha: int = 512
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "o_proj", "k_proj"])
    lora_dropout: float = 0.2


@dataclass
class fsdp_config:
    mixed_precision: bool = True
    use_fp16: bool = False
    # HYBRID_SHARD "Full Shard within a node DDP cross Nodes",
    # SHARD_GRAD_OP "Shard only Gradients and Optimizer States", NO_SHARD "Similar to DDP".
    sharding_strategy: ShardingStrategy = ShardingStrategy.NO_SHARD

    # Require HYBRID_SHARD to be set. This flag can extend the HYBRID_SHARD by allowing sharding
    # a model on customized number of GPUs (Sharding_group) and Replicas over Sharding_group.
    hsdp: bool = False

    # requires hsdp to be set. This specifies the sharding group size,
    # number of GPUs that you model can fit into to form a replica of a model.
    sharding_group_size: int = 0

    # requires hsdp to be set. This specifies the replica group size,
    # which is world_size/sharding_group_size.
    replica_group_size: int = 0

    # alternatively can use SHARDED_STATE_DICT save one file per rank, and can resize the world-size.
    checkpoint_type: StateDictType = StateDictType.SHARDED_STATE_DICT
    fsdp_activation_checkpointing: bool = True
    fsdp_cpu_offload: bool = False
    pure_bf16: bool = False
    optimizer: str = "AdamW"


@dataclass
class wandb_config:
    project: str = "llama3_research"  # wandb project name
    entity: Optional[str] = None  # wandb entity name
    job_type: Optional[str] = None
    tags: Optional[List[str]] = None
    group: Optional[str] = None
    notes: Optional[str] = None
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
            elif isinstance(config, train_config):
                print(f"Warning: unknown parameter {k}")