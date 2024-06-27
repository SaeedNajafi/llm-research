import dataclasses
from typing import Any

import wandb

from src.configs import fsdp_config as FSDP_CONFIG
from src.configs import lora_config as LORA_CONFIG
from src.configs import train_config as TRAIN_CONFIG
from src.configs import update_config
from src.configs import wandb_config as WANDB_CONFIG


def setup_wandb(train_config: TRAIN_CONFIG, fsdp_config: FSDP_CONFIG, lora_config: LORA_CONFIG, **kwargs: Any) -> Any:
    """Setup the wandb account."""
    wandb_config = WANDB_CONFIG()
    update_config(wandb_config, **kwargs)
    init_dict = dataclasses.asdict(wandb_config)
    run = wandb.init(**init_dict)
    run.config.update(train_config)
    run.config.update(fsdp_config, allow_val_change=True)
    run.config.update(lora_config, allow_val_change=True)
    return run
