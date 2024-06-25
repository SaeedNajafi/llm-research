# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import functools
from typing import Any

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def get_size_policy(min_params: float = 1e8) -> Any:
    num_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_params)
    return num_wrap_policy


def get_llama_wrapper() -> Any:
    """We register our main layer class and use the fsdp transformer wrapping
    policy ensures embedding layers are in the root fsdp unit for shared access
    and that fsdp units map to transformer layers."""
    # ====   use new transformer wrapper

    llama_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            LlamaDecoderLayer,
        },
    )

    return llama_auto_wrap_policy
