#!/bin/bash

module --force purge

eval "$(conda shell.bash hook)"
conda activate llm-env

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

if [ "$CLUSTER_NAME" = "vcluster" ]; then
    export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
fi

export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN
export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1
