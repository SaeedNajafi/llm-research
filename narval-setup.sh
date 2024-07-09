#!/bin/bash

set -e


ENV_NAME="llm-env"

module load arrow/16.1.0
conda create -n ${ENV_NAME} python=3.10
conda activate ${ENV_NAME}
CONDA_OVERRIDE_CUDA="11.8" conda install tensorflow=2.14.0 tensorflow-hub -c conda-forge
CONDA_OVERRIDE_CUDA="11.8" conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
CONDA_OVERRIDE_CUDA="11.8" conda install nvidia/label/cuda-11.8.0::cuda-nvcc
pip3 install -e .[dev]
export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
MAX_JOBS=8 pip3 install --no-cache-dir flash-attn --no-build-isolation
