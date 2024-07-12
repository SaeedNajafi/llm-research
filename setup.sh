#!/bin/bash

set -e


ENV_NAME=$1

conda create -n ${ENV_NAME} python=3.11

eval "$(conda shell.bash hook)"

conda activate ${ENV_NAME}

echo "Installing git."
conda install -c anaconda git

echo "Installing git-lfs."
conda install conda-forge::git-lfs

echo "Installing rust."
conda install conda-forge::rust

echo "Installing pytorch related modules."
CONDA_OVERRIDE_CUDA="12.1" conda install pytorch torchvision torchtriton torchserve torchtext magma-cuda121 torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

echo "Installing tensorflow related modules."
CONDA_OVERRIDE_CUDA="12.1" conda install tensorflow tensorflow-hub -c conda-forge

echo "Installing cuda-nvcc."
CONDA_OVERRIDE_CUDA="12.1" conda install cuda-nvcc -c nvidia

# Get rid of cluster python.
module --force purge

echo "Upgrade pip."
pip3 install --upgrade pip

echo "Install the editable version of llm-research."
pip3 install -e .[dev]

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

echo "Install flash-attention and vllm-flash-attn and vllm-nccl-cu12."

MAX_JOBS=8 pip3 install --no-cache-dir flash-attn --no-build-isolation
pip3 install vllm-flash-attn
pip3 install vllm-nccl-cu12

echo "Install flashinfer for vllm and gemma2 models."
pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3
