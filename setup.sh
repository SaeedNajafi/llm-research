#!/bin/bash

set -e


ENV_NAME=$1

conda create -n ${ENV_NAME} python=3.11

eval "$(conda shell.bash hook)"

conda activate ${ENV_NAME}

echo "Installing git."
conda install -c anaconda git -y

echo "Installing git-lfs."
conda install conda-forge::git-lfs -y

echo "Installing rust."
conda install conda-forge::rust -y

echo "Installing cuda-nvcc."
CONDA_OVERRIDE_CUDA="11.8" conda install nvidia/label/cuda-12.2.1::cuda-nvcc -y

# Get rid of cluster python.
module --force purge

echo "Upgrade pip."
pip3 install --upgrade pip

echo "Install torch and tensorflow."
pip3 install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 tensorflow tensorflow-hub

echo "Install the editable version of llm-research."
pip3 install -e .[dev]

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

echo "Install flash-attention and vllm-flash-attn"

pip3 install --no-cache-dir flash-attn --no-build-isolation

pip3 install vllm ray llvmlite vllm-flash-attn

echo "Install flashinfer for vllm and gemma2 models."
pip3 install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.3

echo "Fix sqlite"
conda remove sqlite -y
conda install sqlite -y
