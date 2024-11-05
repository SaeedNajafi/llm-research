#!/bin/bash

set -e

ENV_NAME=$1

# Get rid of cluster python.
module --force purge

conda create -n ${ENV_NAME} python=3.11

eval "$(conda shell.bash hook)"

conda activate ${ENV_NAME}

echo "Installing cxx-compiler and gxx and gcc."
conda install -c conda-forge cxx-compiler -y
conda install -c conda-forge gxx==12.2 -y
conda install -c conda-forge gcc==12.2 -y
conda install -c conda-forge level-zero -y
conda install libevent -y

# echo "Installing git."
conda install -c anaconda git -y

# echo "Installing git-lfs."
conda install -c conda-forge git-lfs -y

# echo "Installing rust."
conda install -c conda-forge rust -y

echo "Installing cuda drivers."
conda install -c nvidia/label/cuda-12.1.1 cuda -y
conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc -y
conda install -c nvidia/label/cuda-12.1.1 cuda-toolkit -y
conda install -c conda-forge cudnn -y

echo "Upgrade pip."
pip3 install --upgrade pip

echo "Install torch."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# pip3 install tensorflow tensorflow-hub

echo "Install the editable version of llm-research."
pip3 install -e .[dev]

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# echo "Install flash-attention and vllm-flash-attn"

# pip3 install --no-cache-dir flash-attn --no-build-isolation

# pip3 install vllm ray llvmlite vllm-flash-attn

# echo "Install flashinfer for vllm and gemma2 models."
# pip3 install flashinfer -i https://flashinfer.ai/whl/cu124/torch2.4

echo "Fix sqlite"
conda remove sqlite -y
conda install sqlite -y
