#!/bin/bash

set -e

function narval_up () {
    module --force purge

    eval "$(conda shell.bash hook)"
    conda activate llm-env

    export CUDA_HOME=$CONDA_PREFIX
    export NCCL_HOME=$CONDA_PREFIX
    export LD_LIBRARY_PATH=$CONDA_PREFIX/lib
}

function vcluster_up () {
    module load cuda11.8+cudnn8.9.6

    source ${PWD}/llm-env/bin/activate

    export PATH=${PWD}/llm-env/bin:$PATH
    export TRITON_PTXAS_PATH=/pkgs/cuda-11.8/bin/ptxas
    export XDG_RUNTIME_DIR=""

}
