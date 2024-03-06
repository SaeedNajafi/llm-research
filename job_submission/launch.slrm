#!/bin/bash
#SBATCH --job-name=llama-research
#SBATCH --nodes=1
#SBATCH --mem=0
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres=gpu:4
#SBATCH --output=llama-research.%j.out
#SBATCH --error=llama-research.%j.err
#SBATCH --partition=a40
#SBATCH --qos=m2
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1
#SBATCH --time=01:00:00

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

export NCCL_IB_DISABLE=1  # Our cluster does not have InfiniBand. We need to disable usage using this flag.
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=WARN

export TORCH_DISTRIBUTED_DEBUG=DETAIL  # Uncomment these flags for debugging communication
export TORCH_CPP_LOG_LEVEL=INFO
export LOGLEVEL=INFO
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1

module load python/3.10.12 cuda-11.8
source llm-env/bin/activate
export PATH=${PWD}/llm-env/bin:$PATH
export CUDA_HOME=/pkgs/cuda-11.8

# Process input args
SCRIPT=$1
LOG_DIR=$2
LOG_PATH="${LOG_DIR}/log_${SLURM_JOB_ID}_rank_${SLURM_PROCID}.log"

echo "Placing logs in: ${LOG_DIR}"
echo "World size: ${SLURM_NTASKS}"
echo "Number of nodes: ${SLURM_NNODES}"
nvidia-smi
NUM_GPUs=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "GPUs per node: ${NUM_GPUs}"

# Make logging directories.
mkdir -p "${LOG_DIR}"

torchrun --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
        --rdzv-id $RDVZ_ID \
        --rdzv-backend c10d ${SCRIPT} >> ${LOG_PATH} 2>&1
