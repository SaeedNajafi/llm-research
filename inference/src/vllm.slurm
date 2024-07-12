#!/bin/bash

#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1

date;hostname;pwd

nvidia-smi

module --force purge

eval "$(conda shell.bash hook)"

conda activate llm-env

export CUDA_HOME=$CONDA_PREFIX
export NCCL_HOME=$CONDA_PREFIX
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# source ${SRC_DIR}/src/find_port.sh

# Write server url to file
hostname=${SLURMD_NODENAME}
echo $hostname
export MASTER_ADDR=$(hostname)
echo "r$SLURM_NODEID master: $MASTER_ADDR"
#vllm_port_number=$(find_available_port $hostname 8080 65535)
vllm_port_number=3456
echo "Server address: http://${hostname}:${vllm_port_number}/v1"
echo "http://${hostname}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}

VLLM_ATTENTION_BACKEND=FLASHINFER python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size ${NUM_GPUS} \
    --dtype ${VLLM_DATA_TYPE} \
    --max-logprobs ${VLLM_MAX_LOGPROBS}
