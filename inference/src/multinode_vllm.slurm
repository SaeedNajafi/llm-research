#!/bin/bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --wait-all-nodes=1

date;pwd

nvidia-smi

module --force purge

source activate llm-env

export CUDA_HOME=$CONDA_PREFIX

export NCCL_HOME=$CONDA_PREFIX

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

# source ${SRC_DIR}/find_port.sh

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Find port for head node
# head_node_port=$(find_available_port $head_node_ip 8080 65535)
head_node_port=3456

# Starting the Ray head node
ip_head=$head_node_ip:$head_node_port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"

srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$head_node_port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &

# Starting the Ray worker nodes
# Optional, though may be useful in certain versions of Ray < 1.0.
sleep 10

# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
            ray start --address "$ip_head" \
            --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${NUM_GPUS}" --block &

    sleep 5
done

#vllm_port_number=$(find_available_port $head_node_ip 8080 65535)

vllm_port_number=3457

echo "Server address: http://${head_node_ip}:${vllm_port_number}/v1"
echo "http://${head_node_ip}:${vllm_port_number}/v1" > ${VLLM_BASE_URL_FILENAME}


# source ${VENV_BASE}/bin/activate
VLLM_ATTENTION_BACKEND=FLASHINFER python3 -m vllm.entrypoints.openai.api_server \
    --model ${VLLM_MODEL_WEIGHTS} \
    --host "0.0.0.0" \
    --port ${vllm_port_number} \
    --tensor-parallel-size $((NUM_NODES*NUM_GPUS)) \
    --dtype ${VLLM_DATA_TYPE} \
    --load-format safetensors \
    --trust-remote-code \
    --max-logprobs ${VLLM_MAX_LOGPROBS}
