# Training

### Example job submission on vcluster for training.

```sh
bash job_submission/launch.sh \
LAUNCH_MODE=train \
CLUSTER_NAME=vcluster \
NNODES=1 \
GPUS_PER_NODE=2 \
CPUS_PER_GPU=6 \
GPU_TYPE=a40 \
QOS=normal \
TIME=02:00:00 \
MEM_PER_CPU=3 \
SCRIPT=src/squadv2_finetuning.py \
LOG_DIR=training_logs \
CONFIG_FILE=configs/gemma2_squadv2_8192_13_lora_flags.txt
```

### Example job submission on narval for training.

```sh
bash job_submission/launch.sh \
LAUNCH_MODE=train \
CLUSTER_NAME=narval \
NNODES=4 \
GPUS_PER_NODE=4 \
CPUS_PER_GPU=8 \
TIME=2-00:00:00 \
MEM_PER_CPU=4 \
ACCOUNT=rrg-afyshe \
SCRIPT=src/squadv2_finetuning.py \
LOG_DIR=training_logs \
CONFIG_FILE=configs/gemma2_squadv2_8192_13_lora_flags.txt
```

# Inference

When you launch the server, we need to give the model path and the max-log-probs for the vllm script.

| Models | VLLM_MAX_LOGPROBS |
|:----------:|:----------:|
| gemma2 | 256000 |
| llama3 | 128256 |


### Example job submission on narval for serving.

```sh
bash job_submission/launch.sh \
LAUNCH_MODE=server \
CLUSTER_NAME=narval \
NNODES=1 \
GPUS_PER_NODE=4 \
CPUS_PER_GPU=8 \
TIME=04:00:00 \
MEM_PER_CPU=4 \
ACCOUNT=rrg-afyshe \
LOG_DIR=server_logs \
VLLM_MODEL_WEIGHTS=/home/saeednjf/nearline/rrg-afyshe/pre-trained-models/gemma-2-9b-it \
VLLM_MAX_LOGPROBS=256000
```
