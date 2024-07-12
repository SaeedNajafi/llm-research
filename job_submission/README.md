### Example job submission on vcluster

```sh
bash job_submission/launch_training.sh CLUSTER_NAME=vcluster \
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
