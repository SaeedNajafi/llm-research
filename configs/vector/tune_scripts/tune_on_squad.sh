#!/bin/bash

main_path="/scratch/ssd004/scratch/snajafi/vector-backup/rl-exps-november/squadv2_1024_13"
gradient_accumulation_steps=(1 2 4 8 16 32)
lrs=(0.000001 0.000005 0.00001 0.00005 0.0001 0.0005 0.001)
lr_mins=(0.0000001 0.0000005 0.000001 0.000005 0.00001 0.00005 0.0001)


for g_step_i in ${!gradient_accumulation_steps[@]};
do
    g_step=${gradient_accumulation_steps[$g_step_i]}
    for lr_i in ${!lrs[@]};
    do
        lr=${lrs[$lr_i]}
        lr_min=${lr_mins[$lr_i]}
        run_name=iterative_finetuning_samplesize_8-gradient_accu_steps_${g_step}-lr_${lr}
        checkpoint_folder=${main_path}/${run_name}
        prediction_file=${checkpoint_folder}/internal_validation_prediction_squadv2.csv
        bash job_submission/launch.sh \
            COMMAND=sbatch \
            LAUNCH_MODE=train \
            CLUSTER_NAME=vcluster \
            NNODES=1 \
            NPROC_PER_NODE=4 \
            GPUS_PER_NODE=4 \
            CPUS_PER_GPU=6 \
            GPU_TYPE=a40 \
            QOS=m3 \
            TIME=04:00:00 \
            MEM_PER_CPU=3 \
            SCRIPT=src/squadv2_finetuning.py \
            LOG_DIR=training_logs \
            CONFIG_FILE=configs/vector/llama3.2_iterative_squadv2_13_1024_lora_flags.txt \
            CHECKPOINT_FOLDER=${checkpoint_folder} \
            PREDICTION_FILE=${prediction_file} \
            GRADIENT_ACCUMULATION_STEPS=${g_step} \
            LR=${lr} \
            LR_MIN=${lr_min} \
            RUN_NAME=${run_name} \
            RUN_MODE=tuning
    done
done
