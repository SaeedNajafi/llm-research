#!/bin/bash

main_path="/scratch/ssd004/scratch/snajafi/vector-backup/rl-exps-november/squadv2_1024_13"
top_ps=(0.975 0.95 0.925 0.9 0.875 0.85)
temperatures=(2.0 1.75 1.5 1.25 1.0 0.75 0.5 0.25 0.0001)


for top_p_i in ${!top_ps[@]};
do
    top_p=${top_ps[$top_p_i]}
    for temp_i in ${!temperatures[@]};
    do
        temp=${temperatures[$temp_i]}
        run_name=mml_version_1_samplesize_8-gradient_accu_steps_${g_step}-lr_${lr}-top_p-${top_P}-temp-${temp}
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
            CONFIG_FILE=configs/vector/llama3.2_mml_squadv2_13_1024_lora_flags.txt \
            CHECKPOINT_FOLDER=${checkpoint_folder} \
            PREDICTION_FILE=${prediction_file} \
            RUN_NAME=${run_name} \
            TEMP=${temp} \
            TOP_P=${top_p} \
            GRADIENT_ACCUMULATION_STEPS=16 \
            LR=0.00005 \
            LR_MIN=0.000005 \
            RUN_MODE=tuning
    done
done
