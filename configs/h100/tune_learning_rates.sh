#!/bin/bash

main_path="/home/saeed/checkpoints/rl-dpo-mml-exps-december/squadv2_1024_13/"
gradient_accumulation_steps=(16)
lrs=(0.00001 0.00005 0.0001)
lr_mins=(0.000001 0.000005 0.00001)

export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

for g_step_i in ${!gradient_accumulation_steps[@]};
do
    g_step=${gradient_accumulation_steps[$g_step_i]}
    for lr_i in ${!lrs[@]};
    do
        lr=${lrs[$lr_i]}
        lr_min=${lr_mins[$lr_i]}
        run_name="iterative_samplesize_8-gradient_accu_steps_${g_step}-lr_${lr}-top_p_0.95-temp_1.0"
        NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" TOKENIZERS_PARALLELISM=false WANDB_MODE=offline torchrun --nproc-per-node=8 \
            --nnodes=1 \
            --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
            --rdzv-id $RDVZ_ID \
            --rdzv-backend c10d codes/llm-research/src/squadv2_finetuning.py \
            --flagfile codes/llm-research/configs/h100/llama3.2_iterative_squadv2_13_1024_lora_flags.txt \
            --checkpoint_folder ${main_path}/${run_name} \
            --prediction_file ${main_path}/${run_name}/internal_validation_prediction_squadv2.csv \
            --gradient_accumulation_steps ${g_step} \
            --compute_true_validation_loss=false \
            --lr ${lr} \
            --lr_min ${lr_min} \
            --run_name ${run_name} > codes/logs/${run_name}_logs.txt 2>&1
    done
done
