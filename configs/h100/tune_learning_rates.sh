#!/bin/bash

main_path="/home/saeed/checkpoints/rl-dpo-mml-exps-december/squadv2_1024_13/"
top_ps=(0.975 0.95 0.925 0.9 0.875 0.85)
temperatures=(2.0 1.75 1.5 1.25 1.0 0.75 0.5 0.25 0.0001)


export MASTER_ADDR="$(hostname --fqdn)"
export MASTER_PORT="$(python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1])')"
export RDVZ_ID=$RANDOM
echo "RDZV Endpoint $MASTER_ADDR:$MASTER_PORT"

for top_p_i in ${!top_ps[@]};
do
    top_p=${top_ps[$top_p_i]}
    for temp_i in ${!temperatures[@]};
    do
        temp=${temperatures[$temp_i]}
        run_name="mml_version_1_samplesize_8-gradient_accu_steps_16-lr_0.00005-top_p-${top_p}-temp-${temp}"
        NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES="0,1,2,3" TOKENIZERS_PARALLELISM=false WANDB_MODE=offline torchrun --nproc-per-node=4 \
            --nnodes=1 \
            --rdzv-endpoint $MASTER_ADDR:$MASTER_PORT \
            --rdzv-id $RDVZ_ID \
            --rdzv-backend c10d codes/llm-research/src/squadv2_finetuning.py \
            --flagfile codes/llm-research/configs/h100/llama3.2_mml_squadv2_13_1024_lora_flags.txt \
            --checkpoint_folder ${main_path}/${run_name} \
            --prediction_file ${main_path}/${run_name}/internal_validation_prediction_squadv2.csv \
            --num_epochs=3 \
            --train_top_p=${top_p} \
            --train_temperature=${temp} \
            --compute_true_validation_loss=false \
            --run_name ${run_name} > codes/logs/${run_name}_logs.txt 2>&1
    done
done
