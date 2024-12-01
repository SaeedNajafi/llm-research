#!/bin/bash

# For reading key=value arguments and exporting as env variables.
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

MEM=$((MEM_PER_CPU*CPUS_PER_GPU*GPUS_PER_NODE))

export CURR_DIR="$(pwd)"

if [ "${LAUNCH_MODE}" = "train" ]; then
    if [ "${CLUSTER_NAME}" = "narval" ]; then
        if [ "${RUN_MODE}" = "tuning" ]; then
            ${COMMAND} --account ${ACCOUNT} \
                --nodes ${NNODES} \
                --gpus-per-node ${GPUS_PER_NODE} \
                --cpus-per-gpu ${CPUS_PER_GPU} \
                --mem ${MEM}G \
                --time ${TIME} \
                ${CURR_DIR}/job_submission/train.slrm ${CURR_DIR}/${SCRIPT} \
                ${CURR_DIR}/${CONFIG_FILE} ${CURR_DIR}/${LOG_DIR} ${NPROC_PER_NODE} \
                ${RUN_MODE} ${CHECKPOINT_FOLDER} ${PREDICTION_FILE} \
                ${GRADIENT_ACCUMULATION_STEPS} ${LR} ${LR_MIN} ${RUN_NAME}
        else
            ${COMMAND} --account ${ACCOUNT} \
                --nodes ${NNODES} \
                --gpus-per-node ${GPUS_PER_NODE} \
                --cpus-per-gpu ${CPUS_PER_GPU} \
                --mem ${MEM}G \
                --time ${TIME} \
                ${CURR_DIR}/job_submission/train.slrm ${CURR_DIR}/${SCRIPT} \
                ${CURR_DIR}/${CONFIG_FILE} ${CURR_DIR}/${LOG_DIR} ${NPROC_PER_NODE}
        fi

    elif [ "${CLUSTER_NAME}" = "vcluster" ]; then
        if [ "${RUN_MODE}" = "tuning" ]; then
            ${COMMAND} --nodes ${NNODES} \
                --gpus-per-node ${GPUS_PER_NODE} \
                --cpus-per-gpu ${CPUS_PER_GPU} \
                --mem ${MEM}G \
                --time ${TIME} \
                --partition ${GPU_TYPE} \
                --qos ${QOS} \
                ${CURR_DIR}/job_submission/train.slrm ${CURR_DIR}/${SCRIPT} \
                ${CURR_DIR}/${CONFIG_FILE} ${CURR_DIR}/${LOG_DIR} ${NPROC_PER_NODE} \
                ${RUN_MODE} ${CHECKPOINT_FOLDER} ${PREDICTION_FILE} \
                ${GRADIENT_ACCUMULATION_STEPS} ${LR} ${LR_MIN} ${RUN_NAME}
        else
            ${COMMAND} --nodes ${NNODES} \
                --gpus-per-node ${GPUS_PER_NODE} \
                --cpus-per-gpu ${CPUS_PER_GPU} \
                --mem ${MEM}G \
                --time ${TIME} \
                --partition ${GPU_TYPE} \
                --qos ${QOS} \
                ${CURR_DIR}/job_submission/train.slrm ${CURR_DIR}/${SCRIPT} \
                ${CURR_DIR}/${CONFIG_FILE} ${CURR_DIR}/${LOG_DIR} ${NPROC_PER_NODE}
        fi
    fi

elif [ "${LAUNCH_MODE}" = "server" ]; then
    if [ "${CLUSTER_NAME}" = "narval" ]; then
        ${COMMAND} --account ${ACCOUNT} \
            --nodes ${NNODES} \
            --gpus-per-node ${GPUS_PER_NODE} \
            --cpus-per-gpu ${CPUS_PER_GPU} \
            --mem ${MEM}G \
            --time ${TIME} \
            ${CURR_DIR}/job_submission/server.slrm ${CURR_DIR}/${LOG_DIR}

    elif [ "${CLUSTER_NAME}" = "vcluster" ]; then
        ${COMMAND} --nodes ${NNODES} \
            --gpus-per-node ${GPUS_PER_NODE} \
            --cpus-per-gpu ${CPUS_PER_GPU} \
            --mem ${MEM}G \
            --time ${TIME} \
            --partition ${GPU_TYPE} \
            --qos ${QOS} \
            ${CURR_DIR}/job_submission/server.slrm ${CURR_DIR}/${LOG_DIR}

    fi
fi
