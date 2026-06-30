#!/bin/bash
MODEL_SIZE=4B
TIME=$1

export MODEL="nemotronh-4B"
export MASTER_ADDR=${MASTER_ADDR}
export GPUS_PER_NODE=$PROC_PER_NODE
export MASTER_PORT=6001
export NNODES=$NODE_COUNT
export NODE_RANK=$NODE_RANK
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export PP_MODE=$PP_MODE
# NVSHMEM PP P2P (schedules.NvshmemP2PCommunicator): export OCTOPIPE_NVSHMEM_P2P=1 before run
export OCTOPIPE_NVSHMEM_P2P=${OCTOPIPE_NVSHMEM_P2P:-1}
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}

if [ -n "$TIME" ]; then
    echo "Time set to: $TIME"
else
    #for rlaunch DEBUG: set params from actual GPU count
    TIME=$(date +"%Y-%m-%d-%H%M-%S")
    echo "Time now: $TIME"
    NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    if [ -z "$NUM_GPUS" ] || [ "$NUM_GPUS" -eq 0 ]; then
        echo "WARNING: no GPU detected, defaulting to 1"
        NUM_GPUS=1
    fi
    export MASTER_ADDR=${MASTER_ADDR:-localhost}
    export GPUS_PER_NODE=$NUM_GPUS
    export MASTER_PORT=6001
    export NNODES=1
    export NODE_RANK=0
    export WORLD_SIZE=$NUM_GPUS
    export PP_MODE=$PP_MODE
    echo "Using $NUM_GPUS GPU(s), WORLD_SIZE=$WORLD_SIZE"
fi

set -ex

export PYTHONWARNINGS="ignore"
# export NCCL_DEBUG="INFO"
# export TORCHDYNAMO_DISABLE=1 # 不禁用会报错 但不影响运行

cd /mnt/shared-storage-user/ailab-sys/guojihu/Megatron-LM
source sh/nemotronh/$MODEL_SIZE/config.sh
source sh/nemotronh/$MODEL_SIZE/model_args.sh

# conda
# source /mnt/shared-storage-user/ailab-sys/guojihu/miniconda/etc/profile.d/conda.sh
TRAIN_FILE=/mnt/shared-storage-user/ailab-sys/guojihu/Megatron-LM/Megatron-LM/pretrain_mamba.py

# profile stage time
# export CUDA_LAUNCH_BLOCKING=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

TENSORBOARD_DIR=/mnt/shared-storage-user/ailab-sys/guojihu/Megatron-LM/traces/${TIME}/${LOG_DIR_NAME}/
configs+=(--tensorboard-dir $TENSORBOARD_DIR)

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR $TRAIN_FILE \
    ${configs[@]} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOG_ARGS[@]} \
    2>&1 | tee sh/nemotronh/$MODEL_SIZE/results/$TIME.log
