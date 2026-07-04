#!/bin/bash
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../../.." && pwd)
WORKSPACE_ROOT=$(cd "$REPO_ROOT/.." && pwd)

export WORKSPACE_ROOT
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export GPUS_PER_NODE=${PROC_PER_NODE:-${GPUS_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}}
export MASTER_PORT=${MASTER_PORT:-6001}
export NNODES=${NODE_COUNT:-${NNODES:-1}}
export NODE_RANK=${NODE_RANK:-0}
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
export PP_MODE=$PP_MODE
export OCTOPIPE_NVSHMEM_P2P=1
export NVSHMEM_REMOTE_TRANSPORT=${NVSHMEM_REMOTE_TRANSPORT:-none}
export ENABLE_OCTOPIPE_PROFILER=${ENABLE_OCTOPIPE_PROFILER:-0}
export CUDA_DEVICE_MAX_CONNECTIONS=1

if [ -n "$TIME" ]; then
    echo "Time set to: $TIME"
else
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

cd "$REPO_ROOT"
source "$SCRIPT_DIR/config.sh"
source "$SCRIPT_DIR/model_args.sh"

TRAIN_FILE="$REPO_ROOT/pretrain_mamba.py"
TENSORBOARD_DIR="${WORKSPACE_ROOT}/traces/${TIME}/${LOG_DIR_NAME}/"
configs+=(--tensorboard-dir $TENSORBOARD_DIR)
mkdir -p "$SCRIPT_DIR/results"

torchrun --nnodes $NNODES --nproc-per-node $GPUS_PER_NODE --node_rank $NODE_RANK --master-port $MASTER_PORT --master-addr $MASTER_ADDR $TRAIN_FILE \
    ${configs[@]} \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${LOG_ARGS[@]} \
    2>&1 | tee "$SCRIPT_DIR/results/$TIME.log"
