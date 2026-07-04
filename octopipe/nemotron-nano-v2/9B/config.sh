#!/bin/bash
JOB=NEMOTRON_NANO_V2
LOG_DIR_NAME="${JOB}"

EP_SIZE=1
PP_SIZE=4
TP_SIZE=1

SEQ_LEN=$((1024 * 4))
MICRO_BATCH_SIZE=1
GLOBAL_BATCH_SIZE=8
TRAIN_ITERS=100
EVAL_ITERS=0
EVAL_INTERVAL=1
LOG_INTERVAL=1
OVERLAP_WARMUP_FLUSH=""
VPP=""

RECOMP_GRANULARITY=""
RECOMP_METHOD="uniform"
RECOMP_LAYER=1
PP_MODE="octopipe"
OCTOPIPE_BWD_SPLITTING=True
OCTOPIPE_CONFIG_DIR="debug_config/nemotron"
OCTOPIPE_CONFIG_YAML="octopipe/nemotron-nano-v2/9B/octopipe_config.yaml"


PP_LAYOUT="E,t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4,L" 
PP_LAYOUT="Et|(t|)*30tL"
PP_LAYOUT=""

configs=(
    --num-workers 4
    --micro-batch-size $MICRO_BATCH_SIZE
    --seq-length $SEQ_LEN
    --data-path ${DATA_PATH:-${WORKSPACE_ROOT}/data/tinystories_deepseek_train_text_document}
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model ${TOKENIZER_MODEL:-${WORKSPACE_ROOT}/tokenizers/deepseek}
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --train-iters $TRAIN_ITERS
    --lr-decay-iters 320000
    --split 949,50,1
    --distributed-backend nccl
    --lr 0.00015
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 1e-2
    --clip-grad 1.0
    --lr-warmup-fraction .01
    --use-distributed-optimizer
    --use-flash-attn
    --transformer-impl transformer_engine
)

if [ -n "$RECOMP_GRANULARITY" ]; then
    configs+=(--recompute-activations)
    configs+=(--recompute-granularity $RECOMP_GRANULARITY)
fi

LOG_ARGS=(
    --log-memory-to-tensorboard
    --log-world-size-to-tensorboard
    --log-interval $LOG_INTERVAL
    --eval-iters $EVAL_ITERS
    --eval-interval $EVAL_INTERVAL
    --log-throughput
    --profile
    --profile-ranks 0 1 2 3
    --use-pytorch-profiler
    --profile-step-start 5
    --profile-step-end 6
    --timing-log-level 2
)

if [ "$TP_SIZE" -gt 1 ]; then
    configs+=(--tp-comm-overlap)
    configs+=(--sequence-parallel)
fi

LOG_DIR_NAME+="_PP${PP_SIZE}_TP${TP_SIZE}_SEQ${SEQ_LEN}"

if [ -n "$OVERLAP_WARMUP_FLUSH" ]; then
    configs+=(--overlap-p2p-communication-warmup-flush)
    echo "PP_WARMUP_P2P_OVERLAP=True"
    LOG_DIR_NAME="${LOG_DIR_NAME}_OverlapWarmupFlush"
fi 

if [ "$PP_SIZE" -gt 1 ]; then
    LOG_DIR_NAME+="_${PP_MODE}"

    if [ -n "$VPP" ]; then
        configs+=(--num-layers-per-virtual-pipeline-stage $VPP)
        LOG_DIR_NAME+="_${VPP}"
    fi

    if [ -n "$PP_LAYOUT" ]; then
        configs+=(--pipeline-model-parallel-layout "$PP_LAYOUT")
    fi

    if [ "$PP_MODE" = "octopipe" ]; then
        configs+=(--octopipe)
        if [ "$OCTOPIPE_BWD_SPLITTING" = "True" ]; then
            configs+=(--octopipe-bwd-splitting)
        fi
        configs+=(--octopipe-config-yaml $OCTOPIPE_CONFIG_YAML)
    fi
fi
