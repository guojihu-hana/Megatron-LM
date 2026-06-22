#!/bin/bash
JOB=NEMOTRONH
LOG_DIR_NAME="${JOB}"
# RuntimeError: Using async gradient all reduce requires setting the environment variable CUDA_DEVICE_MAX_CONNECTIONS to 1
export CUDA_DEVICE_MAX_CONNECTIONS=1
# tp-comp-overlap时
export UB_SKIPMC=1

# unset NCCL_DEBUG
# unset NCCL_DEBUG_SUBSYS

TOP_K=2
EXPERT_NUM="" # None means no MoE
MOE_FREQ=1
EP_SIZE=1
PP_SIZE=4
TP_SIZE=1
MOE_FFN_SIZE=$((1024*1))
SEQ_LEN=$((1024 * 4))
GLOBAL_BATCH_SIZE=$((2 * $PP_SIZE))
TRAIN_ITERS=20
EVAL_ITERS=0
EVAL_INTERVAL=1
LOG_INTERVAL=1

# Recomputation settings
RECOMP_GRANULARITY="" # 空串时不重计算 非空时可选：[selective, full]
RECOMP_METHOD="uniform"
RECOMP_LAYER=1

PP_MODE="octopipe" # 1f1b, octopipe

# OctoPipe settings
OCTOPIPE_CONFIG_DIR="debug_config/nemotron"

# 1F1B settings
VPP="" # set to 1 to enable Interleaved-1F1B
if [ "$PP_MODE" = "octopipe" ]; then
    VPP=""
fi
PP_LAYOUT="" # Set PP_LAYOUT for implementing Mist

# 'A string that describes a custom pipeline model parallel layout. '
# 'e.g., "E|(t|)*3,m|m||L". E, L, t, m denotes embedding, loss, transformer '
# 'decoder layer, and mtp layer, respectively. Stages are split by "|". '
# 'Replicated stages or layers can be described with multiplication. '
# 'Commas can be used cosmetically. '
# 'Default None is not using this argument to set the layout.'

configs=(
    --num-workers 4
    --mock-data
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
    --profile-ranks 0 1 2 3 4 5 6 7
    --use-pytorch-profiler
    --profile-step-start 5
    --profile-step-end 6
    --timing-log-level 2
)

MOE_ARGS=()
if [ -z "$EXPERT_NUM" ] || [ "$EXPERT_NUM" = "None" ]; then
    echo "No MoE"
else
    echo "EXPERT_NUM: $EXPERT_NUM"
    MOE_ARGS=(
        --num-experts $EXPERT_NUM
        --moe-layer-freq $MOE_FREQ
        --moe-ffn-hidden-size $MOE_FFN_SIZE
        --moe-router-topk $TOP_K
        --moe-router-load-balancing-type aux_loss
        --moe-aux-loss-coeff 1e-2
        --moe-token-dispatcher-type alltoall
        --overlap-param-gather
        --overlap-grad-reduce
        --moe-grouped-gemm
    )
    LOG_DIR_NAME+="_EP${EP_SIZE}"
fi

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
        configs+=(--pipeline-model-parallel-layout $PP_LAYOUT)
    fi

    if [ "$PP_MODE" = "octopipe" ]; then
        configs+=(--octopipe)
        configs+=(--octopipe-config-dir $OCTOPIPE_CONFIG_DIR)
    fi
fi