#!/bin/bash
# srun --job-name=megatron --nodes=1 --ntasks-per-node=1 --gpus-per-task=8 --cpus-per-task=8 --time=0:10:0 --partition=llm_s 脚本.sh
GPUS_PER_TASK=8
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
MICRO_BATCH_SIZE=1   # global batch size = microbatch num * microbatch size * dp
GLOBAL_BATCH_SIZE=8 # 不设置时默认=microbatch size * dp，此时 microbatch num=1 
TRAIN_ITERS=100
EVAL_ITERS=0
EVAL_INTERVAL=1
LOG_INTERVAL=1
OVERLAP_WARMUP_FLUSH=""
VPP="" # 不设置时默认为空，不为空时表示vpp的layer数 

RECOMP_GRANULARITY="" # 空串时不重计算 非空时可选：[selective, full]
RECOMP_METHOD="uniform"
RECOMP_LAYER=1
PP_MODE="octopipe"
PP_MODE="1f1b"
OCTOPIPE_BWD_SPLITTING=True # 使用OctoPipe workloads来拆分backward into dgrad (b) and wgrad (w) phases.
OCTOPIPE_CONFIG_DIR="debug_config/nemotron"
OCTOPIPE_CONFIG_YAML="sh/nemotronh/4B/octopipe_config.yaml"


# --pipeline-model-parallel-layout 
PP_LAYOUT="E,t*4|t*4|t*4|t*4|t*4|t*4|t*4|t*4,L" 
PP_LAYOUT="Et|(t|)*30tL"
PP_LAYOUT=""

# MoE do not support fp16 (change to bf16), bias linear (disable bias linear)
# --data-path /cpfs01/user/guojihu/MegatronLM/Megatron-LM/dataset/data/gpt2_pretrain_text_document \
configs=(
    --num-workers 4
    --mock-data
    --tensor-model-parallel-size $TP_SIZE
    --pipeline-model-parallel-size $PP_SIZE
    --expert-model-parallel-size $EP_SIZE
    # --vocab-file /mnt/shared-storage-user/ailab-sys/guojihu/Megatron-LM/data/gpt2-vocab.json
    # --merge-file /mnt/shared-storage-user/ailab-sys/guojihu/Megatron-LM/data/gpt2-merges.txt
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
    # --profile-layer-time
    --profile-ranks 0 1 2 3
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
        configs+=(--pipeline-model-parallel-layout "$PP_LAYOUT")
    fi

    if [ "$PP_MODE" = "octopipe" ]; then
        configs+=(--octopipe)
        if [ "$OCTOPIPE_BWD_SPLITTING" = "True" ]; then
            configs+=(--octopipe-bwd-splitting)
        fi
        # configs+=(--octopipe-config-dir $OCTOPIPE_CONFIG_DIR)
        configs+=(--octopipe-config-yaml $OCTOPIPE_CONFIG_YAML)
    fi
fi