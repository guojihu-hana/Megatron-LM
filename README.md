# OctoPipe: Reducing Pipeline Bubbles for Heterogeneous Models via Co-Optimizing Partitioning, Placement, and Scheduling (SC26)

This repository contains the OctoPipe runtime configuration loader and example schedules used by this Megatron-LM fork. OctoPipe replaces the standard pipeline schedule with a schedule driven by logical stage workloads. It supports multiple logical stages per physical pipeline rank and uses the NVSHMEM P2P communicator for pipeline activation and gradient traffic.

## Quick Start

1. Use the NVIDIA PyTorch container:

```bash
nvcr.io/nvidia/pytorch:25.12-py3
```

2. Install Megatron-LM.
3. Install NVSHMEM.
4. Run the Nemotron-Nano-v2 9B example (require 4 H800 GPU):

```bash
bash Megatron-LM/octopipe/nemotron-nano-v2/9B/run.sh
```

## What OctoPipe Adds

OctoPipe introduces three concepts on top of normal pipeline parallelism:

- **Logical stage (`sid`)**: a model segment in the OctoPipe partition.
- **Device / pipeline rank (`did`)**: the physical pipeline rank that owns one or more logical stages.
- **Workload schedule**: ordered forward, backward, and optional weight-gradient workloads for each logical stage and microbatch.

The training code uses the OctoPipe config to:

- build `sid -> did`, `did -> sid`, and `sid -> cid` mappings;
- select the model chunk for each local logical stage;
- choose the OctoPipe forward-backward schedule when `--octopipe` is enabled;
- optionally split TransformerEngine backward into `b` and `w` workloads with `--octopipe-bwd-splitting`;
- use the NVSHMEM P2P communicator by default for pipeline traffic.

## Code Layout

```text
octopipe/
  generate_inst.py                 # Parses and builds runtime OctoPipe config
  debug_config/*/                  # Legacy partition/placement/result examples
  nemotronh/4B/octopipe_config.yaml # YAML example
  nemotron-nano-v2/9B/             # Nemotron-Nano-v2 9B example scripts and config

megatron/core/pipeline_parallel/schedules.py
  forward_backward_pipelining_of_octopipe()
  forward_backward_pipelining_of_octopipe_nvshmem()

megatron/core/pipeline_parallel/p2p_communication.py
  NvshmemP2PCommunicator
  OctoPipeP2PCommunicator
```

## Enabling OctoPipe

OctoPipe is enabled with:

```bash
--octopipe
```

You must provide exactly one config source:

```bash
--octopipe-config-yaml path/to/octopipe_config.yaml
```

or:

```bash
--octopipe-config-dir debug_config/nemotron
```

`--octopipe-config-dir` is resolved under `Megatron-LM/octopipe/` and must contain:

```text
partition.txt
placement.txt
result.txt
```

`--octopipe-config-yaml` may be absolute, relative to the current working directory, relative to the Megatron-LM source root, or relative to the outer workspace root.

The Nemotron-Nano-v2 9B example is available under:

```bash
cd octopipe/nemotron-nano-v2/9B
bash run.sh
```

Example from `octopipe/nemotron-nano-v2/9B/config.sh`:

```bash
PP_MODE="octopipe"
OCTOPIPE_BWD_SPLITTING=True
OCTOPIPE_CONFIG_YAML="octopipe/nemotron-nano-v2/9B/octopipe_config.yaml"

configs+=(--octopipe)
configs+=(--octopipe-bwd-splitting)
configs+=(--octopipe-config-yaml $OCTOPIPE_CONFIG_YAML)
```

## Config Format

### YAML Format

YAML is the preferred format because it keeps partition, placement, and scheduling in one file.

```yaml
schedule: octopipe
partition: [2, 3, 4, 3]
placement: [[0, 2], [1, 3]]
scheduling:
  - (f, 0, 0, 0, 0, 695)
  - (f, 0, 1, 1, 695, 1407)
  - (b, 0, 3, 1, 3000, 3600)
```

Fields:

- `partition`: number of layers owned by each logical stage. `len(partition)` is the number of OctoPipe stages.
- `placement`: a 2D list indexed by `did`; each element lists the logical stages placed on that physical pipeline rank.
- `scheduling`: workload entries. Tuple form is `(type, mid, sid, did, start_time, end_time)`.

Workload types:

- `f`: forward compute.
- `b`: backward dgrad compute.
- `w`: delayed weight-gradient compute, only valid with `--octopipe-bwd-splitting`.

Important: `did` is treated as a **pipeline group rank**, not a per-node local GPU id. In multi-node runs, `did` values must be unique in the pipeline group, for example `0..pp_size-1`, not repeated as `0..7` on every node.

### Legacy Directory Format

The legacy directory format is still supported:

```text
partition.txt
placement.txt
result.txt
```

`partition.txt` contains a list of per-stage layer counts.

`placement.txt` contains the 2D placement list, for example:

```python
[[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]
```

`result.txt` contains one compute workload per line:

```text
f_0_0_0, 0, 695
f_0_1_1, 695, 1407
b_0_15_3, 15049, 16380
```

The token format is:

```text
<type>_<microbatch_id>_<stage_id>_<device_id>, <start_time>, <end_time>
```

## Runtime Config Built by `generate_inst.py`

`generate_inst.py` parses the input config and builds a runtime dict with:

- `sid->did`: logical stage to physical pipeline rank.
- `did->sid`: physical pipeline rank to local logical stages.
- `sid->cid`: logical stage to local model chunk index on that rank.
- `workloads`: compute plus inserted send/recv workloads.
- `comp_workloads`: compute-only workload order.
- `layout`: generated pipeline layout string.
- `layer_idx_offset`: layer offset for each logical stage.
- `partition`: original logical-stage partition.

The training path gets this config through `megatron.training.global_vars.get_octopipe_config()`.

## Schedule Selection

When pipeline parallel size is greater than 1:

- `--octopipe` selects the OctoPipe schedule.
- the default execution path is `forward_backward_pipelining_of_octopipe_nvshmem()`.

The NVSHMEM path is comp-driven: it parses send/recv relationships from the workload list, but the main loop executes only compute workloads. It receives the required tensor immediately before compute and sends the produced tensor immediately after compute.

## Backward Splitting

Enable backward splitting with:

```bash
--octopipe-bwd-splitting
```

Requirements and behavior:

- Requires `--octopipe`.
- Requires `--transformer-impl transformer_engine`.
- The schedule must contain `w` workloads.
- `b` workloads compute dgrad, while `w` workloads execute delayed TransformerEngine weight-gradient computation.

## NVSHMEM P2P Runtime

OctoPipe uses NVSHMEM P2P by default for performance:

```bash
export OCTOPIPE_NVSHMEM_P2P=1
```

Useful environment variables:

```bash
export NVSHMEM_MAX_CTAS=2
export OCTOPIPE_NVSHMEM_P2P_NUM_SLOTS=8
export OCTOPIPE_NVSHMEM_P2P_BUFFER_FACTOR=1
export OCTOPIPE_NVSHMEM_TRACE_MAX=-1
export OCTOPIPE_NVSHMEM_VALIDATE_EXPECTED_META=0
```

The communicator initializes NVSHMEM over the pipeline process group. It maps pipeline group ranks to NVSHMEM PE ids `0..pp_size-1`, then uses fixed-size symmetric staging slots for routed OctoPipe traffic.

Slot memory scales with:

```text
pp_world_size * OCTOPIPE_NVSHMEM_P2P_NUM_SLOTS * slot_bytes
```

If GPU memory is tight, reduce `OCTOPIPE_NVSHMEM_P2P_NUM_SLOTS` or explicitly size the slot with:

```bash
export OCTOPIPE_NVSHMEM_P2P_BUFFER_BYTES=<bytes>
```

### Multi-node Note

Do not force:

```bash
export NVSHMEM_REMOTE_TRANSPORT=none
```

for multi-node runs. That disables the NVSHMEM remote transport path. The code only sets this automatically when:

```bash
export OCTOPIPE_NVSHMEM_SINGLE_NODE=1
```

Some example scripts currently default `NVSHMEM_REMOTE_TRANSPORT` to `none`; override or remove that default before testing multi-node NVSHMEM P2P.

For multi-node runs, also verify:

- the OctoPipe `did` values are global pipeline ranks, not local GPU ids;
- NVSHMEM, RDMA/NIC, container device mounts, and transport environment are configured;
- each process uses the expected CUDA device from `LOCAL_RANK`;
- symmetric heap usage fits the chosen `pp_world_size`, slot count, and slot size.

## Debugging

Useful checks:

```bash
export OCTOPIPE_NVSHMEM_TRACE_MAX=100
export OCTOPIPE_NVSHMEM_VALIDATE_EXPECTED_META=1
```

Common failures:

- **Missing config**: `--octopipe` requires either `--octopipe-config-yaml` or `--octopipe-config-dir`.
- **Wrong placement**: every logical stage must appear exactly once in `placement`.
- **Wrong multi-node did**: repeated local ids can route tensors to the wrong PP rank.
- **Missing `w` workload**: `--octopipe-bwd-splitting` requires at least one `w` workload.
- **NVSHMEM OOM**: total symmetric slot memory grows with `pp_world_size`.
- **NVSHMEM hang at slot wait**: increase slots, inspect trace logs, or validate route metadata.
- **Multi-node NVSHMEM failure**: check that `NVSHMEM_REMOTE_TRANSPORT` is not `none` and that the cluster transport is configured.

## Minimal Run Checklist

1. Set normal Megatron pipeline parallel arguments, including `--pipeline-model-parallel-size`.
2. Add `--octopipe`.
3. Add exactly one config source: `--octopipe-config-yaml` or `--octopipe-config-dir`.
4. If the schedule contains `w` workloads, add `--octopipe-bwd-splitting` and use TransformerEngine.
5. Use the default NVSHMEM P2P path.
6. For multi-node NVSHMEM P2P, make sure `NVSHMEM_REMOTE_TRANSPORT` is not `none`.
7. Confirm `placement` uses pipeline group rank ids.

## References in This Repo

- `octopipe/generate_inst.py`
- `octopipe/debug_config/`
- `octopipe/nemotronh/4B/octopipe_config.yaml`
- `octopipe/nemotron-nano-v2/9B/run.sh`
- `octopipe/nemotron-nano-v2/9B/octopipe_config.yaml`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/pipeline_parallel/p2p_communication.py`
- `../OctoPipeNvshmemCommunicator.md`

---

<div align="center">

Megatron-LM and Megatron Core
=============================

<h4>GPU-optimized library for training transformer models at scale</h4>

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)
[![version](https://img.shields.io/badge/release-0.15.0-green)](./CHANGELOG.md)
[![license](https://img.shields.io/badge/license-Apache-blue)](./LICENSE)

<div align="left">

## About

This repository contains two components: **Megatron-LM** and **Megatron Core**.

**Megatron-LM** is a reference example that includes Megatron Core plus pre-configured training scripts, ideal for research teams, learning distributed training, and quick experimentation.

**Megatron Core** is a composable library with GPU-optimized building blocks for custom training frameworks. It provides transformer building blocks, advanced parallelism strategies (TP, PP, DP, EP, and CP), mixed precision support (FP16, BF16, FP8, and FP4), and model architectures, ideal for framework developers and ML engineers building custom training pipelines.

**[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** provides bidirectional Hugging Face ↔ Megatron checkpoint conversion with production-ready recipes.

## Getting Started

**Install from PyPI:**

```bash
uv pip install megatron-core
```

**Or clone and install from source:**

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
uv pip install -e .
```

> **Note:** Building from source can use a lot of memory. If the build runs out of memory, limit parallel compilation jobs by setting `MAX_JOBS` (for example, `MAX_JOBS=4 uv pip install -e .`).

For NVIDIA GPU Cloud (NGC) container setup and all installation options, review the **[Installation Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/install.html)**.

- **[Your First Training Run](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/quickstart.html)** - End-to-end training examples with data preparation
- **[Parallelism Strategies](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html)** - Scale training across GPUs with TP, PP, DP, EP, and CP
- **[Contribution Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)** - How to contribute to Megatron Core

# Latest News

- **[2026/05]** **[DeepSeek-V4 initial support](https://github.com/NVIDIA/Megatron-LM/issues/4468)** - Megatron Core's `dev` branch includes the initial DeepSeek-V4 implementation; Megatron Bridge provides [conversion, inference, and pretraining recipes](https://github.com/NVIDIA-NeMo/Megatron-Bridge/tree/main/examples/models/deepseek_v4).
- **[2026/04]** **[Advancing Emerging Optimizers for Accelerated LLM Training with NVIDIA Megatron](https://developer.nvidia.com/blog/advancing-emerging-optimizers-for-accelerated-llm-training-with-nvidia-megatron/)** - Muon and other emerging optimizers are now supported in Megatron Core via the new **[Emerging-Optimizers](https://github.com/NVIDIA-NeMo/Emerging-Optimizers)** library.
- **[2026/03]** **[Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/abs/2603.07685)** - Technical report on scaling MoE training with integrated optimizations for memory, communication, and computation.
- **[2026/03]** **[Implementing Falcon-H1 Hybrid Architecture in Megatron Core](https://developer.nvidia.com/blog/implementing-falcon-h1-hybrid-architecture-in-nvidia-megatron-core/)** - Technology Innovation Institute (TII) contributes Falcon-H1 hybrid transformer-Mamba architecture and BitNet ternary quantization support to Megatron Core.
- **[2026/03]** **[Megatron Core Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4003)** - Roadmap for upcoming Megatron Core features and improvements.
- **[2026/03]** **Deprecating Python 3.10 support:** The upcoming 0.17.0 release drops Python 3.10 support. Downstream applications must raise their lower boundary to 3.12 to stay compatible with Megatron Core.
- **[2026/01]** **[Dynamic Context Parallelism](https://developer.nvidia.com/blog/speeding-up-variable-length-training-with-dynamic-context-parallelism-and-nvidia-megatron-core/)** - Up to 1.48x speedup for variable-length sequence training with adaptive CP sizing.
- **[2025/12]** **Megatron Core development has moved to GitHub.** All development and CI now happen in the open, and community contributions are welcome.
- **[2025/10]** **[Megatron Dev Branch](https://github.com/NVIDIA/Megatron-LM/tree/dev)** - Early access branch with experimental features.
- **[2025/10]** **[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)** - Bidirectional converter for interoperability between Hugging Face and Megatron checkpoints, featuring production-ready recipes for popular models.
- **[2025/08]** **[Mixture of Experts (MoE) Q3–Q4 2025 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/1729)** - Comprehensive roadmap for MoE features including DeepSeek-V3, Qwen3, advanced parallelism strategies, FP8 optimizations, and Blackwell performance enhancements.
- **[2025/08]** **[GPT-OSS Model](https://github.com/NVIDIA/Megatron-LM/issues/1739)** - Megatron Core integrates advanced features including YaRN RoPE scaling, attention sinks, and custom activation functions.
- **[2025/06]** **[Megatron MoE Model Zoo](https://github.com/yanring/Megatron-MoE-ModelZoo)** - Best practices and optimized configurations for training DeepSeek-V3, Mixtral, and Qwen3 MoE models with performance benchmarking and checkpoint conversion tools.

[Previous News](docs/discussions/README.md#previous-news)

# Project Structure

```
Megatron-LM/
├── megatron/
│   ├── core/                    # Megatron Core (kernels, parallelism, building blocks)
│   │   ├── models/              # Transformer models
│   │   ├── transformer/         # Transformer building blocks
│   │   ├── tensor_parallel/     # Tensor parallelism
│   │   ├── pipeline_parallel/   # Pipeline parallelism
│   │   ├── distributed/         # Distributed training (FSDP, DDP)
│   │   ├── optimizer/           # Optimizers
│   │   ├── datasets/            # Dataset loaders
│   │   ├── inference/           # Inference engines and server
│   │   └── export/              # Model export (example: TensorRT-LLM)
│   ├── training/                # Training scripts
│   ├── legacy/                  # Legacy components
│   ├── post_training/           # Post-training (quantization, distillation, pruning, etc.)
│   └── rl/                      # Reinforcement learning (including RLHF)
├── examples/                    # Ready-to-use training examples
├── tools/                       # Utility tools
├── tests/                       # Comprehensive test suite
└── docs/                        # Documentation
```

# Performance Benchmarking

For the latest performance benchmarking results, refer to [NVIDIA Megatron Bridge Performance Summary](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-summary.html).

The codebase efficiently trains models from 2B to 462B parameters across thousands of GPUs, achieving up to **47% Model FLOP Utilization (MFU)** on H100 clusters.

![Model table](images/model_table.png)

**Benchmark Configuration:**

- **Vocabulary size**: 131,072 tokens
- **Sequence length**: 4,096 tokens
- **Model scaling**: Varied hidden size, attention heads, and layers to achieve target parameter counts
- **Communication optimizations**: Fine-grained overlapping with DP (`--overlap-grad-reduce`, `--overlap-param-gather`), TP (`--tp-comm-overlap`), and PP (enabled by default)

**Key Results:**

- **6,144 H100 GPUs**: Successfully benchmarked 462B parameter model training.
- **Superlinear scaling**: MFU increases from 41% to 47–48% with model size.
- **End-to-end measurement**: Throughputs include all operations (data loading, optimizer steps, communication, and logging).
- **Production ready**: Full training pipeline with checkpointing and fault tolerance.
- *Note: Performance results measured without training to convergence*

## Weak Scaling Results

The weak scaled results show superlinear scaling (MFU increases from 41% for the smallest model considered to 47–48% for the largest models); this is because larger GEMMs have higher arithmetic intensity and are consequently more efficient to execute.

![Weak scaling](images/weak_scaling.png)

## Strong Scaling Results

This test strong scales the standard GPT-3 model (slightly more than 175 billion parameters due to larger vocabulary size) from 96 H100 GPUs to 4,608 GPUs, using the same batch size of 1,152 sequences throughout. Communication becomes more exposed at larger scale, leading to a reduction in MFU from 47% to 42%.

![Strong scaling](images/strong_scaling.png)

# Roadmaps

- **[2026 Q2 Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4997)**
- **[2026 Q2 MoE-Specific Roadmap](https://github.com/NVIDIA/Megatron-LM/issues/4815)** [`dev` branch first developments]

# Resources

## Getting Help

- 📖 **[Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html)** - Official guides and API reference
- 🐛 **[Issues](https://github.com/NVIDIA/Megatron-LM/issues)** - Bug reports and feature requests

## Contributing

Contributions are welcome. Ways to contribute:

- 🐛 **Report bugs** - Help improve reliability
- 💡 **Suggest features** - Shape the future of Megatron Core
- 📝 **Improve docs** - Make Megatron Core more accessible
- 🔧 **Submit PRs** - Contribute code improvements

**→ [Contributing Guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/developer/contribute.html)**

## Citation

If you use Megatron in your research or project, use the following citation:

```bibtex
@article{megatron-lm,
  title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism},
  author={Shoeybi, Mohammad and Patwary, Mostofa and Puri, Raul and LeGresley, Patrick and Casper, Jared and Catanzaro, Bryan},
  journal={arXiv preprint arXiv:1909.08053},
  year={2019}
}
```
