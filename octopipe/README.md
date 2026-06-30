# OctoPipe

This directory contains the OctoPipe runtime configuration loader and example schedules used by this Megatron-LM fork. OctoPipe replaces the standard pipeline schedule with a schedule driven by logical stage workloads. It supports multiple logical stages per physical pipeline rank and can optionally use the NVSHMEM P2P communicator for pipeline activation and gradient traffic.

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
- use NVSHMEM P2P by default unless `OCTOPIPE_NVSHMEM_P2P=0`.

## Code Layout

```text
octopipe/
  generate_inst.py                 # Parses and builds runtime OctoPipe config
  debug_config/*/                  # Legacy partition/placement/result examples
  nemotronh/4B/octopipe_config.yaml # YAML example

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

Example from the workspace-level script `../../sh/nemotron-nano-v2/9B/config.sh`:

```bash
PP_MODE="octopipe"
OCTOPIPE_BWD_SPLITTING=True
OCTOPIPE_CONFIG_YAML="sh/nemotron-nano-v2/9B/octopipe_config.yaml"

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
- `OCTOPIPE_NVSHMEM_P2P=1` selects `forward_backward_pipelining_of_octopipe_nvshmem()`. This is the default when the variable is unset.
- `OCTOPIPE_NVSHMEM_P2P=0` selects the NCCL-based OctoPipe path.
- otherwise it selects `forward_backward_pipelining_of_octopipe()`.

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

## NVSHMEM P2P Mode

NVSHMEM P2P is enabled by default. To make the setting explicit:

```bash
export OCTOPIPE_NVSHMEM_P2P=1
```

Disable it with:

```bash
export OCTOPIPE_NVSHMEM_P2P=0
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
5. NVSHMEM P2P is default; export `OCTOPIPE_NVSHMEM_P2P=0` only if you want the NCCL path.
6. For multi-node NVSHMEM P2P, make sure `NVSHMEM_REMOTE_TRANSPORT` is not `none`.
7. Confirm `placement` uses pipeline group rank ids.

## References in This Repo

- `octopipe/generate_inst.py`
- `octopipe/debug_config/`
- `octopipe/nemotronh/4B/octopipe_config.yaml`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/pipeline_parallel/p2p_communication.py`
- `../../OctoPipeNvshmemCommunicator.md`
