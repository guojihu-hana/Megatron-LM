# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import atexit
import contextlib
import inspect
import os
from functools import partial
from typing import Callable, Dict, Iterator, List, Optional, Union

import torch
from torch.autograd.variable import Variable

from megatron.core import parallel_state
from megatron.core.enums import ModelType
from megatron.core.weight_gradient_store import WeightGradStore
from megatron.core.pipeline_parallel.fine_grained_activation_offload import (
    FineGrainedActivationOffloadingInterface as off_interface,
)
from megatron.core.pipeline_parallel.multimodule_communicator import MultiModulePipelineCommunicator
from megatron.core.pipeline_parallel.p2p_communication import (
    NvshmemP2PCommunicator,
    OctoPipeP2PCommunicator,
    P2PCommunicator,
)
from megatron.core.pipeline_parallel.utils import (
    is_pp_first_stage,
    is_pp_last_stage,
    is_vp_first_stage,
    is_vp_last_stage,
)
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)
from megatron.core.transformer.cuda_graphs import create_cudagraphs, set_current_microbatch
from megatron.core.transformer.moe.paged_stash import paged_stash_reset
from megatron.core.transformer.moe.router import MoEAuxLossAutoScaler
from megatron.core.utils import (
    drain_embedding_wgrad_compute,
    get_attr_wrapped_model,
    get_model_config,
    get_model_type,
    nvtx_range_pop,
    nvtx_range_push,
)

from .combined_1f1b import (
    combined_1f1b_schedule_for_interleaved_pipelining,
    combined_1f1b_schedule_for_no_pipelining,
)
from .hybrid_cp_schedule import hybrid_context_parallel_forward_backward

# Types
Shape = Union[List[int], torch.Size]


class OctoPipeStageTimeProfiler:
    """Low-overhead CUDA-event timing for OctoPipe f/b/w stage workloads.

    The profiler is intentionally env-gated.  It records CUDA events during
    sampled steps, aggregates completed events asynchronously, and prints one
    summary at process exit by default.
    """

    def __init__(self, pp_rank: int):
        self.pp_rank = pp_rank
        self.enabled = os.environ.get("ENABLE_OCTOPIPE_PROFILER", "0") == "1"
        self.interval = max(1, int(os.environ.get("OCTOPIPE_STAGE_TIME_INTERVAL", "1")))
        self.warmup = max(0, int(os.environ.get("OCTOPIPE_STAGE_TIME_WARMUP", "10")))
        self.step = 0
        self._capture = False
        self._events = []
        self._pending = []
        self._summary = {}
        self._finalized = False
        if self.enabled:
            atexit.register(self.finalize)

    def start_step(self):
        if not self.enabled:
            return
        self.step += 1
        self._drain_ready_events()
        self._capture = self.step > self.warmup and self.step % self.interval == 0
        self._events = []

    def begin(self, sid: int, wtype: str, mid: int):
        if not self.enabled or not self._capture:
            return None
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        token = (int(sid), str(wtype), int(mid), start, end)
        return token

    def end(self, token):
        if token is None:
            return
        token[4].record()
        self._events.append(token)

    def call(self, sid: int, wtype: str, mid: int, func, *args, **kwargs):
        token = self.begin(sid, wtype, mid)
        try:
            return func(*args, **kwargs)
        finally:
            self.end(token)

    def finish_step(self):
        if not self.enabled:
            return
        if self._capture and self._events:
            self._pending.append((self.step, self._events))
        self._events = []
        self._capture = False
        self._drain_ready_events()

    def finalize(self):
        if not self.enabled or self._finalized:
            return
        self._finalized = True
        self._drain_ready_events(force=True)
        self._print_summary()

    def _drain_ready_events(self, force: bool = False):
        if not self._pending:
            return
        remaining = []
        for step, events in self._pending:
            if force:
                try:
                    for _, _, _, _, end in events:
                        end.synchronize()
                except RuntimeError:
                    remaining.append((step, events))
                    continue
            elif not all(end.query() for _, _, _, _, end in events):
                remaining.append((step, events))
                continue
            self._accumulate_events(events)
        self._pending = remaining

    def _accumulate_events(self, events):
        for sid, wtype, _mid, start, end in events:
            elapsed_ms = start.elapsed_time(end)
            stats = self._summary.setdefault(sid, {}).setdefault(
                wtype, {"count": 0, "sum": 0.0, "min": float("inf"), "max": 0.0}
            )
            stats["count"] += 1
            stats["sum"] += elapsed_ms
            stats["min"] = min(stats["min"], elapsed_ms)
            stats["max"] = max(stats["max"], elapsed_ms)

    def _print_summary(self):
        if not self._summary:
            return
        try:
            global_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else -1
        except RuntimeError:
            global_rank = -1
        for sid in sorted(self._summary):
            parts = []
            for wtype in ("f", "b", "w"):
                stats = self._summary[sid].get(wtype)
                if stats is None:
                    continue
                avg = stats["sum"] / max(1, stats["count"])
                parts.append(
                    f"{wtype}:count={stats['count']} sum={stats['sum']:.3f}ms "
                    f"avg={avg:.3f}ms min={stats['min']:.3f}ms max={stats['max']:.3f}ms"
                )
            if parts:
                print(
                    f"[octopipe-stage-time][rank={global_rank} pp_rank={self.pp_rank} "
                    f"steps={self.step} sid={sid}] " + " | ".join(parts),
                    flush=True,
                )


def get_forward_backward_func(pp_size: Optional[int] = None, vp_size: Optional[int] = None):
    """Retrieves the appropriate forward_backward function given the
    configuration of parallel_state.

    Returns a function that will perform all of the forward and
    backward passes of the model given the pipeline model parallel
    world size and virtual pipeline model parallel world size in the
    global parallel_state.

    Note that if using sequence parallelism, the sequence length component of
    the tensor shape is updated to original_sequence_length /
    tensor_model_parallel_world_size.

    The function returned takes the following arguments:

    forward_step_func (required): A function that takes a data
        iterator and a model as its arguments and return the model's
        forward output and the loss function. The loss function should
        take one torch.Tensor and return a torch.Tensor of loss and a
        dictionary of string -> torch.Tensor.

        A third argument, checkpoint_activations_microbatch, indicates
        that the activations for this microbatch should be
        checkpointed. A None value for this argument indicates that
        the default from the configuration should be used. This is
        used when the
        num_microbatches_with_partial_activation_checkpoints is used.

        For example:

        def loss_func(loss_mask, output_tensor):
            losses = output_tensor.float()
            loss_mask = loss_mask.view(-1).float()
            loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

            # Reduce loss for logging.
            averaged_loss = average_losses_across_data_parallel_group([loss])

            return loss, {'lm loss': averaged_loss[0]}

        def forward_step(data_iterator, model):
            data, loss_mask = next(data_iterator)
            output = model(data)
            return output, partial(loss_func, loss_mask)


        forward_backward_func(forward_step_func=forward_step, ...)


    data_iterator (required): an iterator over the data, will be
        passed as is to forward_step_func. Expected to be a list of
        iterators in the case of interleaved pipeline parallelism.

    model (required): the actual model. Expected to be a list of modules in the case of interleaved
        pipeline parallelism. Must be a (potentially wrapped) megatron.core.models.MegatronModule.

    num_microbatches (int, required):
        The number of microbatches to go through

    seq_length (int, required): Sequence length of the current global batch. If this is a dual-stack
        transformer, this is the encoder's sequence length. This is ignored if variable_seq_lengths
        in the config is True. Otherwise, each microbatch in the current global batch size must use
        this sequence length.

    micro_batch_size (int, required): The number of sequences in a microbatch.

    decoder_seq_length (int, optional): The sequence length for the decoder in a dual-stack
        transformer. This is ignored for a single-stack transformer.

    forward_only (optional, default = False): Perform only the forward step.

    collect_non_loss_data (optional, bool, default=False): TODO.

    first_val_step (bool, optional): Is the first step of the validation phase. Used by
        Transformer Engine modules to only update their fp8 weights only on the first validation
        step.

    adjust_tensor_shapes_fn (Callable, optional): A function that adjusts the receive and send
        tensor shapes. Only applicable in forward_backward_pipelining_without_interleaving for now.
        Takes in a list of receive shapes and a list of send shapes and returns the adjusted
        respective list of shapes. Thus it is not used in the other forward-backward functions
        which have different shape handling.

    force_all_reduce (bool, optional): If true, force use of all-reduce for gradient reduction
        instead of reduce-scatter (if using distributed optimizer) in this iteration to ensure all
        data-parallel ranks have fully reduced gradients. This is useful for easier wgrad saving
        (can just inspect DP replica 0 to get full set of wgrads for entire model).

    Args:
        pp_size (Optional[int]): Pipeline model parallel size to use.
        vp_size (Optional[int]): Virtual pipeline model parallel size to use.
            If both pp_size and vp_size are None, both values fall back to parallel_state.
            Otherwise, provided values are used as-is and None is treated as an explicit input.

    """
    if pp_size is None and vp_size is None:
        pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

    if pp_size > 1:
        from megatron.training import get_args
        args = get_args()
        if getattr(args, 'octopipe', False):
            if os.environ.get("MEGATRON_NVSHMEM_P2P", "0") == "1":
                forward_backward_func = forward_backward_pipelining_of_octopipe_nvshmem
            else:
                forward_backward_func = forward_backward_pipelining_of_octopipe
        elif vp_size is not None:
            forward_backward_func = forward_backward_pipelining_with_interleaving
        else:
            if parallel_state.get_virtual_pipeline_model_parallel_world_size() is not None:
                forward_backward_func = forward_backward_pipelining_with_interleaving
            else:
                forward_backward_func = forward_backward_pipelining_without_interleaving
    else:
        forward_backward_func = forward_backward_no_pipelining
    return forward_backward_func


def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.

    Supports multiple formats:
    - torch.Tensor: Deallocates the tensor directly
    - List[Tensor]: Recursively deallocates each element
    - Dict[str, Tensor]: Recursively deallocates each value (for multi-module pipelines)
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return

    # Handle dict format (multi-module pipelines)
    if isinstance(out, dict):
        for value in out.values():
            deallocate_output_tensor(value, deallocate_pipeline_outputs)
        return

    # Handle list format
    if isinstance(out, list):
        for item in out:
            deallocate_output_tensor(item, deallocate_pipeline_outputs)
        return

    # Base case: deallocate tensor
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)


def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''

    # assert output.numel() == 1, "output should be pseudo-'freed' in schedule, to optimize memory"
    assert isinstance(output, torch.Tensor), "output == '%s'." % type(output).__name__
    assert isinstance(grad_output, (torch.Tensor, type(None))), (
        "grad_output == '%s'." % type(grad_output).__name__
    )

    # Handle scalar output
    if grad_output is None:
        assert output.numel() == 1, "implicit grad requires scalar output."
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call c++ engine [ see torch/csrc/autograd/python_engine.cpp ]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )


def get_tensor_device(tensor: Union[torch.Tensor, Dict[str, torch.Tensor]]):
    """Get the device of a tensor or a dictionary of tensors."""
    if isinstance(tensor, dict):
        return next(iter(tensor.values())).device
    return tensor.device


def _get_mtp_loss_scale(config, device: torch.device) -> torch.Tensor:
    """Get the MTP loss scale on the output tensor device."""

    def _normalize_loss_scale(loss_scale, scale_func_name: str) -> torch.Tensor:
        loss_scale = torch.as_tensor(loss_scale, device=device)
        if loss_scale.numel() != 1:
            raise ValueError(
                f"{scale_func_name} must return a scalar or size-1 tensor for MTP loss scaling, "
                f"but returned a tensor with {loss_scale.numel()} elements."
            )
        return loss_scale

    mtp_grad_scale_func = getattr(config, 'mtp_grad_scale_func', None)
    if mtp_grad_scale_func is not None:
        return _normalize_loss_scale(mtp_grad_scale_func(), "mtp_grad_scale_func")
    if config.grad_scale_func is not None:
        return _normalize_loss_scale(
            config.grad_scale_func(torch.ones(1, device=device)), "grad_scale_func"
        )
    return torch.ones(1, device=device)


def forward_step_calc_loss(
    model,
    output_tensor,
    loss_func,
    config,
    vp_stage,
    collect_non_loss_data,
    num_microbatches,
    forward_data_store,
    cp_group_size=None,
    is_last_stage=None,
):
    """Calculate the loss and number of tokens for forward_step()"""

    from megatron.core.transformer.experimental_attention_variant.dsa import (
        DSAIndexerLossAutoScaler,
    )
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    model_vp_stage = getattr(model, "vp_stage", None)
    if vp_stage is not None and model_vp_stage is not None:
        assert (
            vp_stage == model_vp_stage
        ), f"vp_stage ({vp_stage}) doesn't match model_vp_stage ({model_vp_stage})"

    if cp_group_size is None and is_last_stage is None:
        # fallback to parallel state
        cp_group_size = parallel_state.get_context_parallel_world_size()
        is_last_stage = parallel_state.is_pipeline_last_stage(
            ignore_virtual=False, vp_stage=vp_stage
        )
    else:
        assert is_last_stage is not None, "is_last_stage must be provided"
        if is_last_stage:
            assert cp_group_size is not None, "cp_group_size must be provided on last stage"

    num_tokens = torch.tensor(0, dtype=torch.int)
    if is_last_stage:
        if loss_func is None:
            forward_data_store.append(output_tensor)
        elif not collect_non_loss_data:
            outputs = loss_func(output_tensor)
            if len(outputs) == 3:
                output_tensor, num_tokens, loss_reduced = outputs
                if not config.calculate_per_token_loss:
                    # Protect against division by zero when all tokens are masked
                    #   in a microbatch.
                    output_tensor /= torch.clamp(num_tokens, min=1)
                    output_tensor /= num_microbatches
            else:
                # preserve legacy loss averaging behavior (ie, over the number of microbatches)
                assert len(outputs) == 2
                output_tensor, loss_reduced = outputs
                output_tensor *= cp_group_size
                output_tensor /= num_microbatches
            forward_data_store.append(loss_reduced)
        else:
            data = loss_func(output_tensor, non_loss_data=True)
            forward_data_store.append(data)

    if config.timers is not None:
        config.timers('forward-compute').stop()

    # Set the loss scale for the auxiliary loss of the MoE layer.
    # Since we use a trick to do backward on the auxiliary loss, we need to set the scale
    # explicitly.
    if hasattr(config, 'num_moe_experts') and config.num_moe_experts is not None:
        # Calculate the loss scale based on moe_grad_scale_func (preferred),
        # grad_scale_func (fallback), or default to 1.
        device = get_tensor_device(output_tensor)
        moe_grad_scale_func = getattr(config, 'moe_grad_scale_func', None)
        if moe_grad_scale_func is not None:
            loss_scale = moe_grad_scale_func()
        elif config.grad_scale_func is not None:
            loss_scale = config.grad_scale_func(torch.ones(1, device=device))
        else:
            loss_scale = torch.ones(1, device=device)
        # Set the loss scale
        if config.calculate_per_token_loss:
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale)
        else:
            cp_size_for_scaling = cp_group_size if cp_group_size is not None else 1
            MoEAuxLossAutoScaler.set_loss_scale(loss_scale * cp_size_for_scaling / num_microbatches)

    # Set the loss scale for Multi-Token Prediction (MTP) loss.
    if hasattr(config, 'mtp_num_layers') and config.mtp_num_layers is not None:
        # Calculate the loss scale based on mtp_grad_scale_func if available,
        # else fall back to grad_scale_func, else default to 1.
        device = get_tensor_device(output_tensor)
        loss_scale = _get_mtp_loss_scale(config, device)
        # Set the loss scale
        if config.calculate_per_token_loss:
            MTPLossAutoScaler.set_loss_scale(loss_scale)
        else:
            MTPLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    # Set the loss scale for DSA (Dynamic Sparse Attention) indexer loss.
    if getattr(config, 'experimental_attention_variant', None) == 'dsa':
        loss_scale = (
            config.grad_scale_func(torch.ones(1, device=output_tensor.device))
            if config.grad_scale_func is not None
            else torch.ones(1, device=output_tensor.device)
        )
        if config.calculate_per_token_loss:
            DSAIndexerLossAutoScaler.set_loss_scale(loss_scale)
        else:
            DSAIndexerLossAutoScaler.set_loss_scale(loss_scale / num_microbatches)

    return output_tensor, num_tokens


def forward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size,
    collect_non_loss_data=False,
    checkpoint_activations_microbatch=None,
    is_first_microbatch=False,
    current_microbatch=None,
    vp_stage=None,
    is_last_stage=True,
):
    """Forward step for passed-in model.

    If it is the first stage, the input tensor is obtained from the data_iterator.
    Otherwise, the passed-in input_tensor is used.

    Args:
        forward_step_func (callable):
            The forward step function for the model that takes the
            data iterator as the first argument, and model as the second.
            This user's forward step is expected to output a tuple of two elements:

                1. The output object from the forward step. This output object needs to be a
                    tensor or some kind of collection of tensors. The only hard requirement
                    for this object is that it needs to be acceptible as input into the second
                    function.
                2. A function to reduce (optionally) the output from the forward step. This
                    could be a reduction over the loss from the model, it could be a function that
                    grabs the output from the model and reformats, it could be a function that just
                    passes through the model output. This function must have one of the following
                    patterns, and depending on the pattern different things happen internally:

                        a. A tuple of reduced loss and some other data. Note that in this case
                            the first argument is divided by the number of global microbatches,
                            assuming it is a loss, so that the loss is stable as a function of
                            the number of devices the step is split across.
                        b. A triple of reduced loss, number of tokens, and some other data. This
                            is similar to case (a), but the loss is further averaged across the
                            number of tokens in the batch. If the user is not already averaging
                            across the number of tokens, this pattern is useful to use.
                        c. Any arbitrary data the user wants (eg a dictionary of tensors, a list
                            of tensors, etc in the case of inference). To trigger case 3 you need
                            to specify `collect_non_loss_data=True` and you may also want to
                            specify `forward_only=True` in the call to the parent forward_backward
                            function.
        data_iterator (iterator):
            The data iterator.
        model (nn.Module):
            The model to perform the forward step on.
        num_microbatches (int):
            The number of microbatches.
        input_tensor (Tensor or list[Tensor]):
            The input tensor(s) for the forward step.
        forward_data_store (list):
            The list to store the forward data. If you go down path 2.a or
            2.b for the return of your forward reduction function then this will store only the
            final dimension of the output, for example the metadata output by the loss function.
            If you go down the path of 2.c then this will store the entire output of the forward
            reduction function applied to the model output.
        config (object):
            The configuration object.
        collect_non_loss_data (bool, optional):
            Whether to collect non-loss data. Defaults to False.
            This is the path to use if you want to collect arbitrary output from the model forward,
            such as with inference use cases. Defaults to False.
        checkpoint_activations_microbatch (int, optional):
            The microbatch to checkpoint activations.
            Defaults to None.
        is_first_microbatch (bool, optional):
            Whether it is the first microbatch. Defaults to False.
        current_microbatch (int, optional):
            The current microbatch. Defaults to None.
        vp_stage (int, optional):
            The virtual pipeline stage. Defaults to None.
        is_last_stage (bool, optional):
            Whether it is the last stage. Defaults to True.
            Also considering virtual stages.
            In case of PP/VPP, is_last_stage/is_vp_last_stage.

    Returns:
        Tensor or list[Tensor]: The output object(s) from the forward step.
        Tensor: The number of tokens.
    """
    from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler

    if config.timers is not None:
        config.timers('forward-compute', log_level=2).start()

    if is_first_microbatch and hasattr(model, 'set_is_first_microbatch'):
        model.set_is_first_microbatch()
    if current_microbatch is not None:
        set_current_microbatch(model, current_microbatch)

    unwrap_output_tensor = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_output_tensor = True

    set_input_tensor = get_attr_wrapped_model(model, "set_input_tensor")
    set_input_tensor(input_tensor)

    if config.enable_autocast:
        context_manager = torch.autocast("cuda", dtype=config.autocast_dtype)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        if checkpoint_activations_microbatch is None:
            _fsig = inspect.signature(forward_step_func)
            if "is_last_stage" in _fsig.parameters:
                output_tensor, loss_func = forward_step_func(
                    data_iterator, model, is_last_stage=is_last_stage
                )
            else:
                output_tensor, loss_func = forward_step_func(data_iterator, model)
        else:
            _fsig = inspect.signature(forward_step_func)
            if "is_last_stage" in _fsig.parameters:
                output_tensor, loss_func = forward_step_func(
                    data_iterator,
                    model,
                    checkpoint_activations_microbatch,
                    is_last_stage=is_last_stage,
                )
            else:
                output_tensor, loss_func = forward_step_func(
                    data_iterator, model, checkpoint_activations_microbatch
                )
    output_tensor, num_tokens = forward_step_calc_loss(
        model,
        output_tensor,
        loss_func,
        config,
        vp_stage,
        collect_non_loss_data,
        num_microbatches,
        forward_data_store,
        cp_group_size,
        is_last_stage,
    )

    if unwrap_output_tensor:
        return output_tensor, num_tokens
    return [output_tensor], num_tokens


def backward_step(input_tensor, output_tensor, output_tensor_grad, config):
    """Backward step through passed-in output tensor.

    If last stage, output_tensor_grad is None, otherwise gradient of loss
    with respect to stage's output tensor.

    Returns gradient of loss with respect to input tensor (None if first stage)."""

    # NOTE: This code currently can handle at most one skip connection. It
    # needs to be modified slightly to support arbitrary numbers of skip
    # connections.

    if config.timers is not None:
        config.timers('backward-compute', log_level=2).start()

    # Retain the grad on the input_tensor.
    unwrap_input_tensor_grad = False
    if not isinstance(input_tensor, list):
        input_tensor = [input_tensor]
        unwrap_input_tensor_grad = True
    for x in input_tensor:
        if x is not None:
            x.retain_grad()

    if not isinstance(output_tensor, list):
        output_tensor = [output_tensor]
    if not isinstance(output_tensor_grad, list):
        output_tensor_grad = [output_tensor_grad]

    # Backward pass.
    if output_tensor_grad[0] is None and config.grad_scale_func is not None:
        output_tensor[0] = config.grad_scale_func(output_tensor[0])

    # In multi-modal models like VLM, some batches may not have images.
    # When no image is present, the vision encoder (as a separate pipeline stage)
    # will not participate in the computation.
    # This results in a tensor that does not require gradients.
    # In such cases, we intentionally skip the backward pass while preserving zero gradients.
    if output_tensor[0].requires_grad:
        if config.deallocate_pipeline_outputs:
            custom_backward(output_tensor[0], output_tensor_grad[0])
        else:
            torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])

    # Collect the grad of the input_tensor.
    input_tensor_grad = [None]
    if input_tensor is not None:
        input_tensor_grad = []
        for x in input_tensor:
            if x is None:
                input_tensor_grad.append(None)
            else:
                input_tensor_grad.append(x.grad)

    if unwrap_input_tensor_grad:
        input_tensor_grad = input_tensor_grad[0]

    if config.timers is not None:
        config.timers('backward-compute').stop()

    return input_tensor_grad


def _prepare_octopipe_bwd_splitting(config, workloads):
    """Initialize WeightGradStore for OctoPipe backward splitting."""
    WeightGradStore.reset()
    if not getattr(config, "octopipe_bwd_splitting", False):
        return False

    has_w_workload = any(
        workload.get("op") == "comp" and workload.get("type") == "w" for workload in workloads
    )
    if not has_w_workload:
        raise ValueError(
            "--octopipe-bwd-splitting requires at least one 'w' workload in the OctoPipe schedule."
        )

    WeightGradStore.enable_split_bw()
    return True


def _register_octopipe_wgrad_task(model_chunk, chunk=0, tag=None):
    """Queue the chunk-level TE delayed weight-gradient computation."""
    if not WeightGradStore.split_bw():
        return
    backward_dw_modules = _collect_octopipe_backward_dw_modules(model_chunk)
    if not backward_dw_modules:
        raise RuntimeError(
            "--octopipe-bwd-splitting requires model chunks to contain modules "
            "that expose backward_dw()."
        )

    def run_backward_dw():
        for module in reversed(backward_dw_modules):
            module.backward_dw()

    description = ",".join(type(module).__name__ for module in backward_dw_modules)
    WeightGradStore.put_task(run_backward_dw, description=description, chunk=chunk, tag=tag)
    WeightGradStore.flush(chunk=chunk, tag=tag)


def _collect_octopipe_backward_dw_modules(model_chunk):
    """Return top-level modules whose backward_dw should be called for a chunk."""
    modules = list(get_attr_wrapped_model(model_chunk, "modules")())
    selected = []
    skipped_descendants = set()
    for module in modules:
        module_id = id(module)
        if module_id in skipped_descendants:
            continue
        backward_dw = getattr(module, "backward_dw", None)
        if not callable(backward_dw):
            continue
        selected.append(module)
        skipped_descendants.update(id(descendant) for descendant in module.modules())
    return selected


def backward_step_multimodule(
    input_tensor: Dict[str, torch.Tensor],
    output_tensor: Union[torch.Tensor, Dict[str, torch.Tensor]],
    output_tensor_grad: Optional[Dict[str, torch.Tensor]],
    config,
    language_model_module_name: str,
) -> Dict[str, torch.Tensor]:
    """Backward step for multi-module pipelines.

    In multi-module pipelines, tensors are organized as dictionaries with
    module names as keys. Each module's backward pass is performed independently.
    """

    def _unwrap_single_tensor_list(tensor):
        if isinstance(tensor, list):
            assert len(tensor) == 1, "expected a single tensor for multimodule backward"
            return tensor[0]
        return tensor

    # Retain gradients on all input tensors.
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is not None:
            tensor.retain_grad()

    # Last stage: output_tensor is a scalar loss from the language model.
    # Associate it with the language_model_module_name.
    if not isinstance(output_tensor, dict):
        output_tensor = {language_model_module_name: output_tensor}

    # Handle output_tensor_grad: None (last stage) or dict (intermediate stages).
    if not output_tensor_grad:
        output_tensor_grad = {key: None for key in output_tensor.keys()}

    # Apply grad scaling if needed (for last stage only).
    for module_name in output_tensor.keys():
        output_tensor_grad_module = _unwrap_single_tensor_list(output_tensor_grad[module_name])
        if output_tensor_grad_module is None and config.grad_scale_func is not None:
            output_tensor[module_name] = config.grad_scale_func(output_tensor[module_name])

    # Perform backward pass for each module.
    for module_name in output_tensor.keys():
        output_tensor_module = _unwrap_single_tensor_list(output_tensor[module_name])
        output_tensor_grad_module = _unwrap_single_tensor_list(output_tensor_grad[module_name])

        # In multi-modal models like VLM, some batches may not have images.
        # In such cases, skip backward while preserving zero gradients.
        if output_tensor_module is not None and output_tensor_module.requires_grad:
            if config.deallocate_pipeline_outputs:
                custom_backward(output_tensor_module, output_tensor_grad_module)
            else:
                torch.autograd.backward(
                    output_tensor_module, grad_tensors=output_tensor_grad_module
                )

    # Collect gradients for input tensors.
    input_tensor_grad = {}
    for module_name, tensor in input_tensor.items():
        if isinstance(tensor, list):
            tensor = tensor[0]
        if tensor is None:
            input_tensor_grad[module_name] = None
        else:
            input_tensor_grad[module_name] = tensor.grad

    return input_tensor_grad


def check_first_val_step(first_val_step, forward_only, cond):
    """Check if it is the first validation step."""
    if (first_val_step is not None) and forward_only:
        return first_val_step and cond
    else:
        return cond


def _octopipe_f_comp(
    *,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches: int,
    input_tensor,
    forward_data_store,
    config,
    cp_group_size: int,
    collect_non_loss_data: bool,
    checkpoint_activations_microbatch,
    is_first_microbatch: bool,
    current_microbatch: int,
    is_last_stage: bool,
    profiler,
    sid: int,
    wtype: str,
    mid: int,
):
    """Run one OctoPipe forward compute workload, optionally timed by CUDA events."""
    if profiler is None:
        return forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_group_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=is_first_microbatch,
            current_microbatch=current_microbatch,
            is_last_stage=is_last_stage,
        )
    return profiler.call(
        sid,
        wtype,
        mid,
        forward_step,
        forward_step_func,
        data_iterator,
        model,
        num_microbatches,
        input_tensor,
        forward_data_store,
        config,
        cp_group_size=cp_group_size,
        collect_non_loss_data=collect_non_loss_data,
        checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        is_first_microbatch=is_first_microbatch,
        current_microbatch=current_microbatch,
        is_last_stage=is_last_stage,
    )


def _octopipe_b_comp(
    *,
    input_tensor,
    output_tensor,
    output_tensor_grad,
    config,
    octopipe_bwd_splitting: bool,
    model_chunk,
    chunk: int,
    profiler,
    sid: int,
    wtype: str,
    mid: int,
):
    """Run one OctoPipe backward compute workload, optionally timed by CUDA events."""
    if profiler is None:
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        if octopipe_bwd_splitting:
            _register_octopipe_wgrad_task(model_chunk, chunk=chunk, tag=(sid, mid))
        return input_tensor_grad

    def _run_backward():
        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)
        if octopipe_bwd_splitting:
            _register_octopipe_wgrad_task(model_chunk, chunk=chunk, tag=(sid, mid))
        return input_tensor_grad

    return profiler.call(sid, wtype, mid, _run_backward)


def _octopipe_w_comp(
    *,
    chunk: int,
    seq_split_idx: int = 0,
    strict: bool = True,
    profiler,
    sid: int,
    wtype: str,
    mid: int,
):
    """Run one OctoPipe weight-gradient compute workload, optionally timed by CUDA events."""
    if profiler is None:
        return WeightGradStore.pop(
            chunk=chunk, seq_split_idx=seq_split_idx, strict=strict, tag=(sid, mid)
        )
    return profiler.call(
        sid,
        wtype,
        mid,
        WeightGradStore.pop,
        chunk=chunk,
        seq_split_idx=seq_split_idx,
        strict=strict,
        tag=(sid, mid),
    )


def forward_backward_no_pipelining(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,  # unused
    micro_batch_size: int,  # unused
    decoder_seq_length: Optional[int] = None,  # unused
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,  # unused
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run forward and backward passes with no pipeline parallelism"""

    if pg_collection is None:
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)
        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif pg_collection is not None:
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"

    if isinstance(model, list):
        assert len(model) == 1, "non-pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for non-pipeline-parallel schedule"

    config = get_model_config(model)
    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext

    model_type = get_model_type(model)

    forward_data_store = []
    input_tensor, output_tensor_grad = None, None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if config.overlap_moe_expert_parallel_comm and not forward_only:
        forward_data_store, total_num_tokens = combined_1f1b_schedule_for_no_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            partial(check_first_val_step, first_val_step, forward_only),
        )
    elif config.hybrid_context_parallel:
        forward_data_store, total_num_tokens = hybrid_context_parallel_forward_backward(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            output_tensor_grad,
            forward_data_store,
            config,
            collect_non_loss_data,
            first_val_step,
            forward_only,
            no_sync_func,
            total_num_tokens,
            check_first_val_step,
            model_type,
        )
    else:
        with no_sync_func():
            for i in range(num_microbatches - 1):
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator,
                    model,
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    pg_collection.cp.size(),
                    collect_non_loss_data,
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
                    current_microbatch=i,
                )
                total_num_tokens += num_tokens
                if not forward_only:
                    backward_step(input_tensor, output_tensor, output_tensor_grad, config)
                    # Release the autograd graph head before the next forward_step.
                    # Without this, the previous microbatch's output_tensor stays
                    # live until the next iteration rebinds the variable, deferring
                    # autograd-node teardown onto the next forward's dispatch path
                    # and triggering PyTorch's "AccumulateGrad node's stream does
                    # not match" warning. See issue #4124.
                    del output_tensor
        # Run computation for last microbatch out of context handler (want to
        # synchronize gradients).
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            pg_collection.cp.size(),
            collect_non_loss_data,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, num_microbatches == 1
            ),
            current_microbatch=num_microbatches - 1,
        )

        total_num_tokens += num_tokens

        if not forward_only:
            backward_step(input_tensor, output_tensor, output_tensor_grad, config)
            del output_tensor

    if config.finalize_model_grads_func is not None and not forward_only:
        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism and layernorm all-reduce for sequence parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Reset all_gather_pipeline bucket status before next validation iteration
    if forward_only:
        for model_chunk in [model]:
            if (
                hasattr(model_chunk, 'ddp_config')
                and model_chunk.ddp_config.use_megatron_fsdp
                and model_chunk.ddp_config.overlap_param_gather
            ):
                model_chunk.synchronize_param_gather()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    return forward_data_store


def clear_embedding_activation_buffer(config, model, is_last_stage):
    """Clear embedding activation buffer."""

    if is_last_stage and config.defer_embedding_wgrad_compute:
        if isinstance(model, list):
            embedding_module = get_attr_wrapped_model(
                model[-1], 'post_process', return_model_obj=True
            )
        else:
            embedding_module = get_attr_wrapped_model(model, 'post_process', return_model_obj=True)

        # Need to ensure no stray activations exists in this buffer
        embedding_module.embedding_activation_buffer.clear()

        return embedding_module
    else:
        return None


def finish_embedding_wgrad_compute(config, embedding_module, is_last_stage, tp_group):
    """Finish embedding wgrad compute."""
    if is_last_stage and config.defer_embedding_wgrad_compute:
        embedding_activation_buffer = embedding_module.embedding_activation_buffer
        grad_output_buffer = embedding_module.grad_output_buffer
        weight = (
            embedding_module.output_layer.weight
            if embedding_module.share_embeddings_and_output_weights
            else embedding_module.shared_embedding_or_output_weight()
        )

        drain_embedding_wgrad_compute(
            config, embedding_activation_buffer, grad_output_buffer, weight, tp_group
        )


def get_pp_rank_microbatches(
    num_microbatches,
    num_model_chunks,
    microbatch_group_size_per_vp_stage,
    forward_only=False,
    overlap_moe_expert_parallel_comm=False,
    p2p_communicator: Optional[P2PCommunicator] = None,
):
    """Get the number of total, warmup, and remaining microbatches in PP scheduling."""
    if p2p_communicator is not None:
        pipeline_parallel_size = p2p_communicator.pp_group.size()
        pipeline_parallel_rank = p2p_communicator.pp_group.rank()
        virtual_pipeline_parallel_size = p2p_communicator.virtual_pipeline_model_parallel_size
    else:
        pipeline_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()
        pipeline_parallel_rank = parallel_state.get_pipeline_model_parallel_rank()
        virtual_pipeline_parallel_size = (
            parallel_state.get_virtual_pipeline_model_parallel_world_size()
        )

    total_num_microbatches = num_microbatches * num_model_chunks
    are_all_microbatches_in_warmup = False

    if forward_only:
        num_warmup_microbatches = total_num_microbatches
    elif pipeline_parallel_size > 1:
        if virtual_pipeline_parallel_size is None:
            # forward_backward_pipelining_without_interleaving
            num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
        else:
            # forward_backward_pipelining_with_interleaving
            # Run (num_model_chunks-1)*microbatch_group_size_per_vp_stage on
            # all workers, followed by more microbatches after depending on
            # stage ID (more forward passes for earlier stages, later stages can
            # immediately start with 1F1B).
            num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
            num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
            # When enabling overlap_moe_expert_parallel_comm, we schedule one extra micro-batch
            # forward step before the 1f1b stages. This is needed to ensure the forward
            # and backward computations are independent in all 1f1b steps.
            if overlap_moe_expert_parallel_comm:
                num_warmup_microbatches = num_warmup_microbatches + 1
    else:
        # forward_backward_no_pipelining
        # This path is only used for cuda graph capturing compatibility for the PP=1 case.
        num_warmup_microbatches = 0

    if num_warmup_microbatches >= total_num_microbatches:
        num_warmup_microbatches = total_num_microbatches
        are_all_microbatches_in_warmup = True
    num_microbatches_remaining = total_num_microbatches - num_warmup_microbatches

    return (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    )


def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table


def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,  # unused
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""

    # Convention used in this function:
    # num_microbatches for number of microbatches per pipeline stage;
    # num_model_chunks for virtual pipeline size;
    # then total_num_microbatches = num_microbatches * num_model_chunks.
    # Their corresponding index variables are
    # microbatch_id in [0, num_microbatches)
    # model_chunk_id in [0, num_model_chunks)
    # virtual_microbatch_id in [0, total_num_microbatches)

    config = get_model_config(model[0])
    if p2p_communicator is None and pg_collection is None:
        p2p_communicator = P2PCommunicator(
            pp_group=parallel_state.get_pipeline_model_parallel_group(), config=config
        )
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.cp = cp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.pp = pp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        cp_size = cp_group.size()
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection"
            " provide none or provide all the process groups"
        )

    assert isinstance(model, list), "interleaved pipeline parallelism expected model chunking"
    assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    assert isinstance(
        data_iterator, list
    ), "interleaved pipeline parallelism expected each model chunk to have a data iterator"
    assert (
        adjust_tensor_shapes_fn is None
    ), "adjust_tensor_shapes_fn is not supported for interleaved pipeline parallelism"

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    if config.overlap_p2p_comm and config.batch_p2p_comm:
        raise ValueError("Can not use both overlap_p2p_comm and batch_p2p_comm")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        # vp is ignored for clear_embedding_activation_buffer
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if isinstance(no_sync_func, list):

        def multi_no_sync():
            stack = contextlib.ExitStack()
            for model_chunk_no_sync_func in config.no_sync_func:
                stack.enter_context(model_chunk_no_sync_func())
            return stack

        no_sync_func = multi_no_sync
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    if config.grad_sync_func is not None and not isinstance(config.grad_sync_func, list):
        config.grad_sync_func = [config.grad_sync_func for _ in model]

    if config.param_sync_func is not None and not isinstance(config.param_sync_func, list):
        config.param_sync_func = [config.param_sync_func for _ in model]

    # Disable config.grad_sync_func and config.param_sync_func if only running forward passes.
    # They will be re-enabled at the end of this function.
    grad_sync_func, param_sync_func = None, None
    if forward_only:
        grad_sync_func, param_sync_func = config.grad_sync_func, config.param_sync_func
        config.grad_sync_func, config.param_sync_func = None, None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Model chunk IDs with synchronized grads
    synchronized_model_chunks = set()

    input_tensors = [[] for _ in range(len(model))]
    output_tensors = [[] for _ in range(len(model))]
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    forward_data_store = []
    output_tensor_grads = None
    if not forward_only:
        output_tensor_grads = [[] for _ in range(len(model))]
    else:
        output_tensor_grads = None

    pipeline_parallel_size = p2p_communicator.pp_group.size()
    pipeline_parallel_rank = p2p_communicator.pp_group.rank()

    if (
        config.microbatch_group_size_per_vp_stage > num_microbatches
        or config.microbatch_group_size_per_vp_stage < pipeline_parallel_size
    ):
        msg = (
            'The number of contiguous micro-batches in a virtual pipeline stage'
            f'should range in [PP={pipeline_parallel_size} , M={num_microbatches}]'
        )
        raise ValueError(msg)

    # If the final micro-batch group has fewer micro-batches than pipeline-parallel size,
    # the pipeline will have dependency bubbles.
    final_microbatch_group_size = num_microbatches % config.microbatch_group_size_per_vp_stage
    if 0 < final_microbatch_group_size < pipeline_parallel_size:
        msg = 'The remainder of M (the total micro-batches) divided by N (number of '
        msg += 'contiguous micro-batches in a virtual pipeline stage) should be 0, '
        msg += 'or larger than or equal to the pipeline-parallel size, but it is '
        msg += f'{final_microbatch_group_size}. '
        msg += 'Otherwise, it introduces dependency bubbles in the pipeline '
        msg += 'and reduces throughput.'
        raise RuntimeError(msg)

    model_type = get_model_type(model[0])

    tensor_shape = [seq_length, micro_batch_size, config.hidden_size]
    tensor_shape[0] = tensor_shape[0] // cp_group.size()
    if config.sequence_parallel:
        tensor_shape[0] = tensor_shape[0] // tp_group.size()

    # Compute number of warmup and remaining microbatches.
    # seems only used for vpp
    num_model_chunks = len(model)
    (
        total_num_microbatches,
        are_all_microbatches_in_warmup,
        num_warmup_microbatches,
        num_microbatches_remaining,
    ) = get_pp_rank_microbatches(
        num_microbatches,
        num_model_chunks,
        config.microbatch_group_size_per_vp_stage,
        forward_only=forward_only,
        overlap_moe_expert_parallel_comm=config.overlap_moe_expert_parallel_comm,
        p2p_communicator=p2p_communicator,
    )

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Synchronize params for first two model chunks
    if config.param_sync_func is not None:
        config.param_sync_func[0](model[0].parameters())
        config.param_sync_func[1](model[1].parameters())

    # Create a tunable schedule lookup table.
    # The schedule lookup table uses the virtual_microbatch_id to find the corresponding
    # microbatch_id and model_chunk_id. For example, the tunable schedule table for
    # PP2 N3M5 with VP2 is constructed as below:
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    schedule_table = get_schedule_table(
        num_microbatches, len(model), config.microbatch_group_size_per_vp_stage
    )

    # Decouple individual lookup table for microbatch_id and model_chunk_id.
    # For example, the micro-batch table for PP2 N3M5 with VP2 is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # microbatch_id         | 0 1 2 0 1 2 3 4 3 4
    # Similarly, the model chunk table is
    # virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
    # model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
    # Both tables are indexed with virtual_microbatch_id.
    microbatch_id_table, model_chunk_id_table = zip(*schedule_table)

    def get_model_chunk_id(virtual_microbatch_id, forward):
        """Helper method to get the model chunk ID given the iteration number."""
        model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
        if not forward:
            model_chunk_id = num_model_chunks - model_chunk_id - 1
        return model_chunk_id

    def get_microbatch_id_in_model_chunk(iteration_id, forward):
        """Helper method to get the microbatch_id within model chunk given the iteration number."""
        assert forward
        microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
        return microbatch_id_in_model_chunk

    def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
        """Helper method to count number of released (i.e. popped from input_tensors)
        microbatches for a model chunk."""
        if forward_only:  # Micro-batch is released after forward prop.
            return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
        else:  # Micro-batch is released after backward prop.
            # Zero backward prop in warmup.
            if virtual_microbatch_id < num_warmup_microbatches:
                return 0
            else:
                backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
                model_chunk_id = num_model_chunks - model_chunk_id - 1
                return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)

    def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the first for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == 0
        else:
            return False

    def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
        """Check if an iteration is the last for a model chunk."""
        if virtual_microbatch_id < total_num_microbatches:
            return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
        else:
            return False

    def recv_tensor_from_previous_stage(virtual_microbatch_id, forward):
        """Determine if peers are sending, and where in data structure
        to put received tensors.
        Return a boolean if the pipeline stage expects to recv from peers, and the
        corresponding model_chunk_id for the received tensor.
        """
        recv = True
        # The leading pipeline stage is the first rank in fwd and the last rank in bwd.
        is_leading_pipeline_stage = (
            is_pp_first_stage(p2p_communicator.pp_group)
            if forward
            else is_pp_last_stage(p2p_communicator.pp_group)
        )

        last_model_chunk = (num_model_chunks - 1) if forward else 0

        if is_leading_pipeline_stage:
            # The leading pipeline stage is ahead of the ending pipeline stage
            # (i.e. last rank in fwd and first rank in bwd) by (pipeline_parallel_size - 1).
            # Let's consider bwd as an example with PP 4:
            #       0 1 2 3 ...
            #     0 1 2 3 ...
            #   0 1 2 3 ...
            # 0 1 2 3 ...
            if virtual_microbatch_id < (pipeline_parallel_size - 1):
                # The ending stage has not produced any tensors, so no recv will be initiated.
                recv = False
                next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)
            else:
                # Find the model chunk of the aligned microbatches in the ending stage.
                # For example, microbatch 0 in the ending stage is aligned with microbatch 3
                # in the leading stage.
                next_model_chunk_id = get_model_chunk_id(
                    virtual_microbatch_id - (pipeline_parallel_size - 1), forward
                )
            # Last model chunk in the final stage does not produce tensors.
            if next_model_chunk_id == last_model_chunk:
                recv = False
            if forward:
                # Model chunk id increases in forward.
                next_model_chunk_id += 1
            else:
                # Model chunk id decreases in backward.
                next_model_chunk_id -= 1
        else:
            next_model_chunk_id = get_model_chunk_id(virtual_microbatch_id + 1, forward)

        return recv, next_model_chunk_id

    def forward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id, microbatch_id):
        """Preprocess for forward_step_helper"""
        # launch param synchronization for next model chunk
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.param_sync_func is not None:
            param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
            if (
                param_sync_virtual_microbatch_id < total_num_microbatches
                and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
            ):
                param_sync_chunk_id = (
                    get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
                )
                if 1 < param_sync_chunk_id < num_model_chunks:
                    config.param_sync_func[param_sync_chunk_id](
                        model[param_sync_chunk_id].parameters()
                    )

        # forward step
        if _is_vp_first_stage(vp_stage=model_chunk_id) and is_pp_first_stage(pp_group):
            if len(input_tensors[model_chunk_id]) == len(output_tensors[model_chunk_id]):
                input_tensors[model_chunk_id].append(None)

        # For non-depth-first pipeline schedules, the first rank would buffer multiple received
        # activation tensors for a model chunk until accessed during warmup.
        # This input buffering is needed to overlap the computation with the receipt of
        # the next inputs. To index the proper buffered inputs for forword_step, we use
        # microbatch_id offset with number of released microbatches that have completed backprop.
        offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
        input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]

        return input_tensor

    def forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens):
        """Postprocess for forward_step_helper"""
        output_tensors[model_chunk_id].append(output_tensor)

        nonlocal total_num_tokens
        total_num_tokens += num_tokens

        # If forward-only, no need to save tensors for a backward pass.
        if forward_only:
            # Release the tensor that have completed forward step.
            input_tensors[model_chunk_id].pop(0)
            output_tensors[model_chunk_id].pop()

        return

    def forward_step_helper(virtual_microbatch_id, checkpoint_activations_microbatch):
        """Helper method to run forward step with model split into chunks"""
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=True)
        microbatch_id = get_microbatch_id_in_model_chunk(virtual_microbatch_id, forward=True)

        input_tensor = forward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id, microbatch_id
        )

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator[model_chunk_id],
            model[model_chunk_id],
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step,
                forward_only,
                is_first_microbatch_for_model_chunk(virtual_microbatch_id),
            ),
            current_microbatch=microbatch_id,
            vp_stage=model_chunk_id,
            is_last_stage=_is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group),
        )

        forward_step_helper_postprocess(model_chunk_id, output_tensor, num_tokens)

        return output_tensor

    def backward_step_helper_preprocess(virtual_microbatch_id, model_chunk_id):
        """Preprocess for backward_step_helper"""
        # launch grad synchronization (default)
        if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
            virtual_microbatch_id
        ):
            enable_grad_sync()
            synchronized_model_chunks.add(model_chunk_id)

        # pylint: disable=E0606
        if _is_vp_last_stage(vp_stage=model_chunk_id) and is_pp_last_stage(pp_group):
            if len(output_tensor_grads[model_chunk_id]) == 0:
                output_tensor_grads[model_chunk_id].append(None)
        input_tensor = input_tensors[model_chunk_id].pop(0)
        output_tensor = output_tensors[model_chunk_id].pop(0)
        output_tensor_grad = output_tensor_grads[model_chunk_id].pop(0)

        return input_tensor, output_tensor, output_tensor_grad

    def backward_step_helper_postprocess(virtual_microbatch_id):
        """Postprocess for backward_step_helper"""
        # launch grad synchronization (custom grad sync)
        # Note: Asynchronous communication tends to slow down compute.
        # To reduce idling from mismatched microbatch times, we launch
        # asynchronous communication at the same time across the
        # pipeline-parallel group.
        if config.grad_sync_func is not None:
            grad_sync_virtual_microbatch_id = virtual_microbatch_id - pipeline_parallel_rank
            if grad_sync_virtual_microbatch_id >= 0 and is_last_microbatch_for_model_chunk(
                grad_sync_virtual_microbatch_id
            ):
                grad_sync_chunk_id = get_model_chunk_id(
                    grad_sync_virtual_microbatch_id, forward=False
                )
                enable_grad_sync()
                config.grad_sync_func[grad_sync_chunk_id](model[grad_sync_chunk_id].parameters())
                synchronized_model_chunks.add(grad_sync_chunk_id)
        disable_grad_sync()

    def backward_step_helper(virtual_microbatch_id):
        """Helper method to run backward step with model split into chunks"""
        nonlocal output_tensor_grads
        model_chunk_id = get_model_chunk_id(virtual_microbatch_id, forward=False)

        input_tensor, output_tensor, output_tensor_grad = backward_step_helper_preprocess(
            virtual_microbatch_id, model_chunk_id
        )

        input_tensor_grad = backward_step(input_tensor, output_tensor, output_tensor_grad, config)

        backward_step_helper_postprocess(virtual_microbatch_id)

        return input_tensor_grad

    def forward_backward_helper_wrapper(
        f_virtual_microbatch_id=None,
        b_virtual_microbatch_id=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
        checkpoint_activations_microbatch=None,
    ):
        """
        wrap forward_helper, backward_helper, and combined_forward_backward_helper in a unified way
        """
        if config.overlap_moe_expert_parallel_comm and not forward_only:  # Combined 1F1B path
            return combined_1f1b_schedule_for_interleaved_pipelining(
                config,
                forward_step_func,
                data_iterator,
                model,
                num_microbatches,
                forward_data_store,
                forward_step_helper_preprocess,
                forward_step_helper_postprocess,
                backward_step_helper_preprocess,
                backward_step_helper_postprocess,
                get_microbatch_id_in_model_chunk,
                get_model_chunk_id,
                partial(check_first_val_step, first_val_step, forward_only),
                is_first_microbatch_for_model_chunk,
                collect_non_loss_data,
                f_virtual_microbatch_id=f_virtual_microbatch_id,
                b_virtual_microbatch_id=b_virtual_microbatch_id,
                pre_forward=pre_forward,
                pre_backward=pre_backward,
                post_forward=post_forward,
                post_backward=post_backward,
            )
        else:  # Conventional interleaved 1F1B path
            forward_output_tensor = None
            backward_input_tensor_grad = None
            # forward pass
            if f_virtual_microbatch_id is not None:
                forward_model_chunk_id = get_model_chunk_id(f_virtual_microbatch_id, forward=True)
                if pre_forward is not None:
                    pre_forward()
                forward_output_tensor = forward_step_helper(
                    f_virtual_microbatch_id, checkpoint_activations_microbatch
                )
                if post_forward is not None:
                    forward_output_tensor = post_forward(forward_output_tensor)

            # Backward pass.
            if b_virtual_microbatch_id is not None:
                backward_model_chunk_id = get_model_chunk_id(b_virtual_microbatch_id, forward=False)
                if pre_backward is not None:
                    pre_backward()
                backward_input_tensor_grad = backward_step_helper(b_virtual_microbatch_id)
                if post_backward is not None:
                    backward_input_tensor_grad = post_backward(backward_input_tensor_grad)
            return forward_output_tensor, backward_input_tensor_grad

    # ==============================main logic=========================================
    _is_vp_first_stage = partial(
        is_vp_first_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    _is_vp_last_stage = partial(
        is_vp_last_stage, vp_size=config.virtual_pipeline_model_parallel_size
    )
    pp_group = p2p_communicator.pp_group

    # Run warmup forward passes.
    nvtx_range_push(suffix="warmup")
    input_tensors[0].append(
        p2p_communicator.recv_forward(
            tensor_shape, _is_vp_first_stage(vp_stage=0) and is_pp_first_stage(pp_group)
        )
    )

    fwd_wait_handles = None
    fwd_wait_recv_handles = None
    bwd_wait_handles = None
    bwd_wait_recv_handles = None
    if is_pp_first_stage(p2p_communicator.pp_group):
        fwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        fwd_recv_buffer_size = 1
    if is_pp_last_stage(p2p_communicator.pp_group):
        bwd_recv_buffer_size = (
            config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
        )
    else:
        bwd_recv_buffer_size = 1
    fwd_recv_buffer = [None] * fwd_recv_buffer_size
    bwd_recv_buffer = [None] * bwd_recv_buffer_size
    recv_prev_wait_handles = []
    send_next_wait_handle = None
    send_prev_wait_handle = None
    recv_next_wait_handles = []

    for k in range(num_warmup_microbatches):
        cur_model_chunk_id = get_model_chunk_id(k, forward=True)

        if config.overlap_p2p_comm_warmup_flush:
            if (
                not (
                    _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group)
                )
                and k != 0
            ):
                assert recv_prev_wait_handles, (
                    f'pp rank {pipeline_parallel_rank}, iteration {k},'
                    'should have registered recv handle'
                )
                recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                recv_prev_wait_handle.wait()

        # Determine if tensor should be received from previous stage.
        recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

        # No receive in last iteration when recv iteration k+1.
        if k == (total_num_microbatches - 1):
            recv_prev = False

        # Prefetch recv for iteration k+1 for non-first ranks.
        if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(
            p2p_communicator.pp_group
        ):
            fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
                p2p_communicator.send_forward_recv_forward(
                    output_tensor=None,  # No output_tensor to send.
                    recv_prev=recv_prev,
                    tensor_shape=tensor_shape,
                    overlap_p2p_comm=True,
                )
            )

            if fwd_wait_recv_handles:
                recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        output_tensor, _ = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Don't send tensor downstream if on last stage.
        if _is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group):
            output_tensor = None

        # Send and receive tensors as appropriate (send tensors computed
        # in this iteration; receive tensors for next iteration).
        if not config.overlap_p2p_comm_warmup_flush:
            if (
                k == (num_warmup_microbatches - 1)
                and not config.overlap_p2p_comm
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False
                (input_tensor, output_tensor_grad) = (
                    p2p_communicator.send_forward_backward_recv_forward_backward(
                        output_tensor,
                        input_tensor_grad,
                        recv_prev=recv_prev,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                    )
                )
                output_tensor_grads[num_model_chunks - 1].append(output_tensor_grad)
            else:
                input_tensor = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
                )
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
        else:
            if not is_pp_first_stage(p2p_communicator.pp_group):
                # Send only since recv prefetched.
                _, fwd_wait_handles = p2p_communicator.send_forward_recv_forward(
                    output_tensor, recv_prev=False, tensor_shape=tensor_shape, overlap_p2p_comm=True
                )
            else:  # No prefetch for first rank, so both send and recv initiated.
                fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
            if send_next_wait_handle is not None:
                send_next_wait_handle.wait()
            if fwd_wait_handles is not None:
                send_next_wait_handle = (
                    fwd_wait_handles.pop("send_next") if "send_next" in fwd_wait_handles else None
                )
                if "recv_prev" in fwd_wait_handles:
                    recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
            # isend() copies asynchronously; wait until the copy is done before
            # freeing the source buffer, otherwise the next PP stage gets corrupted data.
            if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                send_next_wait_handle.wait()
                send_next_wait_handle = None

            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(
                    fwd_recv_buffer[k % fwd_recv_buffer_size]
                )
                fwd_recv_buffer[(k + 1) % fwd_recv_buffer_size] = None

        if config.overlap_p2p_comm:
            if (
                k == (num_warmup_microbatches - 1)
                and not forward_only
                and not are_all_microbatches_in_warmup
            ):
                input_tensor_grad = None
                recv_next = True
                if is_pp_last_stage(p2p_communicator.pp_group):
                    recv_next = False

                (bwd_recv_buffer[-1], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                if recv_next:
                    output_tensor_grads[num_model_chunks - 1].append(bwd_recv_buffer[-1])
    nvtx_range_pop(suffix="warmup")

    # Run 1F1B in steady state.
    nvtx_range_push(suffix="steady")
    for k in range(num_microbatches_remaining):
        # Forward pass.
        forward_k = k + num_warmup_microbatches

        # Decide to checkpoint all layers' activations of the current micro-batch.
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                forward_k % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        cur_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
        if config.overlap_p2p_comm:

            backward_k = k

            # Sync forward recv
            def pp_pre_forward(vp_stage=None):
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_prev_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, fwd iteration {forward_k}, '
                            'should have registered recv handle'
                        )
                        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                        recv_prev_wait_handle.wait()
                    else:
                        if recv_prev_wait_handles is not None and recv_prev_wait_handles:
                            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
                            recv_prev_wait_handle.wait()

                deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Async forward send / receive
            def pp_post_forward(output_tensor, vp_stage=None):
                nonlocal send_next_wait_handle
                nonlocal fwd_recv_buffer
                nonlocal fwd_wait_handles
                nonlocal recv_prev_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(forward_k, forward=True)
                # Last virtual stage no activation tensor to send.
                if _is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group):
                    output_tensor = None

                recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                    forward_k, forward=True
                )

                # If last iteration, don't receive; we already received one extra
                # before the start of the for loop.
                if k == (num_microbatches_remaining - 1):
                    recv_prev = False

                # Send activation tensor to the next stage and receive activation tensor from the
                # previous stage
                fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
                    p2p_communicator.send_forward_recv_forward(
                        output_tensor,
                        recv_prev=recv_prev,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_next_wait_handle is not None:
                    send_next_wait_handle.wait()
                if fwd_wait_handles is not None:
                    send_next_wait_handle = (
                        fwd_wait_handles.pop("send_next")
                        if "send_next" in fwd_wait_handles
                        else None
                    )
                    if "recv_prev" in fwd_wait_handles:
                        recv_prev_wait_handles.append(fwd_wait_handles.pop("recv_prev"))
                # isend() copies asynchronously; wait until the copy is done before
                # freeing the source buffer, otherwise the next PP stage gets corrupted data.
                if send_next_wait_handle is not None and config.deallocate_pipeline_outputs:
                    send_next_wait_handle.wait()
                    send_next_wait_handle = None
                # assert fwd_wait_handles is not None

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.
                if recv_prev:
                    input_tensors[next_forward_model_chunk_id].append(
                        fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
                    )
                    fwd_recv_buffer[(forward_k + 1) % fwd_recv_buffer_size] = None

                return output_tensor

            # Sync backward recv
            def pp_pre_backward(vp_stage=None):
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                if not (_is_vp_last_stage(vp_stage=vp_stage) and is_pp_last_stage(pp_group)):
                    if config.overlap_p2p_comm_warmup_flush:
                        assert recv_next_wait_handles, (
                            f'pp rank {pipeline_parallel_rank}, bwd iteration {backward_k}, '
                            'should have registered recv next handle'
                        )
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()
                    else:
                        if recv_next_wait_handles is not None and recv_next_wait_handles:
                            recv_next_wait_handle = recv_next_wait_handles.pop(0)
                            recv_next_wait_handle.wait()

            # Async backward send / receive
            def pp_post_backward(input_tensor_grad, vp_stage=None):
                nonlocal send_prev_wait_handle
                nonlocal bwd_wait_handles
                nonlocal recv_next_wait_handles
                if vp_stage is None:
                    vp_stage = get_model_chunk_id(backward_k, forward=False)
                # First virtual stage no activation gradient tensor to send.
                if _is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group):
                    input_tensor_grad = None

                recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                    backward_k, forward=False
                )

                (bwd_recv_buffer[backward_k % bwd_recv_buffer_size], bwd_wait_handles) = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )
                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))

                # Put input_tensor and output_tensor_grad in data structures in the
                # right location.

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[backward_k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(backward_k + 1) % bwd_recv_buffer_size] = None
                return input_tensor_grad

            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                pre_forward=pp_pre_forward,
                pre_backward=pp_pre_backward,
                post_forward=pp_post_forward,
                post_backward=pp_post_backward,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )

        else:  # No p2p overlap.
            backward_k = k
            output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
                f_virtual_microbatch_id=forward_k,
                b_virtual_microbatch_id=backward_k,
                checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            )
            # Send output_tensor and input_tensor_grad, receive input_tensor
            # and output_tensor_grad.

            # Determine if current stage has anything to send in either direction,
            # otherwise set tensor to None.
            forward_model_chunk_id = get_model_chunk_id(forward_k, forward=True)
            if _is_vp_last_stage(vp_stage=forward_model_chunk_id) and is_pp_last_stage(pp_group):
                output_tensor = None

            backward_model_chunk_id = get_model_chunk_id(backward_k, forward=False)
            if _is_vp_first_stage(vp_stage=backward_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(
                forward_k, forward=True
            )

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                backward_k, forward=False
            )

            # If last iteration, don't receive; we already received one extra
            # before the start of the for loop.
            if k == (num_microbatches_remaining - 1):
                recv_prev = False

            # Communicate tensors.
            (input_tensor, output_tensor_grad) = (
                p2p_communicator.send_forward_backward_recv_forward_backward(
                    output_tensor,
                    input_tensor_grad,
                    recv_prev=recv_prev,
                    recv_next=recv_next,
                    tensor_shape=tensor_shape,
                )
            )
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
            # Put input_tensor and output_tensor_grad in data structures in the
            # right location.
            if recv_prev:
                input_tensors[next_forward_model_chunk_id].append(input_tensor)
            if recv_next:
                output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)
    nvtx_range_pop(suffix="steady")

    # Run cooldown backward passes (flush out pipeline) for the last model chunk.
    nvtx_range_push(suffix="cooldown")
    curr_vp_stage = config.virtual_pipeline_model_parallel_size - 1
    if not forward_only:
        if bwd_wait_handles is not None:
            for bwd_wait_handle in bwd_wait_handles.values():
                bwd_wait_handle.wait()

        if are_all_microbatches_in_warmup:
            output_tensor_grads[num_model_chunks - 1].append(
                p2p_communicator.recv_backward(
                    tensor_shape,
                    is_last_stage=(
                        _is_vp_last_stage(vp_stage=curr_vp_stage) and is_pp_last_stage(pp_group)
                    ),
                )
            )
        for k in range(num_microbatches_remaining, total_num_microbatches):
            cur_model_chunk_id = get_model_chunk_id(k, forward=False)
            if (
                not (_is_vp_last_stage(vp_stage=cur_model_chunk_id) and is_pp_last_stage(pp_group))
                and k != 0
            ):
                if config.overlap_p2p_comm_warmup_flush:
                    assert recv_next_wait_handles, (
                        f'pp rank {pipeline_parallel_rank}, backward iteration {k}, '
                        'should have registered recv next handle'
                    )
                    recv_next_wait_handle = recv_next_wait_handles.pop(0)
                    recv_next_wait_handle.wait()
                else:
                    if recv_next_wait_handles is not None and recv_next_wait_handles:
                        recv_next_wait_handle = recv_next_wait_handles.pop(0)
                        recv_next_wait_handle.wait()

            recv_next, next_backward_model_chunk_id = recv_tensor_from_previous_stage(
                k, forward=False
            )

            if k == (total_num_microbatches - 1):
                recv_next = False

            # Prefetch recv for backward iteration k+1 for non last ranks.
            if config.overlap_p2p_comm_warmup_flush and not is_pp_last_stage(
                p2p_communicator.pp_group
            ):
                bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_recv_handles = (
                    p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad=None,  # No input_tensor_grad to send.
                        recv_next=recv_next,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                )

                if bwd_wait_recv_handles:
                    recv_next_wait_handles.append(bwd_wait_recv_handles.pop("recv_next"))

            _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

            # First virtual stage no activation gradient tensor to send.
            if _is_vp_first_stage(vp_stage=cur_model_chunk_id) and is_pp_first_stage(pp_group):
                input_tensor_grad = None

            if config.overlap_p2p_comm_warmup_flush:
                if not is_pp_last_stage(p2p_communicator.pp_group):
                    _, bwd_wait_handles = p2p_communicator.send_backward_recv_backward(
                        input_tensor_grad,
                        recv_next=False,
                        tensor_shape=tensor_shape,
                        overlap_p2p_comm=True,
                    )
                else:
                    bwd_recv_buffer[k % bwd_recv_buffer_size], bwd_wait_handles = (
                        p2p_communicator.send_backward_recv_backward(
                            input_tensor_grad,
                            recv_next=recv_next,
                            tensor_shape=tensor_shape,
                            overlap_p2p_comm=True,
                        )
                    )

                if send_prev_wait_handle is not None:
                    send_prev_wait_handle.wait()
                if bwd_wait_handles is not None:
                    send_prev_wait_handle = (
                        bwd_wait_handles.pop("send_prev")
                        if "send_prev" in bwd_wait_handles
                        else None
                    )
                    if "recv_next" in bwd_wait_handles:
                        recv_next_wait_handles.append(bwd_wait_handles.pop("recv_next"))
                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(
                        bwd_recv_buffer[k % bwd_recv_buffer_size]
                    )
                    bwd_recv_buffer[(k + 1) % bwd_recv_buffer_size] = None

            else:
                output_tensor_grad = p2p_communicator.send_backward_recv_backward(
                    input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
                )

                if recv_next:
                    output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)

        if send_prev_wait_handle is not None:
            send_prev_wait_handle.wait()

        # Launch any remaining grad reductions.
        enable_grad_sync()
        if config.grad_sync_func is not None:
            for model_chunk_id in range(num_model_chunks):
                if model_chunk_id not in synchronized_model_chunks:
                    config.grad_sync_func[model_chunk_id](model[model_chunk_id].parameters())
                    synchronized_model_chunks.add(model_chunk_id)
    nvtx_range_pop(suffix="cooldown")

    nvtx_range_push(suffix="misc")
    assert (
        not recv_prev_wait_handles
    ), 'recv_prev_wait_handles should be cleared at the end of a step'
    assert (
        not recv_next_wait_handles
    ), 'recv_next_wait_handles should be cleared at the end of a step'

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).

        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()
    # Restore config.grad_sync_func and config.param_sync_func.
    if forward_only:
        config.grad_sync_func, config.param_sync_func = grad_sync_func, param_sync_func

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()
    nvtx_range_pop(suffix="misc")

    return forward_data_store


def get_tensor_shapes(
    *,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: int,
    config,
    tp_group: Optional[torch.distributed.ProcessGroup] = None,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """Determine tensor shapes for pipeline communication.

    Returns [()] for variable_seq_lengths mode (shapes exchanged dynamically),
    or computed shapes for fixed sequence length mode.
    """
    tensor_shapes = []

    if config.variable_seq_lengths:
        # Shapes exchanged dynamically during P2P communication
        tensor_shapes.append(())
        return tensor_shapes

    # Fixed sequence lengths - compute shape
    effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
    effective_seq_length = effective_seq_length // cp_group.size()

    if config.sequence_parallel:
        effective_seq_length = effective_seq_length // tp_group.size()

    tensor_shapes.append((effective_seq_length, micro_batch_size, config.hidden_size))
    return tensor_shapes


def forward_backward_pipelining_without_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[
        Union[ProcessGroupCollection, MultiModuleProcessGroupCollection]
    ] = None,
    force_all_reduce: Optional[bool] = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""

    if isinstance(model, list):
        assert (
            len(model) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        model = model[0]
    if isinstance(data_iterator, list):
        assert (
            len(data_iterator) == 1
        ), "non-interleaved pipeline-parallel schedule does not support model chunking"
        data_iterator = data_iterator[0]

    config = get_model_config(model)
    if config.overlap_p2p_comm:
        raise ValueError(
            "Non-interleaved pipeline parallelism does not support overlapping p2p communication"
        )

    tp_group, cp_group, cp_size = None, None, None

    # Determine if this is a multi-module pipeline
    # (used for validation and backward function selection)
    is_multimodule = isinstance(pg_collection, MultiModuleProcessGroupCollection) or isinstance(
        p2p_communicator, MultiModulePipelineCommunicator
    )

    if p2p_communicator is None and pg_collection is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if os.environ.get("MEGATRON_NVSHMEM_P2P", "0") == "1":
            p2p_communicator = NvshmemP2PCommunicator(pp_group=pp_group, config=config)
        else:
            p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = cp_group.size()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        pg_collection.tp_dp_cp = parallel_state.get_tensor_and_data_parallel_group(
            with_context_parallel=True
        )

    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"

        if is_multimodule:
            # Multi-module: use language model's CP size for loss scaling
            if not config.variable_seq_lengths:
                raise ValueError(
                    "config.variable_seq_lengths=True required for multi-module pipelines"
                )
            if pg_collection.has_language_model():
                cp_size = pg_collection.get_language_model_cp_size()
            else:
                # Encoder-only ranks should not use CP loss scaling.
                cp_size = None

        elif isinstance(pg_collection, ProcessGroupCollection):
            # Single-module: extract tp/cp groups and cp_size
            assert hasattr(pg_collection, 'tp'), "pg_collection must have tp"
            assert hasattr(pg_collection, 'cp'), "pg_collection must have cp"
            tp_group = pg_collection.tp
            cp_group = pg_collection.cp
            cp_size = cp_group.size()

        else:
            raise TypeError(
                f"pg_collection must be ProcessGroupCollection or "
                f"MultiModuleProcessGroupCollection, got {type(pg_collection)}"
            )
    else:
        raise ValueError("Provide both p2p_communicator and pg_collection, or neither")

    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, p2p_communicator.is_pp_last_stage
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    if getattr(config, "moe_paged_stash", False):
        paged_stash_reset(enabled=not forward_only, config=config)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = p2p_communicator.total_stages - p2p_communicator.current_stage - 1
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    # Select backward function based on whether multi-module or single-module
    if is_multimodule:
        backward_func = partial(
            backward_step_multimodule,
            language_model_module_name=pg_collection.language_model_module_name,
        )
    else:
        backward_func = backward_step

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = []
        output_tensors = []
    forward_data_store = []

    # Run warmup forward passes.
    for i in range(num_warmup_microbatches):
        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                i % max_outstanding_backprops
                >= config.num_microbatches_with_partial_activation_checkpoints
            )
        else:
            checkpoint_activations_microbatch = None

        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )
        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(first_val_step, forward_only, i == 0),
            current_microbatch=i,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
        total_num_tokens += num_tokens

        if not forward_only:
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

    # Before running 1F1B, need to receive first forward tensor.
    # If all microbatches are run in warmup / cooldown phase, then no need to
    # receive this tensor here.
    if num_microbatches_remaining > 0:
        input_tensor = p2p_communicator.recv_forward(
            recv_tensor_shapes, p2p_communicator.is_pp_first_stage
        )

    # Run 1F1B in steady state.
    for i in range(num_microbatches_remaining):
        last_iteration = i == (num_microbatches_remaining - 1)

        # Decide to checkpoint all layers' activations of the current micro-batch
        if max_outstanding_backprops is not None:
            checkpoint_activations_microbatch = (
                (i + num_warmup_microbatches) % max_outstanding_backprops
            ) >= config.num_microbatches_with_partial_activation_checkpoints
        else:
            checkpoint_activations_microbatch = None

        output_tensor, num_tokens = forward_step(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            input_tensor,
            forward_data_store,
            config,
            cp_group_size=cp_size,
            collect_non_loss_data=collect_non_loss_data,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
            is_first_microbatch=check_first_val_step(
                first_val_step, forward_only, (i == 0) and (num_warmup_microbatches == 0)
            ),
            current_microbatch=i + num_warmup_microbatches,
            is_last_stage=p2p_communicator.is_pp_last_stage,
        )
        total_num_tokens += num_tokens

        if forward_only:
            p2p_communicator.send_forward(output_tensor, p2p_communicator.is_pp_last_stage)
            if not last_iteration:
                input_tensor = p2p_communicator.recv_forward(
                    recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )
        else:
            output_tensor_grad = p2p_communicator.send_forward_recv_backward(
                output_tensor, send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            # Add input_tensor and output_tensor to end of list.
            input_tensors.append(input_tensor)
            output_tensors.append(output_tensor)
            deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

            # Pop input_tensor and output_tensor from the start of the list for
            # the backward pass.
            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            # Enable grad sync for the last microbatch in the batch if the full
            # backward pass completes in the 1F1B stage.
            if num_warmup_microbatches == 0 and last_iteration:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            if last_iteration:
                input_tensor = None
                p2p_communicator.send_backward(
                    input_tensor_grad, p2p_communicator.is_pp_first_stage
                )
            else:
                input_tensor = p2p_communicator.send_backward_recv_forward(
                    input_tensor_grad, recv_tensor_shapes, p2p_communicator.is_pp_first_stage
                )

    # Run cooldown backward passes.
    if not forward_only:
        for i in range(num_warmup_microbatches):

            # Enable async grad reduction in the last backward pass
            # Note: If grad sync function is provided, only enable
            # async grad reduction in first pipeline stage. Other
            # pipeline stages do grad reduction during pipeline
            # bubble.
            if i == num_warmup_microbatches - 1:
                if config.grad_sync_func is None or p2p_communicator.is_pp_first_stage:
                    enable_grad_sync()

            input_tensor = input_tensors.pop(0)
            output_tensor = output_tensors.pop(0)

            output_tensor_grad = p2p_communicator.recv_backward(
                send_tensor_shapes, p2p_communicator.is_pp_last_stage
            )

            input_tensor_grad = backward_func(
                input_tensor, output_tensor, output_tensor_grad, config
            )

            p2p_communicator.send_backward(input_tensor_grad, p2p_communicator.is_pp_first_stage)

        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, p2p_communicator.is_pp_last_stage, tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            [model],
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if getattr(config, 'fine_grained_activation_offloading', False):
        off_interface.reset()

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if hasattr(config, 'cuda_graph_impl') and config.cuda_graph_impl == "local":
        create_cudagraphs()

    return forward_data_store

def forward_backward_pipelining_of_octopipe(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    octopipe_config: Dict = None,
    force_all_reduce: bool = False,
):
    """Run non-interleaved 1F1B schedule, with communication between pipeline
    stages. Returns dictionary with losses if the last stage, empty dict otherwise."""
    """Run OctoPipe schedule, execution order of computation and communication is defined in workloads. 
    Returns dictionary with losses if the last stage, empty dict otherwise."""

    """
    octopipe_config:
        octopipe_config["workloads"][i] denotes the ordered execution sequence of computation and
        communication workloads assigned to the i-th pipeline parallel rank.
        octopipe_config["sid->did"][i]: returns the device idx of stage i.
        octopipe_config["did->sid"][i]: a list of stage idxs of device i,
    """

    if isinstance(model, list):
        assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    else:
        assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module or a list of modules"
        model = [model]
    if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]
    
    config = get_model_config(model[0])
    
    if p2p_communicator is None and pg_collection is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        if os.environ.get("MEGATRON_NVSHMEM_P2P", "0") == "1":
            p2p_communicator = NvshmemP2PCommunicator(pp_group=pp_group, config=config)
        else:
            p2p_communicator = P2PCommunicator(pp_group=pp_group, config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    elif p2p_communicator is not None and pg_collection is not None:
        model_type = get_model_type(model[0])
        assert model_type != ModelType.encoder_and_decoder, (
            "encoder PP stages not yet supported when passing custom process groups. "
            "support coming soon!"
        )
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        assert hasattr(pg_collection, 'tp'), "pg_collection must have tp_group"
        assert hasattr(pg_collection, 'cp'), "pg_collection must have cp_group"
        assert hasattr(pg_collection, 'embd'), (
            "pg_collection must have a embd. In previous version, it is used default "
            "`parallel_state.default_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use "
            " `parallel_state.get_embedding_group()` "
            "If you don't need embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pos_embd'), (
            "pg_collection must have a pos_embd. In previous version, it is used default "
            "`parallel_state.default_position_embedding_ranks` to create the process group. "
            " If you are using the default process group, please use  "
            " `parallel_state.get_position_embedding_group()` "
            "If you don't need pos_embd_group, you need to explicitly set it to None."
        )
        assert hasattr(pg_collection, 'pp'), "pg_collection must have pp_group"
        assert hasattr(pg_collection, 'dp_cp'), "pg_collection must have dp_cp_group"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection "
            "provide none or provide all the process groups"
        )

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    pp_size = parallel_state.get_pipeline_model_parallel_world_size()
    workloads = octopipe_config["workloads"][pp_rank]
    device_stage_mapping = octopipe_config["did->sid"]

    stages = device_stage_mapping[pp_rank]

    stage_device_mapping = octopipe_config["sid->did"]
    stage_chunk_mapping = octopipe_config["sid->cid"]
    first_stage_sid = 0
    last_stage_sid = max(list(stage_chunk_mapping.keys()))
    local_cids = [stage_chunk_mapping[sid] for sid in stages]
    max_local_cid = max(local_cids) if local_cids else -1
    if max_local_cid >= len(model):
        raise RuntimeError(
            f"OctoPipe local stage mapping requires model chunk {max_local_cid}, "
            f"but this rank only has {len(model)} model chunk(s)."
        )
    if max_local_cid >= len(data_iterator):
        raise RuntimeError(
            f"OctoPipe local stage mapping requires data iterator chunk {max_local_cid}, "
            f"but this rank only has {len(data_iterator)} data iterator chunk(s)."
        )
    if isinstance(p2p_communicator, NvshmemP2PCommunicator):
        p2p_communicator.register_workload_routes(workloads)
    
    # Needed only when gradients are finalized in M-Core
    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    # Disable async grad reductions
    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        """Disable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        """Enable asynchronous grad reductions"""
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    # Compute number of warmup microbatches.
    num_warmup_microbatches = (
        p2p_communicator.pp_group.size() - p2p_communicator.pp_group.rank() - 1
    )
    num_warmup_microbatches = min(num_warmup_microbatches, num_microbatches)
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches

    # Checkpoint the activations of partial Transformer layers in a number of micro-batches
    # within the maximum outstanding micro-batch backpropagations.
    # Micro-batches with the ids less than 'num_microbatches_with_partial_activation_checkpoints'
    # checkpoint partial Transformer layers (or skip checkpointing) and
    # the rest of micro-batches within a window of micro-batches checkpoint
    # all Transformer layers. The window of micro-batches is set by the maximum
    # outstanding backpropagations and becomes smaller at later pipeline stages.
    # Please refer the appendix C in https://arxiv.org/pdf/2205.05198.pdf
    max_outstanding_backprops = None
    if config.num_microbatches_with_partial_activation_checkpoints is not None:
        max_outstanding_backprops = num_warmup_microbatches + 1

    rank = p2p_communicator.pp_group.rank()
    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    # Input, output tensors only need to be saved when doing backward passes
    input_tensors = None
    output_tensors = None
    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")

    if not forward_only:
        input_tensors = {}
        input_tensors_handles = {}
        output_tensors = {}
        input_tensor_grads = {}
        output_tensor_grads = {}
        output_tensor_grads_handles = {}

        input_tensors_recv_buffer = {}
        output_tensor_grads_recv_buffer = {}
        for mid in range(num_microbatches):
            input_tensors[mid] = {}
            input_tensors_handles[mid] = {}
            output_tensors[mid] = {}
            input_tensor_grads[mid] = {}
            output_tensor_grads[mid] = {}
            output_tensor_grads_handles[mid] = {}

            input_tensors_recv_buffer[mid] = {}
            output_tensor_grads_recv_buffer[mid] = {}

            for sid in stages:
                input_tensors[mid][sid] = None
                input_tensors_handles[mid][sid] = None
                output_tensors[mid][sid] = None
                input_tensor_grads[mid][sid] = None
                output_tensor_grads[mid][sid] = None
                output_tensor_grads_handles[mid][sid] = None

                input_tensors_recv_buffer[mid][sid] = None
                output_tensor_grads_recv_buffer[mid][sid] = None

    forward_data_store = []

    octopipe_bwd_splitting = (
        _prepare_octopipe_bwd_splitting(config, workloads) if not forward_only else False
    )

    for wid, workload in enumerate(workloads):
        # print(f"PP {pp_rank} bgn {wid}, {workload}", flush=True)
        op = workload['op']
        wtype = workload['type']
        mid = workload['mid']
        stop = False
        if op == 'comp':
            sid = workload['sid']
            cid = stage_chunk_mapping[sid]
            if wtype == 'f':
                if sid == first_stage_sid:
                    input_tensor = None
                else:
                    if input_tensors[mid][sid] is not None:
                        input_tensor = input_tensors[mid][sid]
                    elif input_tensors_handles[mid][sid] is not None:
                        for idx, handle in enumerate(input_tensors_handles[mid][sid]):
                            handle.wait()
                            input_tensors_handles[mid][sid][idx] = None
                        input_tensors[mid][sid] = input_tensors_recv_buffer[mid][sid]
                        input_tensor = input_tensors[mid][sid]
                    elif input_tensors_recv_buffer[mid][sid] is not None:
                        input_tensors[mid][sid] = input_tensors_recv_buffer[mid][sid]
                        input_tensor = input_tensors[mid][sid]
                    else:
                        raise ("Wrong Data Flow.")
                output_tensor, num_tokens = forward_step(
                    forward_step_func,
                    data_iterator[cid],
                    model[cid],
                    num_microbatches,
                    input_tensor,
                    forward_data_store,
                    config,
                    cp_group_size=pg_collection.cp.size(),
                    collect_non_loss_data=collect_non_loss_data,
                    checkpoint_activations_microbatch=None, # NOTE: not supported logic
                    is_first_microbatch=check_first_val_step(first_val_step, forward_only, mid == 0),
                    current_microbatch=mid,
                    is_last_stage= sid == last_stage_sid,
                )
                total_num_tokens += num_tokens

                if not forward_only:
                    output_tensors[mid][sid] = output_tensor
                    # if output_tensor is not None: # 会报错，而且可能会导致recv无法完成
                    #     deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
            elif wtype == 'b':
                if sid == last_stage_sid:
                    output_tensor_grad = None
                else:
                    if output_tensor_grads[mid][sid] is not None:
                        output_tensor_grad = output_tensor_grads[mid][sid]
                    elif output_tensor_grads_handles[mid][sid] is not None:
                        for idx, handle in enumerate(output_tensor_grads_handles[mid][sid]):
                            handle.wait()
                            output_tensor_grads_handles[mid][sid][idx] = None
                        output_tensor_grads[mid][sid] = output_tensor_grads_recv_buffer[mid][sid]
                        output_tensor_grad = output_tensor_grads[mid][sid]
                    elif output_tensor_grads_recv_buffer[mid][sid] is not None:
                        output_tensor_grads[mid][sid] = output_tensor_grads_recv_buffer[mid][sid]
                        output_tensor_grad = output_tensor_grads[mid][sid]
                    else:
                        raise ("Wrong Data Flow.")

                if sid == first_stage_sid:
                    input_tensor = None
                else:
                    input_tensor = input_tensors[mid][sid]
                output_tensor = output_tensors[mid][sid]

                input_tensor_grad = backward_step(
                    input_tensor, output_tensor, output_tensor_grad, config
                )
                if octopipe_bwd_splitting:
                    _register_octopipe_wgrad_task(model[cid], chunk=cid, tag=(sid, mid))

                input_tensor_grads[mid][sid] = input_tensor_grad
            elif wtype == 'w':
                # NOTE: W only FIFO execution order
                if not octopipe_bwd_splitting:
                    raise ValueError(
                        "OctoPipe schedule contains a 'w' workload, but "
                        "--octopipe-bwd-splitting is disabled."
                    )
                WeightGradStore.pop(chunk=cid, tag=(sid, mid))
            else:
                raise ValueError(f"{op} Workload Type Error: {wtype}")

        elif op == 'send':
            src_sid = workload['sender_sid']
            dst_sid = workload['recver_sid']
            dst_rank = stage_device_mapping[dst_sid]
            nvshmem_route_kwargs = {}
            if isinstance(p2p_communicator, NvshmemP2PCommunicator):
                nvshmem_route_kwargs = {
                    "sender_sid": src_sid,
                    "recver_sid": dst_sid,
                    "mid": mid,
                }
            if wtype == 'f':
                output_tensor = output_tensors[mid][src_sid]
                p2p_communicator.send_tensor_async(
                    output_tensor,
                    p2p_communicator._get_global_rank(dst_rank),
                    **nvshmem_route_kwargs,
                )
            elif wtype == 'b':
                input_tensor_grad = input_tensor_grads[mid][src_sid]
                p2p_communicator.send_tensor_async(
                    input_tensor_grad,
                    p2p_communicator._get_global_rank(dst_rank),
                    **nvshmem_route_kwargs,
                )
            else:
                raise ValueError(f"{op} Workload Type Error: {wtype}")
        elif op == 'recv':
            src_sid = workload['sender_sid']
            dst_sid = workload['recver_sid']
            src_rank = stage_device_mapping[src_sid]
            nvshmem_route_kwargs = {}
            if isinstance(p2p_communicator, NvshmemP2PCommunicator):
                nvshmem_route_kwargs = {
                    "sender_sid": src_sid,
                    "recver_sid": dst_sid,
                    "mid": mid,
                }
            if wtype == 'f':
                input_tensors_recv_buffer[mid][dst_sid], input_tensors_handles[mid][dst_sid] = p2p_communicator.recv_tensor_async(
                    recv_tensor_shapes,
                    recv_src_rank=p2p_communicator._get_global_rank(src_rank),
                    **nvshmem_route_kwargs,
                )
            elif wtype == 'b':
                output_tensor_grads_recv_buffer[mid][dst_sid], output_tensor_grads_handles[mid][dst_sid] = p2p_communicator.recv_tensor_async(
                    send_tensor_shapes,
                    recv_src_rank=p2p_communicator._get_global_rank(src_rank),
                    **nvshmem_route_kwargs,
                )
            else:
                raise ValueError(f"{op} Workload Type Error: {wtype}")
        else:
            raise ValueError(f"Op Type Error: {op}")
        # print(f"PP {pp_rank} end {wid}, {workload}", flush=True)

    if octopipe_bwd_splitting:
        pending_chunks = [
            chunk for chunk in range(len(model)) if WeightGradStore.pending_count(chunk=chunk) > 0
        ]
        if pending_chunks:
            raise RuntimeError(
                "OctoPipe backward splitting left pending WeightGradStore tasks "
                f"for chunks {pending_chunks}; check b/w workload balance."
            )

    if not forward_only:
        # Launch any remaining grad reductions.
        if no_sync_context is not None:
            enable_grad_sync()
            if config.grad_sync_func is not None:
                config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:

        # If defer_embedding_wgrad_compute is enabled we need to do the
        # weight gradient GEMM's here.
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )

        # Finalize model grads (perform full grad all-reduce / reduce-scatter for
        # data parallelism, layernorm all-reduce for sequence parallelism, and
        # embedding all-reduce for pipeline parallelism).
        config.finalize_model_grads_func(
            # [model], # single chunk
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if octopipe_bwd_splitting:
        WeightGradStore.reset()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()

    return forward_data_store


def forward_backward_pipelining_of_octopipe_nvshmem(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
    octopipe_config: Dict = None,
    force_all_reduce: bool = False,
):
    """Comp-driven OctoPipe schedule for NVSHMEM P2P.

    Unlike ``forward_backward_pipelining_of_octopipe``, this path ignores
    explicit send/recv workload entries.  For each compute workload it pulls
    the required tensor just before compute and sends the produced tensor right
    after compute.
    """

    if isinstance(model, list):
        assert all(isinstance(chunk, torch.nn.Module) for chunk in model), "invalid model chunking"
    else:
        assert isinstance(model, torch.nn.Module), "model must be a torch.nn.Module or a list of modules"
        model = [model]
    if not isinstance(data_iterator, list):
        data_iterator = [data_iterator]

    config = get_model_config(model[0])
    if (
        os.environ.get("MEGATRON_NVSHMEM_P2P_BUFFER_BYTES") is None
        and os.environ.get("MEGATRON_NVSHMEM_P2P_DEFAULT_BUFFER_BYTES") is None
    ):
        dtype_size = torch.empty((), dtype=config.pipeline_dtype).element_size()
        buffer_factor = int(os.environ.get("MEGATRON_NVSHMEM_P2P_BUFFER_FACTOR", "1"))
        effective_seq_length = decoder_seq_length if decoder_seq_length is not None else seq_length
        slot_bytes = buffer_factor * int(micro_batch_size) * int(effective_seq_length) * int(config.hidden_size) * dtype_size
        os.environ["MEGATRON_NVSHMEM_P2P_DEFAULT_BUFFER_BYTES"] = str(slot_bytes)

    if p2p_communicator is None and pg_collection is None:
        pp_group = parallel_state.get_pipeline_model_parallel_group()
        p2p_communicator = OctoPipeP2PCommunicator(pp_group=pp_group, config=config)
        tp_group = parallel_state.get_tensor_model_parallel_group()
        cp_group = parallel_state.get_context_parallel_group()
        embd_group = parallel_state.get_embedding_group(check_initialized=False)
        pos_emb_group = parallel_state.get_position_embedding_group(check_initialized=False)

        pg_collection = ProcessGroupCollection()
        pg_collection.tp = tp_group
        pg_collection.pp = pp_group
        pg_collection.embd = embd_group
        pg_collection.pos_embd = pos_emb_group
        pg_collection.cp = cp_group
        pg_collection.dp_cp = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
    elif p2p_communicator is not None and pg_collection is not None:
        assert hasattr(p2p_communicator, 'config'), "p2p_communicator must have a config"
        tp_group = pg_collection.tp
        cp_group = pg_collection.cp
        if not isinstance(p2p_communicator, OctoPipeP2PCommunicator):
            pp_group = pg_collection.pp
            p2p_communicator = OctoPipeP2PCommunicator(pp_group=pp_group, config=config)
    else:
        raise ValueError(
            "Invalid combination of p2p_communicator, pg_collection provide none or provide all"
        )

    pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    workloads = octopipe_config["workloads"][pp_rank]
    device_stage_mapping = octopipe_config["did->sid"]
    stages = device_stage_mapping[pp_rank]
    stage_device_mapping = octopipe_config["sid->did"]
    stage_chunk_mapping = octopipe_config["sid->cid"]
    first_stage_sid = 0
    last_stage_sid = max(list(stage_chunk_mapping.keys()))
    local_cids = [stage_chunk_mapping[sid] for sid in stages]
    max_local_cid = max(local_cids) if local_cids else -1
    if max_local_cid >= len(model):
        raise RuntimeError(
            f"OctoPipe local stage mapping requires model chunk {max_local_cid}, "
            f"but this rank only has {len(model)} model chunk(s)."
        )
    if max_local_cid >= len(data_iterator):
        raise RuntimeError(
            f"OctoPipe local stage mapping requires data iterator chunk {max_local_cid}, "
            f"but this rank only has {len(data_iterator)} data iterator chunk(s)."
        )
    p2p_communicator.register_workload_routes(workloads)

    runtime_cache = octopipe_config.setdefault("_octopipe_nvshmem_runtime_cache", {})
    cache_key = (pp_rank, id(workloads))
    cached = runtime_cache.get(cache_key)
    if cached is None:
        # Build direct producer/consumer maps from explicit OctoPipe communication
        # workloads, but do not execute send/recv workloads in the main loop.
        fwd_src = {}
        fwd_dst = {}
        bwd_src = {}
        bwd_dst = {}
        comp_workloads = []
        stage_global_ranks = {}
        for sid, did in stage_device_mapping.items():
            stage_global_ranks[sid] = p2p_communicator._get_global_rank(did)
        for workload in workloads:
            op = workload.get('op')
            if op == 'comp':
                comp_workloads.append(workload)
                continue
            if op not in ('send', 'recv'):
                continue
            sender_sid = workload['sender_sid']
            recver_sid = workload['recver_sid']
            mid = workload['mid']
            key = (mid, sender_sid, workload['type'])
            recv_key = (mid, recver_sid, workload['type'])
            if workload['type'] == 'f':
                fwd_dst[key] = recver_sid
                fwd_src[recv_key] = sender_sid
            elif workload['type'] == 'b':
                bwd_dst[key] = recver_sid
                bwd_src[recv_key] = sender_sid
        cached = {
            "comp_workloads": comp_workloads,
            "fwd_src": fwd_src,
            "fwd_dst": fwd_dst,
            "bwd_src": bwd_src,
            "bwd_dst": bwd_dst,
            "stage_global_ranks": stage_global_ranks,
        }
        runtime_cache[cache_key] = cached
    else:
        fwd_src = cached["fwd_src"]
        fwd_dst = cached["fwd_dst"]
        bwd_src = cached["bwd_src"]
        bwd_dst = cached["bwd_dst"]
    comp_workloads = cached["comp_workloads"]
    stage_global_ranks = cached["stage_global_ranks"]
    stage_time_profiler = None
    if os.environ.get("ENABLE_OCTOPIPE_PROFILER", "0") == "1":
        profiler_key = (pp_rank, "stage_time_profiler")
        stage_time_profiler = runtime_cache.get(profiler_key)
        if stage_time_profiler is None:
            stage_time_profiler = OctoPipeStageTimeProfiler(pp_rank)
            runtime_cache[profiler_key] = stage_time_profiler
        stage_time_profiler.start_step()

    if config.finalize_model_grads_func is not None and not forward_only:
        embedding_module = clear_embedding_activation_buffer(
            config, model, is_pp_last_stage(p2p_communicator.pp_group)
        )

    if config.timers is not None:
        config.timers('forward-backward', log_level=1).start(barrier=config.barrier_with_L1_time)

    no_sync_func = config.no_sync_func
    if no_sync_func is None:
        no_sync_func = contextlib.nullcontext
    no_sync_context = None

    def disable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is None:
            no_sync_context = no_sync_func()
            no_sync_context.__enter__()

    def enable_grad_sync():
        nonlocal no_sync_context
        if no_sync_context is not None:
            no_sync_context.__exit__(None, None, None)
            no_sync_context = None

    disable_grad_sync()

    recv_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    send_tensor_shapes = get_tensor_shapes(
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        decoder_seq_length=decoder_seq_length,
        config=config,
        tp_group=tp_group,
        cp_group=cp_group,
    )
    if adjust_tensor_shapes_fn is not None:
        recv_tensor_shapes, send_tensor_shapes = adjust_tensor_shapes_fn(
            recv_tensor_shapes, send_tensor_shapes
        )

    input_tensors = {}
    output_tensors = {}
    input_tensor_grads = {}
    output_tensor_grads = {}
    for mid in range(num_microbatches):
        input_tensors[mid] = {sid: None for sid in stages}
        output_tensors[mid] = {sid: None for sid in stages}
        input_tensor_grads[mid] = {sid: None for sid in stages}
        output_tensor_grads[mid] = {sid: None for sid in stages}

    total_num_tokens = torch.zeros([], dtype=torch.int, device="cuda")
    forward_data_store = []
    octopipe_bwd_splitting = (
        _prepare_octopipe_bwd_splitting(config, workloads) if not forward_only else False
    )

    def _recv_for_comp(mid, src_sid, dst_sid, tensor_shapes, requires_grad):
        return p2p_communicator.recv_tensor_blocking(
            tensor_shapes,
            recv_src_rank=stage_global_ranks[src_sid],
            sender_sid=src_sid,
            recver_sid=dst_sid,
            mid=mid,
            requires_grad=requires_grad,
        )

    def _send_after_comp(mid, src_sid, dst_sid, tensor):
        if tensor is None:
            return
        p2p_communicator.send_tensor_async(
            tensor,
            stage_global_ranks[dst_sid],
            sender_sid=src_sid,
            recver_sid=dst_sid,
            mid=mid,
        )
        deallocate_output_tensor(tensor, config.deallocate_pipeline_outputs)

    for workload in comp_workloads:
        wtype = workload['type']
        mid = workload['mid']
        sid = workload['sid']
        cid = stage_chunk_mapping[sid]

        if wtype == 'f':
            if sid == first_stage_sid:
                input_tensor = None
            else:
                input_tensor = input_tensors[mid][sid]
                if input_tensor is None:
                    src_sid = fwd_src.get((mid, sid, 'f'))
                    if src_sid is None:
                        raise RuntimeError(f"Missing forward source for mid={mid}, sid={sid}")
                    input_tensor = _recv_for_comp(
                        mid, src_sid, sid, recv_tensor_shapes, requires_grad=True
                    )
                    input_tensors[mid][sid] = input_tensor

            output_tensor, num_tokens = _octopipe_f_comp(
                forward_step_func=forward_step_func,
                data_iterator=data_iterator[cid],
                model=model[cid],
                num_microbatches=num_microbatches,
                input_tensor=input_tensor,
                forward_data_store=forward_data_store,
                config=config,
                cp_group_size=pg_collection.cp.size(),
                collect_non_loss_data=collect_non_loss_data,
                checkpoint_activations_microbatch=None,
                is_first_microbatch=check_first_val_step(first_val_step, forward_only, mid == 0),
                current_microbatch=mid,
                is_last_stage=sid == last_stage_sid,
                profiler=stage_time_profiler,
                sid=sid,
                wtype=wtype,
                mid=mid,
            )
            total_num_tokens += num_tokens
            if not forward_only:
                output_tensors[mid][sid] = output_tensor

            dst_sid = fwd_dst.get((mid, sid, 'f'))
            if dst_sid is not None:
                _send_after_comp(mid, sid, dst_sid, output_tensor)

        elif wtype == 'b':
            if sid == last_stage_sid:
                output_tensor_grad = None
            else:
                output_tensor_grad = output_tensor_grads[mid][sid]
                if output_tensor_grad is None:
                    src_sid = bwd_src.get((mid, sid, 'b'))
                    if src_sid is None:
                        raise RuntimeError(f"Missing backward source for mid={mid}, sid={sid}")
                    output_tensor_grad = _recv_for_comp(
                        mid, src_sid, sid, send_tensor_shapes, requires_grad=False
                    )
                    output_tensor_grads[mid][sid] = output_tensor_grad

            input_tensor = None if sid == first_stage_sid else input_tensors[mid][sid]
            output_tensor = output_tensors[mid][sid]
            input_tensor_grad = _octopipe_b_comp(
                input_tensor=input_tensor,
                output_tensor=output_tensor,
                output_tensor_grad=output_tensor_grad,
                config=config,
                octopipe_bwd_splitting=octopipe_bwd_splitting,
                model_chunk=model[cid],
                chunk=cid,
                profiler=stage_time_profiler,
                sid=sid,
                wtype=wtype,
                mid=mid,
            )
            input_tensor_grads[mid][sid] = input_tensor_grad

            dst_sid = bwd_dst.get((mid, sid, 'b'))
            if dst_sid is not None:
                _send_after_comp(mid, sid, dst_sid, input_tensor_grad)

        elif wtype == 'w':
            if not octopipe_bwd_splitting:
                raise ValueError(
                    "OctoPipe schedule contains a 'w' workload, but --octopipe-bwd-splitting is disabled."
                )
            _octopipe_w_comp(
                chunk=cid,
                seq_split_idx=0,
                strict=True,
                profiler=stage_time_profiler,
                sid=sid,
                wtype=wtype,
                mid=mid,
            )
        else:
            raise ValueError(f"comp Workload Type Error: {wtype}")

    if octopipe_bwd_splitting:
        pending_chunks = [
            chunk for chunk in range(len(model)) if WeightGradStore.pending_count(chunk=chunk) > 0
        ]
        if pending_chunks:
            raise RuntimeError(
                "OctoPipe backward splitting left pending WeightGradStore tasks "
                f"for chunks {pending_chunks}; check b/w workload balance."
            )

    if not forward_only and no_sync_context is not None:
        enable_grad_sync()
        if config.grad_sync_func is not None:
            config.grad_sync_func(model.parameters())

    if config.finalize_model_grads_func is not None and not forward_only:
        finish_embedding_wgrad_compute(
            config, embedding_module, is_pp_last_stage(p2p_communicator.pp_group), tp_group
        )
        config.finalize_model_grads_func(
            model,
            total_num_tokens if config.calculate_per_token_loss else None,
            pg_collection=pg_collection,
            force_all_reduce=force_all_reduce,
        )

    if config.timers is not None:
        config.timers('forward-backward').stop()

    if stage_time_profiler is not None:
        stage_time_profiler.finish_step()

    if octopipe_bwd_splitting:
        WeightGradStore.reset()

    if (
        hasattr(config, 'cuda_graph_impl')
        and config.cuda_graph_impl == "local"
        and config.cuda_graph_scope != "full_iteration"
    ):
        create_cudagraphs()

    return forward_data_store
