# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import logging
import os
import sys
import time
from importlib import import_module
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.pipeline_parallel.utils import is_pp_first_stage, is_pp_last_stage
from megatron.core.utils import nvtx_decorator

logger = logging.getLogger(__name__)

# Types
Shape = Union[List[int], torch.Size]


def _batched_p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    ops = []
    if tensor_send_prev is not None:
        send_prev_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_prev, prev_pipeline_rank, group
        )
        ops.append(send_prev_op)
    if tensor_recv_prev is not None:
        recv_prev_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_prev, prev_pipeline_rank, group
        )
        ops.append(recv_prev_op)
    if tensor_send_next is not None:
        send_next_op = torch.distributed.P2POp(
            torch.distributed.isend, tensor_send_next, next_pipeline_rank, group
        )
        ops.append(send_next_op)
    if tensor_recv_next is not None:
        recv_next_op = torch.distributed.P2POp(
            torch.distributed.irecv, tensor_recv_next, next_pipeline_rank, group
        )
        ops.append(recv_next_op)
    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
    else:
        reqs = []
    return reqs


def _p2p_ops(
    *,
    tensor_send_prev: Optional[torch.Tensor],
    tensor_recv_prev: Optional[torch.Tensor],
    tensor_send_next: Optional[torch.Tensor],
    tensor_recv_next: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    prev_pipeline_rank: int,
    next_pipeline_rank: int,
):
    reqs = {}
    even_send_odd_recv_group = group
    if group.size() == 2 and torch.distributed.get_backend(group) != 'ucc':
        # Use the global process group for one of the two p2p communications
        # to allow the overlap of the independent communications.
        # Using the global process group is compatible because the pipeline-parallel
        # communications set the source and destination by global rank.
        # The only exception occurs when using the ‘ucc’ backend.
        # Because the global communicator always uses the ‘nccl’ backend,
        # we must ensure the else path is followed for the ‘ucc’ backend.
        even_recv_odd_send_group = torch.distributed.group.WORLD
    else:
        even_recv_odd_send_group = group

    if group.rank() % 2 == 0:
        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["send_prev"] = send_prev_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["recv_next"] = recv_next_req

    else:
        if tensor_recv_prev is not None:
            recv_prev_req = torch.distributed.irecv(
                tensor=tensor_recv_prev, src=prev_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_prev"] = recv_prev_req

        if tensor_send_next is not None:
            send_next_req = torch.distributed.isend(
                tensor=tensor_send_next, dst=next_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_next"] = send_next_req

        if tensor_recv_next is not None:
            recv_next_req = torch.distributed.irecv(
                tensor=tensor_recv_next, src=next_pipeline_rank, group=even_send_odd_recv_group
            )
            reqs["recv_next"] = recv_next_req

        if tensor_send_prev is not None:
            send_prev_req = torch.distributed.isend(
                tensor=tensor_send_prev, dst=prev_pipeline_rank, group=even_recv_odd_send_group
            )
            reqs["send_prev"] = send_prev_req
    return reqs

def _p2p_ops_octopipe(
    *,
    tensor_send: Optional[torch.Tensor],
    tensor_recv: Optional[torch.Tensor],
    group: torch.distributed.ProcessGroup,
    recv_src_rank: int,
    send_dst_rank: int,
):
    reqs = {}
    if tensor_recv is not None:
        recv_prev_req = torch.distributed.irecv(
            tensor=tensor_recv, src=recv_src_rank, group=group
        )
        reqs["recv"] = recv_prev_req

    if tensor_send is not None:
        send_prev_req = torch.distributed.isend(
            tensor=tensor_send, dst=send_dst_rank, group=group
        )
        reqs["send"] = send_prev_req
    
    return reqs

def is_single_shape(x) -> bool:
    """Check if the input is a single shape."""
    if isinstance(x, torch.Size):
        return True
    if isinstance(x, (list, tuple)) and len(x) > 0 and all(isinstance(d, int) for d in x):
        return True
    return False


class P2PCommunicator:
    """P2P (Point-to-Point) Communicator for pipeline parallelism.

    This class handles communication between pipeline stages by managing
    tensor exchanges between consecutive stages in the pipeline.
    """

    def __init__(self, pp_group: dist.ProcessGroup, config: ModelParallelConfig):
        # Basic attrs
        self.pp_group = pp_group
        self.config = config

        world_size = self.pp_group.size()
        curr_rank_in_pg = self.pp_group.rank()
        self.curr_rank_in_pg = curr_rank_in_pg
        next_rank_pg = (curr_rank_in_pg + 1) % world_size
        prev_rank_pg = (curr_rank_in_pg - 1) % world_size

        self.next_rank: int | None = dist.get_global_rank(self.pp_group, next_rank_pg)
        self.prev_rank: int | None = dist.get_global_rank(self.pp_group, prev_rank_pg)
        self.virtual_pipeline_model_parallel_size = (
            config.virtual_pipeline_model_parallel_size
            if config.virtual_pipeline_model_parallel_size is not None
            else None
        )

    @property
    def is_pp_first_stage(self) -> bool:
        """Return True if pp first stage."""
        return is_pp_first_stage(self.pp_group)

    @property
    def is_pp_last_stage(self) -> bool:
        """Return True if pp last stage."""
        return is_pp_last_stage(self.pp_group)

    @property
    def total_stages(self) -> int:
        """Return total number of pipeline stages."""
        return self.pp_group.size()

    @property
    def current_stage(self) -> int:
        """Return current pipeline stage index (0-indexed)."""
        return self.pp_group.rank()

    def _get_global_rank(self, pp_rank: int) -> int:
        return dist.get_global_rank(self.pp_group, pp_rank)

    def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
        """Communicate tensor shapes between stages. Used to communicate
        tensor shapes before the actual tensor communication happens.
        This is required when the sequence lengths across micro batches
        are not uniform.

        Args:
            tensor_send_next: tensor to send to next rank (no tensor sent if
                            set to None).
            tensor_send_prev: tensor to send to prev rank (no tensor sent if
                            set to None).
            recv_prev: boolean for whether tensor should be received from
                    previous rank.
            recv_next: boolean for whether tensor should be received from
                    next rank.
        Returns:
            (recv_prev_shape, recv_next_shape)
        """
        config = self.config
        recv_prev_shape_tensor = None
        recv_next_shape_tensor = None
        send_prev_shape_tensor = None
        send_next_shape_tensor = None
        if recv_prev:
            recv_prev_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if recv_next:
            recv_next_shape_tensor = torch.empty(
                (3,), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_prev is not None:
            send_prev_shape_tensor = torch.tensor(
                tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )
        if tensor_send_next is not None:
            send_next_shape_tensor = torch.tensor(
                tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
            )

        if config.use_ring_exchange_p2p:
            torch.distributed.ring_exchange(
                tensor_send_prev=send_prev_shape_tensor,
                tensor_recv_prev=recv_prev_shape_tensor,
                tensor_send_next=send_next_shape_tensor,
                tensor_recv_next=recv_next_shape_tensor,
                group=self.pp_group,
            )
        else:
            ops = []
            if send_prev_shape_tensor is not None:
                send_prev_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(send_prev_op)
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_prev_shape_tensor, self.prev_rank, self.pp_group
                )
                ops.append(recv_prev_op)
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(send_next_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_next_shape_tensor, self.next_rank, self.pp_group
                )
                ops.append(recv_next_op)
            if len(ops) > 0:
                reqs = torch.distributed.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()

            # To protect against race condition when using batch_isend_irecv().
            # should take this out once the bug with batch_isend_irecv is resolved.
            torch.cuda.synchronize()

        recv_prev_shape = [0, 0, 0]
        if recv_prev_shape_tensor is not None:
            recv_prev_shape = recv_prev_shape_tensor.tolist()

        recv_next_shape = [0, 0, 0]
        if recv_next_shape_tensor is not None:
            recv_next_shape = recv_next_shape_tensor.tolist()

        return recv_prev_shape, recv_next_shape

    def _communicate(
        self,
        *,
        tensor_send_next: Optional[torch.Tensor],
        tensor_send_prev: Optional[torch.Tensor],
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
        wait_on_reqs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Communicate tensors between stages. Used as helper method in other
        communication methods that are used in megatron/schedules.py.

        Args:
            tensor_send_next (torch.Tensor, optional):
                Tensor to send to next rank (no tensor sent if None)

            tensor_send_prev (torch.Tensor, optional):
                Tensor to send to prev rank (no tensor sent if None)

            recv_prev (boolean, required):
                whether tensor should be received from previous rank.

            recv_next (boolean, required):
                whether tensor should be received from next rank.

            tensor_shape (List[int] or torch.Size, required):
                shape of tensor to receive (this method assumes that all
                tensors sent and received in a single function call are
                the same shape).

            wait_on_reqs (boolean, optional, default=False):
                For non-batched p2p communication, wait on each request
                before returning.

        Returns:
            tuple containing

            - tensor_recv_prev: torch.Tensor if recv_prev is True, None otherwise.
            - tensor_recv_next: torch.Tensor if recv_next is True, None otherwise.

        """

        config = self.config
        tensor_recv_prev_func = None
        tensor_recv_next_func = None

        if config.variable_seq_lengths or config.mtp_standalone:
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )
        else:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape

        def create_tensor_recv_prev():
            return torch.empty(
                recv_prev_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        def create_tensor_recv_next():
            return torch.empty(
                recv_next_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        if recv_prev:
            if config.pipeline_dtype is None:
                raise RuntimeError("pipeline_dtype must be provided if recv_prev is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_prev is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_prev_func = create_tensor_recv_prev

        if recv_next:
            if config.pipeline_dtype is None:
                raise RuntimeError("dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv_next_func = create_tensor_recv_next

        # Send tensors in both the forward and backward directions as appropriate.
        if config.use_ring_exchange_p2p:

            def _ring_exchange_wrapper(**kwargs):
                torch.distributed.ring_exchange(**kwargs)
                return []

            p2p_func = _ring_exchange_wrapper
        elif config.batch_p2p_comm:
            assert wait_on_reqs
            p2p_func = _batched_p2p_ops
        else:
            p2p_func = _p2p_ops

        pp_group = self.pp_group
        next_rank = self.next_rank
        prev_rank = self.prev_rank

        if config.use_ring_exchange_p2p or config.batch_p2p_comm:
            reqs = []
        else:
            reqs = {}

        tensor_recv_prev = None
        tensor_recv_next = None
        if tensor_recv_prev_func is not None:
            tensor_recv_prev = tensor_recv_prev_func()

        if tensor_recv_next_func is not None:
            tensor_recv_next = tensor_recv_next_func()

        p2p_reqs = p2p_func(
            tensor_send_prev=tensor_send_prev,
            tensor_recv_prev=tensor_recv_prev,
            tensor_send_next=tensor_send_next,
            tensor_recv_next=tensor_recv_next,
            group=pp_group,
            prev_pipeline_rank=prev_rank,
            next_pipeline_rank=next_rank,
        )
        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        if wait_on_reqs and len(reqs) > 0:
            for req in reqs if isinstance(reqs, list) else reqs.values():
                req.wait()
            reqs = None

        if config.batch_p2p_comm and config.batch_p2p_sync:
            # To protect against race condition when using batch_isend_irecv().
            # User should assert that we have a modern enough PyTorch to not need this
            torch.cuda.synchronize()

        return tensor_recv_prev, tensor_recv_next, reqs

    def _communicate_async(
        self,
        *,
        tensor_send: Optional[torch.Tensor],
        send_dst_rank: int,
        need_recv: bool,
        recv_src_rank: int,
        tensor_shape: Shape,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        config = self.config

        def create_tensor_recv():
            return torch.empty(
                tensor_shape,
                requires_grad=True,
                device=torch.cuda.current_device(),
                dtype=config.pipeline_dtype,
            )

        # Send tensors in both the forward and backward directions as appropriate.
        pp_group = self.pp_group
        reqs = {}

        tensor_recv = None
        if need_recv:
            if config.pipeline_dtype is None:
                raise RuntimeError("dtype must be provided if recv_next is True")
            if tensor_shape is None:
                raise RuntimeError(
                    "tensor_shape must be specified if recv_next is True. "
                    "Common tensor_shape is (seq_length, micro_batch_size, hidden_size)"
                )
            tensor_recv = create_tensor_recv()

        p2p_reqs = _p2p_ops_octopipe(
            tensor_send=tensor_send,
            tensor_recv=tensor_recv,
            group=pp_group,
            send_dst_rank=send_dst_rank,
            recv_src_rank=recv_src_rank,
        )

        if isinstance(p2p_reqs, list):
            reqs.extend(p2p_reqs)
        else:
            reqs.update(p2p_reqs)

        return tensor_recv, reqs
    
    @nvtx_decorator()
    def recv_forward(
        self, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        config = self.config
        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_tensor_shapes:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def recv_backward(
        self, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from next rank in pipeline (backward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        config = self.config
        output_tensor_grads = []
        for tensor_shape in tensor_shapes:
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_tensor_shapes:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_forward(self, output_tensors, is_last_stage: bool) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        config = self.config
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        for output_tensor in output_tensors:
            if not is_last_stage:
                if config.timers is not None:
                    config.timers('forward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('forward-send').stop()
    
    @nvtx_decorator()
    def send_tensor_async(self, send_tensors, dst_rank, stop=False) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        config = self.config
        if not isinstance(send_tensors, list):
            send_tensors = [send_tensors]

        reqs = []
        for send_tensor in send_tensors:
            if config.timers is not None:
                config.timers('forward-send', log_level=2).start()
            
            if stop:
                import pdb
                pdb.set_trace()

            _, req = self._communicate_async(
                tensor_send=send_tensor,
                send_dst_rank=dst_rank,
                need_recv=False,
                recv_src_rank=None,
                tensor_shape=None,
            )

            if config.timers is not None:
                config.timers('forward-send').stop()
        return reqs

    @nvtx_decorator()
    def recv_tensor_async(
        self, tensor_shapes, recv_src_rank
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from next rank in pipeline (backward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        config = self.config
        output_tensor_grads = []
        reqs = []
        for tensor_shape in tensor_shapes:
            if config.timers is not None:
                config.timers('backward-recv', log_level=2).start()
            output_tensor_grad, req = self._communicate_async(
                tensor_send=None,
                send_dst_rank=None,
                need_recv=True,
                recv_src_rank=recv_src_rank,
                tensor_shape=tensor_shape,
            )
            if config.timers is not None:
                config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
            reqs.append(req['recv'])
        if unwrap_tensor_shapes:
            return output_tensor_grads[0], reqs
        return output_tensor_grads, reqs

    @nvtx_decorator()
    def send_forward_async(self, output_tensors, is_last_stage: bool, stop=False) -> None:
        """Send tensor to next rank in pipeline (forward send)."""
        config = self.config
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]

        reqs = []
        for output_tensor in output_tensors:
            if not is_last_stage:
                if config.timers is not None:
                    config.timers('forward-send', log_level=2).start()
                
                if stop:
                    import pdb
                    pdb.set_trace()

                _, _, req = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                    wait_on_reqs=False,
                )
                reqs.append(req["send_next"])
                if config.timers is not None:
                    config.timers('forward-send').stop()
        return reqs

    @nvtx_decorator()
    def send_backward_async(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        config = self.config

        reqs = []
        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                if config.timers is not None:
                    config.timers('backward-send', log_level=2).start()
                _, _, req = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                    wait_on_reqs=False,
                )
                reqs.append(req["send_prev"])
                if config.timers is not None:
                    config.timers('backward-send').stop()
        return reqs

    @nvtx_decorator()
    def recv_forward_async(
        self, tensor_shapes, is_first_stage: bool, stop=False
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from previous rank in pipeline (forward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        reqs = []
        config = self.config
        for tensor_shape in tensor_shapes:
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('forward-recv', log_level=2).start()
                
                if stop:
                    import pdb
                    pdb.set_trace()

                input_tensor, _, req = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                    wait_on_reqs=False,
                )
                
                if config.timers is not None:
                    config.timers('forward-recv').stop()
            input_tensors.append(input_tensor)
            reqs.append(req['recv_prev'])
            
        if unwrap_tensor_shapes:
            return input_tensors[0], reqs
        return input_tensors, reqs

    @nvtx_decorator()
    def recv_backward_async(
        self, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Receive tensor from next rank in pipeline (backward receive)."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]
        config = self.config
        output_tensor_grads = []
        reqs = []
        for tensor_shape in tensor_shapes:
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('backward-recv', log_level=2).start()
                _, output_tensor_grad, req = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                    wait_on_reqs=False,
                )
                if config.timers is not None:
                    config.timers('backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
            reqs.append(req['recv_next'])
        if unwrap_tensor_shapes:
            return output_tensor_grads[0], reqs
        return output_tensor_grads, reqs
    
    @nvtx_decorator()
    def send_backward(self, input_tensor_grads, is_first_stage: bool) -> None:
        """Send tensor to previous rank in pipeline (backward send)."""
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        config = self.config
        for input_tensor_grad in input_tensor_grads:
            if not is_first_stage:
                if config.timers is not None:
                    config.timers('backward-send', log_level=2).start()
                self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=False,
                    recv_next=False,
                    tensor_shape=None,
                )
                if config.timers is not None:
                    config.timers('backward-send').stop()

    @nvtx_decorator()
    def send_forward_recv_backward(
        self, output_tensors, tensor_shapes, is_last_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with next rank in pipeline."""
        config = self.config
        unwrap_output_tensors = False
        if not isinstance(output_tensors, list):
            unwrap_output_tensors = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        output_tensor_grads = []
        for output_tensor, tensor_shape in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                output_tensor_grad = None
            else:
                if config.timers is not None:
                    config.timers('forward-send-backward-recv', log_level=2).start()
                _, output_tensor_grad, _ = self._communicate(
                    tensor_send_next=output_tensor,
                    tensor_send_prev=None,
                    recv_prev=False,
                    recv_next=True,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('forward-send-backward-recv').stop()
            output_tensor_grads.append(output_tensor_grad)
        if unwrap_output_tensors:
            return output_tensor_grads[0]
        return output_tensor_grads

    @nvtx_decorator()
    def send_backward_recv_forward(
        self, input_tensor_grads, tensor_shapes, is_first_stage: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """Batched send and recv with previous rank in pipeline."""
        config = self.config
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        input_tensors = []
        for input_tensor_grad, tensor_shape in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                input_tensor = None
            else:
                if config.timers is not None:
                    config.timers('backward-send-forward-recv', log_level=2).start()
                input_tensor, _, _ = self._communicate(
                    tensor_send_next=None,
                    tensor_send_prev=input_tensor_grad,
                    recv_prev=True,
                    recv_next=False,
                    tensor_shape=tensor_shape,
                )
                if config.timers is not None:
                    config.timers('backward-send-forward-recv').stop()
            input_tensors.append(input_tensor)
        if unwrap_input_tensor_grads:
            return input_tensors[0]
        return input_tensors

    @nvtx_decorator()
    def send_forward_recv_forward(
        self,
        output_tensor: torch.Tensor,
        recv_prev: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
        """Batched recv from previous rank and send to next rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-send-forward-recv', log_level=2).start()
        input_tensor, _, wait_handles = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=None,
            recv_prev=recv_prev,
            recv_next=False,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('forward-send-forward-recv').stop()
        if overlap_p2p_comm:
            return input_tensor, wait_handles
        return input_tensor

    @nvtx_decorator()
    def send_backward_recv_backward(
        self,
        input_tensor_grad: torch.Tensor,
        recv_next: bool,
        tensor_shape: Shape,
        overlap_p2p_comm: bool = False,
    ) -> torch.Tensor:
        """Batched recv from next rank and send to previous rank in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('backward-send-backward-recv', log_level=2).start()
        _, output_tensor_grad, wait_handles = self._communicate(
            tensor_send_next=None,
            tensor_send_prev=input_tensor_grad,
            recv_prev=False,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
            wait_on_reqs=(not overlap_p2p_comm),
        )
        if config.timers is not None:
            config.timers('backward-send-backward-recv').stop()
        if overlap_p2p_comm:
            return output_tensor_grad, wait_handles
        return output_tensor_grad

    @nvtx_decorator()
    def send_forward_backward_recv_forward_backward(
        self,
        output_tensor: torch.Tensor,
        input_tensor_grad: torch.Tensor,
        recv_prev: bool,
        recv_next: bool,
        tensor_shape: Shape,
    ) -> torch.Tensor:
        """Batched send and recv with previous and next ranks in pipeline."""
        config = self.config
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv', log_level=2).start()
        input_tensor, output_tensor_grad, _ = self._communicate(
            tensor_send_next=output_tensor,
            tensor_send_prev=input_tensor_grad,
            recv_prev=recv_prev,
            recv_next=recv_next,
            tensor_shape=tensor_shape,
        )
        if config.timers is not None:
            config.timers('forward-backward-send-forward-backward-recv').stop()
        return input_tensor, output_tensor_grad


def _ensure_local_nvshmem_compat():
    """Patch cuda-core module paths expected by some nvshmem4py builds.

    Keep this local to the OctoPipe communicator so it does not depend on the
    resharding NVSHMEM implementation or inherit future changes there.
    """
    for submod in ("_memory", "_stream"):
        exp_key = f"cuda.core.experimental.{submod}"
        new_key = f"cuda.core.{submod}"
        if exp_key not in sys.modules:
            try:
                sys.modules[exp_key] = import_module(new_key)
            except ImportError:
                pass


def _get_cuda_core_device_class():
    """Return cuda-core Device from the available cuda-core layout."""
    try:
        from cuda.core import Device

        return Device
    except ImportError:
        from cuda.core.experimental import Device

        return Device


class NvshmemP2PCommunicator:
    """NVSHMEM symmetric-memory P2P for pipeline stages (1F1B and OctoPipe).

    OctoPipe uses ``send_tensor_async`` / ``recv_tensor_async`` with peer ranks given as
    **global ranks in the PP process group** (same convention as ``P2PCommunicator``).
    Internally, peers are mapped to NVSHMEM PE indices ``0 .. pp_size-1``.

    Enabled via ``MEGATRON_NVSHMEM_P2P=1``; default NCCL path is unchanged when unset.
    """

    _instances = {}

    def __new__(cls, pp_group, config: object):
        # The OctoPipe schedule can construct a communicator every train
        # iteration when the caller does not pass one in.  NVSHMEM init and
        # symmetric allocations are process-lifetime resources; doing them
        # repeatedly leaks/fragment heaps and dominates iteration time.
        key = (id(pp_group), torch.cuda.current_device())
        instance = cls._instances.get(key)
        if instance is None:
            instance = super().__new__(cls)
            instance._nvshmem_initialized = False
            cls._instances[key] = instance
        return instance

    def __init__(self, pp_group, config: object):
        if getattr(self, "_nvshmem_initialized", False):
            self.config = config
            return
        self.pp_group = pp_group
        self.config = config
        self._rank = pp_group.rank()
        self._world = pp_group.size()
        self._pp_global_ranks = [
            torch.distributed.get_global_rank(pp_group, i) for i in range(self._world)
        ]
        self._global_to_pp = {g: i for i, g in enumerate(self._pp_global_ranks)}
        # NVSHMEM is initialized with rank in [0, pp_size) and nranks == pp_size only.
        # remote_pe in put/get must be a PE index in that same space, NOT a global torch rank.
        # Use linear neighbors (no wrap-around ring).
        self._next_pe = self._rank + 1 if self._rank + 1 < self._world else None
        self._prev_pe = self._rank - 1 if self._rank - 1 >= 0 else None
        self._mailboxes = {}
        self._shape_initialized = set()
        self._stream = None
        self._nv = None
        self._slot_bytes = int(os.environ.get("MEGATRON_NVSHMEM_P2P_BUFFER_BYTES", "0"))
        self._buffer_factor = int(os.environ.get("MEGATRON_NVSHMEM_P2P_BUFFER_FACTOR", "1"))
        self._pool_initialized = False
        self._data_slots = None
        self._send_buffer = None
        self._ready_flags = None
        self._meta = None
        self._meta_src_pe = None
        self._meta_sender_sid = None
        self._meta_recver_sid = None
        self._meta_mid = None
        self._meta_seq = None
        self._meta_num_bytes = None
        self._tmp_ready = None
        self._tmp_meta = None
        self._tmp_meta_src_pe = None
        self._tmp_meta_sender_sid = None
        self._tmp_meta_recver_sid = None
        self._tmp_meta_mid = None
        self._tmp_meta_seq = None
        self._tmp_meta_num_bytes = None
        self._slot_release_seq = {}
        # Per-shape ready sequence ids (pure NVSHMEM handshake, no NCCL send/recv).
        self._send_fwd_seq = {}
        self._send_bwd_seq = {}
        self._recv_fwd_seq = {}
        self._recv_bwd_seq = {}
        self._send_peer_seq = {}
        self._recv_peer_seq = {}
        self._local_peer_queues = {}
        self._pending_peer_payloads = {}
        # Bound pending stash growth to avoid unbounded GPU clones under
        # prolonged out-of-order traffic.
        self._pending_peer_payloads_max = int(
            os.environ.get("MEGATRON_NVSHMEM_PENDING_MAX", "256")
        )
        self._pending_overflow_warned = False
        self._route_slots = {}
        self._route_slot_counts = {}
        self._registered_workloads_id = None
        # Fixed-size NVSHMEM staging slots for routed OctoPipe peer traffic.
        self._peer_slots = int(os.environ.get("MEGATRON_NVSHMEM_P2P_NUM_SLOTS", "8"))
        assert self._peer_slots >= 2, "MEGATRON_NVSHMEM_P2P_NUM_SLOTS must be >= 2"
        self._trace_max = int(os.environ.get("MEGATRON_NVSHMEM_TRACE_MAX", "-1"))
        self._trace_count = 0
        self._quiet_after_send = os.environ.get("MEGATRON_NVSHMEM_P2P_QUIET_AFTER_SEND", "0") == "1"
        self._validate_expected_meta = (
            os.environ.get("MEGATRON_NVSHMEM_VALIDATE_EXPECTED_META", "0") == "1"
        )
        self._total_slots = self._world * self._peer_slots
        self._cuda_wait_until = None
        self._nv_bindings = None
        self._nv_cmp_eq = None
        self._nv_cmp_ge = None
        self._nv_signal_set = None
        self._init_nvshmem()
        self._nvshmem_initialized = True

    def _init_nvshmem(self):
        _ensure_local_nvshmem_compat()
        import nvshmem.core as nvshmem_core
        from nvshmem.bindings import nvshmem as nvshmem_bindings

        Device = _get_cuda_core_device_class()

        # Keep the single-node smoke-test workaround opt-in.  For production
        # multi-node runs, forcing NVSHMEM_REMOTE_TRANSPORT=none disables the
        # remote transport path and can prevent inter-node NVSHMEM P2P.
        if os.environ.get("MEGATRON_NVSHMEM_SINGLE_NODE", "0") == "1":
            os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")

        max_ctas = os.environ.get("NVSHMEM_MAX_CTAS")
        if max_ctas != "2":
            logger.warning(
                "Recommended NVSHMEM_MAX_CTAS=2 for this path. Current value is %r.", max_ctas
            )

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        dev = Device(local_rank)
        dev.set_current()
        self._stream = nvshmem_core.NvshmemStream(torch.cuda.current_stream())

        uid = nvshmem_core.get_unique_id(empty=self._rank != 0)
        objs = [None] * self._world
        torch.distributed.all_gather_object(objs, uid, group=self.pp_group)
        nvshmem_core.init(
            device=dev,
            uid=objs[0],
            rank=self._rank,
            nranks=self._world,
            initializer_method="uid",
        )
        self._nv = nvshmem_core
        self._nv_bindings = nvshmem_bindings
        self._nv_cmp_eq = nvshmem_bindings.Cmp_type.CMP_EQ
        self._nv_cmp_ge = nvshmem_bindings.Cmp_type.CMP_GE
        self._nv_signal_set = nvshmem_bindings.Signal_op.SIGNAL_SET
        self._init_p2p_pool()
        torch.distributed.barrier(group=self.pp_group)

    def _init_p2p_pool(self):
        """Allocate fixed-size symmetric staging buffers for OctoPipe peer traffic."""
        if self._pool_initialized:
            return
        if self._slot_bytes <= 0:
            inferred_bytes = self._infer_p2p_slot_bytes_from_config()
            # Lazily allocating symmetric memory after traffic starts is unsafe because
            # NVSHMEM allocations must happen in identical order on all PEs.  Infer
            # factor*b*s*h when config carries those fields; otherwise use an explicit
            # conservative default and let users override it for 4*b*s*h / 8*b*s*h runs.
            self._slot_bytes = int(
                os.environ.get(
                    "MEGATRON_NVSHMEM_P2P_DEFAULT_BUFFER_BYTES",
                    str(inferred_bytes if inferred_bytes is not None else 256 * 1024 * 1024),
                )
            )
        if self._slot_bytes <= 0:
            raise RuntimeError("MEGATRON_NVSHMEM_P2P_BUFFER_BYTES must be positive")

        torch.distributed.barrier(group=self.pp_group)
        self._data_slots = [self._nv.tensor((self._slot_bytes,), dtype=torch.uint8) for _ in range(self._total_slots)]
        self._send_buffer = self._nv.tensor((self._slot_bytes,), dtype=torch.uint8)
        self._ready_flags = [self._nv.tensor((1,), dtype=torch.int64) for _ in range(self._total_slots)]
        self._meta = [self._nv.tensor((6,), dtype=torch.int64) for _ in range(self._total_slots)]
        # Backward-compatible aliases for helper code and targeted debugging.
        self._meta_src_pe = [m[0:1] for m in self._meta]
        self._meta_sender_sid = [m[1:2] for m in self._meta]
        self._meta_recver_sid = [m[2:3] for m in self._meta]
        self._meta_mid = [m[3:4] for m in self._meta]
        self._meta_seq = [m[4:5] for m in self._meta]
        self._meta_num_bytes = [m[5:6] for m in self._meta]
        self._tmp_ready = self._nv.tensor((1,), dtype=torch.int64)
        self._tmp_meta = self._nv.tensor((6,), dtype=torch.int64)
        self._tmp_meta_src_pe = self._tmp_meta[0:1]
        self._tmp_meta_sender_sid = self._tmp_meta[1:2]
        self._tmp_meta_recver_sid = self._tmp_meta[2:3]
        self._tmp_meta_mid = self._tmp_meta[3:4]
        self._tmp_meta_seq = self._tmp_meta[4:5]
        self._tmp_meta_num_bytes = self._tmp_meta[5:6]
        with torch.no_grad():
            for slot in range(self._total_slots):
                self._data_slots[slot].zero_()
                self._ready_flags[slot].zero_()
                self._meta[slot][0].zero_()
                self._meta[slot][1].fill_(-1)
                self._meta[slot][2].fill_(-1)
                self._meta[slot][3].fill_(-1)
                self._meta[slot][4].zero_()
                self._meta[slot][5].zero_()
            self._send_buffer.zero_()
            self._tmp_ready.zero_()
            self._tmp_meta[0].zero_()
            self._tmp_meta[1].fill_(-1)
            self._tmp_meta[2].fill_(-1)
            self._tmp_meta[3].fill_(-1)
            self._tmp_meta[4].zero_()
            self._tmp_meta[5].zero_()
        torch.cuda.synchronize()
        torch.distributed.barrier(group=self.pp_group)
        self._pool_initialized = True

    def _route_requires_metadata(self, route_key) -> bool:
        return (
            isinstance(route_key, tuple)
            and len(route_key) == 3
            and all(v is not None for v in route_key)
        )

    def _validate_route_key(self, route_key):
        if not self._route_requires_metadata(route_key):
            raise RuntimeError(
                "NvshmemP2PCommunicator routed send/recv requires explicit sender_sid, "
                f"recver_sid, and mid; got route={route_key}."
            )

    def _tensor_nbytes(self, tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    def _shape_nbytes(self, tensor_shape: Shape) -> int:
        shape = tuple(tensor_shape) if not isinstance(tensor_shape, torch.Size) else tuple(tensor_shape)
        numel = 1
        for dim in shape:
            numel *= int(dim)
        return numel * torch.empty((), dtype=self.config.pipeline_dtype).element_size()

    def _slot_slice(self, slot: int, num_bytes: int):
        if slot < 0 or slot >= self._total_slots:
            raise RuntimeError(f"NVSHMEM P2P slot {slot} out of range [0,{self._total_slots})")
        if num_bytes < 0 or num_bytes > self._slot_bytes:
            raise RuntimeError(
                f"NVSHMEM P2P message size {num_bytes} bytes exceeds configured slot size "
                f"{self._slot_bytes} bytes. Increase MEGATRON_NVSHMEM_P2P_BUFFER_BYTES "
                "or reduce MEGATRON_NVSHMEM_P2P_BUFFER_FACTOR-derived payload size."
            )
        return self._data_slots[slot][:num_bytes]

    def _slot_for_route(self, route_key, seq: int, src_group_rank: int = None) -> int:
        # Partition the recv pool by source PE so simultaneous forward/backward
        # neighbors cannot alias the same slot on this rank.
        src_pe = self._rank if src_group_rank is None else int(src_group_rank)
        return src_pe * self._peer_slots + self._peer_slot(route_key, seq)

    def _wait_remote_slot_empty(self, dst_group_rank: int, slot: int):
        """Poll the destination PE's ready flag before reusing its staging slot."""
        self._nv.get(
            dst=self._tmp_ready,
            src=self._ready_flags[slot],
            remote_pe=dst_group_rank,
            stream=self._stream,
        )
        self._nv.quiet(stream=self._stream)
        self._call_nvshmem_wait_until(self._tmp_ready, 0, exact=True)

    def _write_remote_slot_meta(self, slot: int, dst_group_rank: int, route_key, seq: int, num_bytes: int):
        sender_sid, recver_sid, mid = route_key
        with torch.no_grad():
            self._tmp_meta[0].fill_(self._rank)
            self._tmp_meta[1].fill_(int(sender_sid))
            self._tmp_meta[2].fill_(int(recver_sid))
            self._tmp_meta[3].fill_(int(mid))
            self._tmp_meta[4].fill_(int(seq))
            self._tmp_meta[5].fill_(int(num_bytes))
        self._nv.put(
            dst=self._meta[slot],
            src=self._tmp_meta,
            remote_pe=dst_group_rank,
            stream=self._stream,
        )

    def _read_slot_meta(self, slot: int):
        meta = self._meta[slot]
        return {
            "src_pe": int(meta[0].item()),
            "sender_sid": int(meta[1].item()),
            "recver_sid": int(meta[2].item()),
            "mid": int(meta[3].item()),
            "seq": int(meta[4].item()),
            "num_bytes": int(meta[5].item()),
        }

    def _pending_key(self, src_group_rank: int, route_key, seq: int):
        return (int(src_group_rank), self._rank, route_key, int(seq))

    def _try_pop_pending_payload(self, out: torch.Tensor, src_group_rank: int, route_key, seq: int):
        key = self._pending_key(src_group_rank, route_key, seq)
        payload = self._pending_peer_payloads.pop(key, None)
        if payload is None:
            return False
        if payload.numel() != self._tensor_nbytes(out):
            raise RuntimeError(
                f"NVSHMEM pending payload size mismatch on rank {self._rank}: "
                f"payload={payload.numel()} expected={self._tensor_nbytes(out)} route={route_key}"
            )
        with torch.no_grad():
            out.view(torch.uint8).reshape(-1).copy_(payload, non_blocking=True)
        return True

    def _stash_ready_slot(self, slot: int, meta):
        route_key = (meta["sender_sid"], meta["recver_sid"], meta["mid"])
        key = self._pending_key(meta["src_pe"], route_key, meta["seq"])
        if key not in self._pending_peer_payloads:
            if len(self._pending_peer_payloads) >= self._pending_peer_payloads_max:
                raise RuntimeError(
                    "NVSHMEM pending payload stash is full on rank "
                    f"{self._rank}: max={self._pending_peer_payloads_max}. "
                    "Increase MEGATRON_NVSHMEM_PENDING_MAX."
                )
            payload = torch.empty(
                (meta["num_bytes"],), dtype=torch.uint8, device=torch.cuda.current_device()
            )
            with torch.no_grad():
                payload.copy_(self._slot_slice(slot, meta["num_bytes"]), non_blocking=True)
            self._pending_peer_payloads[key] = payload
            self._trace_event(
                f"stash src={meta['src_pe']} dst={self._rank} route={route_key} "
                f"seq={meta['seq']} slot={slot}"
            )
        self._release_slot(slot)

    def _drain_ready_slots(self, src_group_rank: int, expected_route=None, expected_seq=None):
        """Move ready but not-yet-consumed staging slots into owned tensors.

        OctoPipe can send several messages before the matching recv wait is reached.
        Draining prevents the fixed staging pool from filling and blocking senders.
        Returns the slot containing the expected message if it is currently ready.
        """
        expected_slot = None
        start = int(src_group_rank) * self._peer_slots
        end = start + self._peer_slots
        expected_tag = (
            self._ready_tag(expected_route, expected_seq)
            if expected_route is not None and expected_seq is not None
            else None
        )
        for slot in range(start, end):
            ready = int(self._ready_flags[slot][0].item())
            if ready == 0:
                continue
            if expected_tag is not None and ready == expected_tag:
                expected_slot = slot
                continue
            meta = self._read_slot_meta(slot)
            route_key = (meta["sender_sid"], meta["recver_sid"], meta["mid"])
            if (
                meta["src_pe"] == int(src_group_rank)
                and route_key == expected_route
                and meta["seq"] == int(expected_seq)
            ):
                expected_slot = slot
                continue
            self._stash_ready_slot(slot, meta)
        return expected_slot

    def _validate_slot_meta(self, slot: int, src_group_rank: int, route_key, expected_seq: int, expected_bytes: int):
        meta = self._read_slot_meta(slot)
        expected = {
            "src_pe": int(src_group_rank),
            "sender_sid": int(route_key[0]),
            "recver_sid": int(route_key[1]),
            "mid": int(route_key[2]),
            "seq": int(expected_seq),
            "num_bytes": int(expected_bytes),
        }
        mismatches = {k: (meta[k], v) for k, v in expected.items() if meta[k] != v}
        if mismatches:
            raise RuntimeError(
                f"NVSHMEM P2P slot metadata mismatch on rank {self._rank}, slot {slot}, "
                f"route={route_key}: {mismatches}"
            )

    def _release_slot(self, slot: int):
        with torch.no_grad():
            self._ready_flags[slot].zero_()
            self._meta[slot][5].zero_()
            self._meta[slot][1].fill_(-1)
            self._meta[slot][2].fill_(-1)
            self._meta[slot][3].fill_(-1)

    def _infer_p2p_slot_bytes_from_config(self):
        micro_batch_size = getattr(self.config, "micro_batch_size", None)
        seq_length = getattr(self.config, "seq_length", None)
        hidden_size = getattr(self.config, "hidden_size", None)
        if micro_batch_size is None or seq_length is None or hidden_size is None:
            return None
        dtype_size = torch.empty((), dtype=self.config.pipeline_dtype).element_size()
        return int(self._buffer_factor) * int(micro_batch_size) * int(seq_length) * int(hidden_size) * dtype_size

    def _shape_key(self, tensor_shape: Shape):
        shape = tuple(tensor_shape) if isinstance(tensor_shape, (list, tuple, torch.Size)) else (tensor_shape,)
        return (shape, str(self.config.pipeline_dtype))

    def _trace_event(self, message: str):
        # MEGATRON_NVSHMEM_TRACE_MAX semantics:
        #   <0 : disabled
        #    0 : unlimited
        #   >0 : print at most N lines per rank
        if self._trace_max < 0:
            return
        if self._trace_max > 0 and self._trace_count >= self._trace_max:
            return
        print(f"[nvshmem-trace][rank={self._rank}] {message}", flush=True)
        self._trace_count += 1

    def _ensure_mailboxes_for_shape(self, tensor_shape: Shape):
        """Allocate all channel mailboxes for a shape in a synchronized order.

        NVSHMEM symmetric allocations must be consistent across all PEs. Lazy per-branch
        allocations can diverge across ranks and deadlock/hang. We allocate all channels
        together once per shape with PP-group barriers.
        """
        shape = tuple(tensor_shape) if isinstance(tensor_shape, (list, tuple, torch.Size)) else (tensor_shape,)
        key = (shape, str(self.config.pipeline_dtype))
        if key in self._shape_initialized:
            return
        torch.distributed.barrier(group=self.pp_group)
        for channel in (
            "fwd",
            "bwd",
            "tmp_send_fwd",
            "tmp_send_bwd",
            "fwd_ready",
            "bwd_ready",
            "tmp_fwd_ready",
            "tmp_bwd_ready",
        ):
            mkey = (channel, shape, str(self.config.pipeline_dtype))
            if mkey not in self._mailboxes:
                if "ready" in channel:
                    # Sequence-number handshake flags are int32 scalars.
                    t = self._nv.tensor((1,), dtype=torch.int32)
                    t.zero_()
                    self._mailboxes[mkey] = t
                else:
                    self._mailboxes[mkey] = self._nv.tensor(shape, dtype=self.config.pipeline_dtype)

        # The fixed-size staging pool below is the only peer data path.  Avoid
        # allocating the older per-shape peer mailboxes here: they are unused by
        # routed OctoPipe traffic, waste symmetric heap memory, and make NVSHMEM
        # heap exhaustion more likely for large activations.
        torch.distributed.barrier(group=self.pp_group)
        self._shape_initialized.add(key)

    def register_workload_routes(self, workloads):
        workloads_id = id(workloads)
        if self._registered_workloads_id == workloads_id:
            return
        route_counts = {}
        for workload in workloads:
            if workload.get("op") not in ("send", "recv"):
                continue
            route_key = self._msg_route_key(
                sender_sid=workload.get("sender_sid"),
                recver_sid=workload.get("recver_sid"),
                mid=workload.get("mid"),
            )
            route_counts[route_key] = route_counts.get(route_key, 0) + 1
        self._route_slot_counts = route_counts
        self._route_slots = {}
        self._registered_workloads_id = workloads_id

    def _get_mailbox(self, tensor_shape: Shape, channel: str):
        self._ensure_mailboxes_for_shape(tensor_shape)
        key = (channel,) + self._shape_key(tensor_shape)
        if key not in self._mailboxes:
            shape = key[1]
            if "ready" in channel:
                ready_dtype = torch.int64 if "peer_" in channel else torch.int32
                t = self._nv.tensor((1,), dtype=ready_dtype)
                t.zero_()
                self._mailboxes[key] = t
            else:
                self._mailboxes[key] = self._nv.tensor(shape, dtype=self.config.pipeline_dtype)
        return self._mailboxes[key]

    def _communicate(self, tensor_send_next, tensor_send_prev, recv_prev: bool, recv_next: bool, tensor_shape: Shape):
        shape_key = self._shape_key(tensor_shape)
        # Write phase
        if tensor_send_next is not None:
            assert self._next_pe is not None
            send_src = self._get_mailbox(tensor_send_next.shape, "tmp_send_fwd")
            send_src.copy_(tensor_send_next)
            dst = self._get_mailbox(tensor_send_next.shape, "fwd")
            self._nv.put(dst=dst, src=send_src, remote_pe=self._next_pe, stream=self._stream)
            # NVSHMEM-only ready handshake: write sequence id to receiver's ready flag.
            seq = self._send_fwd_seq.get(shape_key, 0) + 1
            self._send_fwd_seq[shape_key] = seq
            ready_src = self._get_mailbox(tensor_send_next.shape, "tmp_fwd_ready")
            ready_src.fill_(seq)
            ready_dst = self._get_mailbox(tensor_send_next.shape, "fwd_ready")
            self._nv.put(dst=ready_dst, src=ready_src, remote_pe=self._next_pe, stream=self._stream)
        if tensor_send_prev is not None:
            assert self._prev_pe is not None
            send_src = self._get_mailbox(tensor_send_prev.shape, "tmp_send_bwd")
            send_src.copy_(tensor_send_prev)
            dst = self._get_mailbox(tensor_send_prev.shape, "bwd")
            self._nv.put(dst=dst, src=send_src, remote_pe=self._prev_pe, stream=self._stream)
            # NVSHMEM-only ready handshake: write sequence id to receiver's ready flag.
            seq = self._send_bwd_seq.get(shape_key, 0) + 1
            self._send_bwd_seq[shape_key] = seq
            ready_src = self._get_mailbox(tensor_send_prev.shape, "tmp_bwd_ready")
            ready_src.fill_(seq)
            ready_dst = self._get_mailbox(tensor_send_prev.shape, "bwd_ready")
            self._nv.put(dst=ready_dst, src=ready_src, remote_pe=self._prev_pe, stream=self._stream)

        tensor_recv_prev = None
        tensor_recv_next = None
        if recv_prev:
            expected = self._recv_fwd_seq.get(shape_key, 0) + 1
            ready_local = self._get_mailbox(tensor_shape, "fwd_ready")
            self._call_nvshmem_wait_until(ready_local, expected)
            self._recv_fwd_seq[shape_key] = expected
            local = self._get_mailbox(tensor_shape, "fwd")
            tensor_recv_prev = local.clone().requires_grad_(True)
        if recv_next:
            expected = self._recv_bwd_seq.get(shape_key, 0) + 1
            ready_local = self._get_mailbox(tensor_shape, "bwd_ready")
            self._call_nvshmem_wait_until(ready_local, expected)
            self._recv_bwd_seq[shape_key] = expected
            local = self._get_mailbox(tensor_shape, "bwd")
            tensor_recv_next = local.clone().requires_grad_(True)
        return tensor_recv_prev, tensor_recv_next, None

    class _NvshmemDoneHandle:
        def wait(self):
            return None

    class _NvshmemPeerRecvHandle:
        """Deferred NVSHMEM peer recv using the fixed OctoPipe staging pool."""

        __slots__ = ("_comm", "_out", "_shape", "_src_pe", "_done", "_route_key")

        def __init__(
            self, comm, out: torch.Tensor, tensor_shape: Shape, src_group_rank: int, route_key=None
        ):
            self._comm = comm
            self._out = out
            self._shape = tensor_shape
            self._src_pe = src_group_rank
            self._done = False
            self._route_key = route_key

        def wait(self):
            if self._done:
                return None
            self._comm._recv_peer_tensor_into(
                self._out, self._shape, self._src_pe, route_key=self._route_key
            )
            self._done = True
            return None

    class _NvshmemLocalRecvHandle:
        """Deferred pop from same-rank FIFO (OctoPipe co-located stages)."""

        __slots__ = ("_comm", "_outs", "_tensor_shapes", "_done", "_queue_key")

        def __init__(self, comm, outs: list, tensor_shapes: list, queue_key=None):
            self._comm = comm
            self._outs = outs
            self._tensor_shapes = tensor_shapes
            self._done = False
            self._queue_key = queue_key

        def wait(self):
            if self._done:
                return None
            key = (
                self._queue_key
                if self._queue_key is not None
                else (self._comm._rank, self._comm._rank, None)
            )
            q = self._comm._local_peer_queues.get(key)
            if q is None or len(q) == 0:
                raise RuntimeError("NVSHMEM local peer recv queue is empty (deferred wait)")
            payload = q.pop(0)
            if len(payload) != len(self._tensor_shapes):
                raise RuntimeError(
                    f"NVSHMEM local peer recv length mismatch: {len(payload)} vs {len(self._tensor_shapes)}"
                )
            for t, s, o in zip(payload, self._tensor_shapes, self._outs):
                if tuple(t.shape) != tuple(s):
                    raise RuntimeError(
                        f"NVSHMEM local peer recv shape mismatch: tensor={tuple(t.shape)} expected={tuple(s)}"
                    )
                with torch.no_grad():
                    o.copy_(t)
            self._done = True
            return None

    def _get_global_rank(self, group_rank: int) -> int:
        return torch.distributed.get_global_rank(self.pp_group, group_rank)

    def _msg_route_key(self, sender_sid=None, recver_sid=None, mid=None):
        # Keep transport routing independent per (sender, receiver, microbatch).
        # Physical symmetric mailboxes are still shared by src+slot only; this
        # key only affects logical matching/tagging/sequence bookkeeping.
        return (sender_sid, recver_sid, mid)

    def _ready_tag(self, route_key, seq: int) -> int:
        # Deterministic 63-bit tag: [sender(16)][recver(16)][mid(16)][seq(15)].
        if isinstance(route_key, tuple) and len(route_key) == 3:
            s_sid, r_sid, mid = route_key
        elif isinstance(route_key, tuple) and len(route_key) == 2:
            s_sid, r_sid = route_key
            mid = 0
        else:
            s_sid, r_sid, mid = (0, 0, 0)
        s = (0 if s_sid is None else int(s_sid)) & 0xFFFF
        r = (0 if r_sid is None else int(r_sid)) & 0xFFFF
        m = (0 if mid is None else int(mid)) & 0xFFFF
        q = int(seq) & 0x7FFF
        return (s << 47) | (r << 31) | (m << 15) | q

    def _decode_ready_tag(self, tag: int):
        tag = int(tag)
        seq = tag & 0x7FFF
        mid = (tag >> 15) & 0xFFFF
        recver_sid = (tag >> 31) & 0xFFFF
        sender_sid = (tag >> 47) & 0xFFFF
        return sender_sid, recver_sid, mid, seq

    def _route_from_tag(self, sender_sid: int, recver_sid: int, mid: int, route_key):
        if isinstance(route_key, tuple) and len(route_key) == 3:
            return (sender_sid, recver_sid, mid)
        if isinstance(route_key, tuple) and len(route_key) == 2:
            return (sender_sid, recver_sid)
        return (sender_sid, recver_sid)

    def _peer_slot(self, route_key, seq: int) -> int:
        # Deterministic mixed hash so sender/recver/mid/seq all affect slot selection.
        if isinstance(route_key, tuple) and len(route_key) == 2:
            s_sid, r_sid = route_key
            mid = 0
        elif isinstance(route_key, tuple) and len(route_key) == 3:
            s_sid, r_sid, mid = route_key
        else:
            s_sid, r_sid, mid = (0, 0, 0)
        s = (0 if s_sid is None else int(s_sid)) & 0xFFFFFFFF
        r = (0 if r_sid is None else int(r_sid)) & 0xFFFFFFFF
        m = (0 if mid is None else int(mid)) & 0xFFFFFFFF
        q = int(seq) & 0xFFFFFFFF
        mixed = (
            (s * 1315423911)
            ^ (r * 2654435761)
            ^ (m * 2246822519)
            ^ (q * 3266489917)
        ) & 0xFFFFFFFF
        return mixed % self._peer_slots

    def _get_cuda_wait_until(self):
        if self._cuda_wait_until is not None:
            return self._cuda_wait_until
        from torch.utils.cpp_extension import load_inline

        cpp_source = r'''
#include <torch/extension.h>

void wait_until(torch::Tensor flag, int64_t expected, bool exact);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wait_until", &wait_until, "Wait until a CUDA scalar reaches expected");
}
'''
        cuda_source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cstdint>

__global__ void wait_until_i32_kernel(const int* flag, int64_t expected, bool exact) {
    const volatile int* volatile_flag = flag;
    while (true) {
        int64_t value = (int64_t)(*volatile_flag);
        if ((exact && value == expected) || (!exact && value >= expected)) {
            break;
        }
        __nanosleep(64);
    }
}

__global__ void wait_until_i64_kernel(const int64_t* flag, int64_t expected, bool exact) {
    const volatile int64_t* volatile_flag = flag;
    while (true) {
        int64_t value = *volatile_flag;
        if ((exact && value == expected) || (!exact && value >= expected)) {
            break;
        }
        __nanosleep(64);
    }
}

void wait_until(torch::Tensor flag, int64_t expected, bool exact) {
    TORCH_CHECK(flag.is_cuda(), "wait flag must be a CUDA tensor");
    TORCH_CHECK(flag.numel() == 1, "wait flag must be scalar");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    if (flag.scalar_type() == at::kInt) {
        wait_until_i32_kernel<<<1, 1, 0, stream>>>(flag.data_ptr<int>(), expected, exact);
    } else if (flag.scalar_type() == at::kLong) {
        wait_until_i64_kernel<<<1, 1, 0, stream>>>(flag.data_ptr<int64_t>(), expected, exact);
    } else {
        TORCH_CHECK(false, "wait flag must be int32 or int64");
    }
}
'''
        module = load_inline(
            name="megatron_nvshmem_wait_until_ge",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=None,
            with_cuda=True,
            verbose=False,
        )
        self._cuda_wait_until = module.wait_until
        return self._cuda_wait_until

    def _call_nvshmem_wait_until(self, ready, expected: int, exact: bool = False):
        signal_wait = getattr(self._nv_bindings, "signal_wait_until_on_stream", None)
        cmp_op = self._nv_cmp_eq if exact else self._nv_cmp_ge
        if signal_wait is not None and cmp_op is not None:
            stream = int(self._stream.__cuda_stream__()[1])
            signal_wait(ready.data_ptr(), int(cmp_op), int(expected), stream)
            return
        self._get_cuda_wait_until()(ready, int(expected), exact)

    def _stream_handle(self) -> int:
        return int(self._stream.__cuda_stream__()[1])

    def _putmem_on_stream(self, dst: torch.Tensor, src: torch.Tensor, num_bytes: int, remote_pe: int) -> bool:
        putmem = getattr(self._nv_bindings, "putmem_on_stream", None)
        if putmem is None:
            return False
        putmem(
            dst.data_ptr(),
            src.data_ptr(),
            int(num_bytes),
            int(remote_pe),
            self._stream_handle(),
        )
        return True

    def _putmem_signal_on_stream(
        self,
        dst: torch.Tensor,
        src: torch.Tensor,
        num_bytes: int,
        signal_addr: torch.Tensor,
        signal_value: int,
        remote_pe: int,
    ) -> bool:
        putmem_signal = getattr(self._nv_bindings, "putmem_signal_on_stream", None)
        if putmem_signal is None or self._nv_signal_set is None:
            return False
        putmem_signal(
            dst.data_ptr(),
            src.data_ptr(),
            int(num_bytes),
            signal_addr.data_ptr(),
            int(signal_value),
            int(self._nv_signal_set),
            int(remote_pe),
            self._stream_handle(),
        )
        return True

    def _resolve_peer_group_rank(self, rank_maybe_global: int) -> int:
        # Prefer interpreting input as global rank if it belongs to this pp_group.
        # This avoids ambiguity when global rank id is in [0, pp_size).
        if rank_maybe_global in self._global_to_pp:
            return self._global_to_pp[rank_maybe_global]
        # Fallback: caller passed pp-group rank directly.
        if 0 <= rank_maybe_global < self._world:
            return rank_maybe_global
        raise RuntimeError(
            f"peer rank {rank_maybe_global} is neither a global rank in this pp_group "
            f"(global ranks={self._pp_global_ranks}) nor a valid pp-group rank [0,{self._world})."
        )

    def _send_peer_tensor(self, t: torch.Tensor, dst_group_rank: int, route_key=None):
        """Put a tensor into a bounded NVSHMEM slot and mark it ready with explicit metadata."""
        if dst_group_rank == self._rank:
            raise RuntimeError("NVSHMEM peer send does not support self-send")
        self._validate_route_key(route_key)
        if not t.is_contiguous():
            t = t.contiguous()
        num_bytes = self._tensor_nbytes(t)
        if num_bytes > self._slot_bytes:
            raise RuntimeError(
                f"NVSHMEM P2P route={route_key} tensor size {num_bytes} bytes exceeds slot size "
                f"{self._slot_bytes} bytes. Configure MEGATRON_NVSHMEM_P2P_BUFFER_BYTES for this run."
            )

        seq_key = (self._rank, dst_group_rank, route_key)
        seq = self._send_peer_seq.get(seq_key, 0) + 1
        slot = self._slot_for_route(route_key, seq, src_group_rank=self._rank)
        self._trace_event(
            f"send src={self._rank} dst={dst_group_rank} route={route_key} seq={seq} slot={slot} bytes={num_bytes}"
        )
        self._send_peer_seq[seq_key] = seq

        # Do not overwrite a remote staging slot until the receiver has copied
        # the previous payload out and cleared the ready flag.
        self._wait_remote_slot_empty(dst_group_rank, slot)

        dst_data = self._slot_slice(slot, num_bytes)
        if not self._putmem_on_stream(dst_data, t, num_bytes, dst_group_rank):
            send_src = self._send_buffer[:num_bytes]
            send_src.copy_(t.view(torch.uint8).reshape(-1))
            self._nv.put(dst=dst_data, src=send_src, remote_pe=dst_group_rank, stream=self._stream)

        with torch.no_grad():
            sender_sid, recver_sid, mid = route_key
            self._tmp_meta[0].fill_(self._rank)
            self._tmp_meta[1].fill_(int(sender_sid))
            self._tmp_meta[2].fill_(int(recver_sid))
            self._tmp_meta[3].fill_(int(mid))
            self._tmp_meta[4].fill_(int(seq))
            self._tmp_meta[5].fill_(int(num_bytes))
            ready_tag = self._ready_tag(route_key, seq)
        if not self._putmem_signal_on_stream(
            self._meta[slot],
            self._tmp_meta,
            self._tmp_meta.numel() * self._tmp_meta.element_size(),
            self._ready_flags[slot],
            ready_tag,
            dst_group_rank,
        ):
            self._nv.put(
                dst=self._meta[slot],
                src=self._tmp_meta,
                remote_pe=dst_group_rank,
                stream=self._stream,
            )
            with torch.no_grad():
                self._tmp_ready.fill_(ready_tag)
            self._nv.put(
                dst=self._ready_flags[slot],
                src=self._tmp_ready,
                remote_pe=dst_group_rank,
                stream=self._stream,
            )
        # Payload, metadata, ready signaling, and later source/staging reuse are
        # ordered on this CUDA stream.  Keep an opt-in quiet for debugging or for
        # NVSHMEM builds that require stronger local completion semantics.
        if self._quiet_after_send:
            self._nv.quiet(stream=self._stream)

    def _recv_peer_tensor_into(
        self, out: torch.Tensor, tensor_shape: Shape, src_group_rank: int, route_key=None
    ):
        """Wait for routed metadata, copy from staging into ``out``, then release the slot."""
        if src_group_rank == self._rank:
            raise RuntimeError("NVSHMEM peer recv does not support self-recv")
        self._validate_route_key(route_key)
        expected_key = (src_group_rank, self._rank, route_key)
        expected = self._recv_peer_seq.get(expected_key, 0) + 1
        expected_tag = self._ready_tag(route_key, expected)
        if self._try_pop_pending_payload(out, src_group_rank, route_key, expected):
            self._recv_peer_seq[expected_key] = expected
            self._trace_event(
                f"recv-pending src={src_group_rank} dst={self._rank} route={route_key} seq={expected}"
            )
            return

        slot = self._slot_for_route(route_key, expected)
        expected_bytes = self._shape_nbytes(tensor_shape)
        if expected_bytes > self._slot_bytes:
            raise RuntimeError(
                f"NVSHMEM P2P route={route_key} expected recv size {expected_bytes} bytes exceeds slot size "
                f"{self._slot_bytes} bytes. Configure MEGATRON_NVSHMEM_P2P_BUFFER_BYTES for this run."
            )

        slot = self._drain_ready_slots(
            src_group_rank, expected_route=route_key, expected_seq=expected
        )
        if slot is None:
            slot = self._slot_for_route(route_key, expected, src_group_rank=src_group_rank)
            self._trace_event(
                f"recv-wait src={src_group_rank} dst={self._rank} route={route_key} "
                f"seq={expected} slot={slot}"
            )
            ready_local = self._ready_flags[slot]
            self._call_nvshmem_wait_until(ready_local, expected_tag, exact=True)
        if self._validate_expected_meta:
            self._validate_slot_meta(slot, src_group_rank, route_key, expected, expected_bytes)
        self._trace_event(
            f"recv src={src_group_rank} dst={self._rank} route={route_key} "
            f"seq={expected} slot={slot} bytes={expected_bytes}"
        )
        self._recv_peer_seq[expected_key] = expected

        local = self._slot_slice(slot, expected_bytes)
        with torch.no_grad():
            out.view(torch.uint8).reshape(-1).copy_(local, non_blocking=True)
            self._release_slot(slot)

    def _recv_peer_tensor(
        self,
        tensor_shape: Shape,
        src_group_rank: int,
        route_key=None,
        *,
        requires_grad: bool = True,
    ):
        out = torch.empty(
            tuple(tensor_shape) if not isinstance(tensor_shape, torch.Size) else tensor_shape,
            dtype=self.config.pipeline_dtype,
            device=torch.cuda.current_device(),
            requires_grad=requires_grad,
        )
        self._recv_peer_tensor_into(out, tensor_shape, src_group_rank, route_key=route_key)
        return out

    def send_tensor_async(
        self,
        send_tensors,
        dst_rank,
        stop=False,
        *,
        sender_sid=None,
        recver_sid=None,
        mid=None,
    ):
        del stop
        if not isinstance(send_tensors, list):
            send_tensors = [send_tensors]
        dst_group_rank = self._resolve_peer_group_rank(dst_rank)
        route_key = self._msg_route_key(sender_sid=sender_sid, recver_sid=recver_sid, mid=mid)
        reqs = []
        if dst_group_rank == self._rank:
            key = (self._rank, self._rank, route_key)
            if key not in self._local_peer_queues:
                self._local_peer_queues[key] = []
            payload = []
            for t in send_tensors:
                if t is None:
                    raise RuntimeError(
                        "NvshmemP2PCommunicator.send_tensor_async: None tensor (same-rank queue)"
                    )
                payload.append(t.clone())
                reqs.append(self._NvshmemDoneHandle())
            self._local_peer_queues[key].append(payload)
            return reqs
        for t in send_tensors:
            if t is None:
                raise RuntimeError("NvshmemP2PCommunicator.send_tensor_async: None tensor in send_tensors")
            self._send_peer_tensor(t, dst_group_rank, route_key=route_key)
            reqs.append(self._NvshmemDoneHandle())
        return reqs

    def recv_tensor_async(
        self,
        tensor_shapes,
        recv_src_rank,
        *,
        sender_sid=None,
        recver_sid=None,
        mid=None,
        requires_grad: bool = True,
    ):
        """Match ``P2PCommunicator``: return preallocated buffers; completion in ``handle.wait()``."""
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]

        src_group_rank = self._resolve_peer_group_rank(recv_src_rank)
        route_key = self._msg_route_key(sender_sid=sender_sid, recver_sid=recver_sid, mid=mid)
        recvs = []
        reqs = []

        def _shape_tuple(s):
            return tuple(s) if not isinstance(s, torch.Size) else tuple(s)

        if src_group_rank == self._rank:
            outs = [
                torch.empty(
                    _shape_tuple(s),
                    dtype=self.config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=requires_grad,
                )
                for s in tensor_shapes
            ]
            h = self._NvshmemLocalRecvHandle(
                self, outs, tensor_shapes, queue_key=(self._rank, self._rank, route_key)
            )
            # OctoPipe loops all handles; one idempotent wait pops the whole payload.
            reqs = [h] * len(tensor_shapes)
            recvs = outs
            if unwrap_tensor_shapes:
                return recvs[0], reqs
            return recvs, reqs

        for s in tensor_shapes:
            out = torch.empty(
                _shape_tuple(s),
                dtype=self.config.pipeline_dtype,
                device=torch.cuda.current_device(),
                requires_grad=requires_grad,
            )
            recvs.append(out)
            reqs.append(
                self._NvshmemPeerRecvHandle(
                    self, out, s, src_group_rank, route_key=route_key
                )
            )

        if unwrap_tensor_shapes:
            return recvs[0], reqs
        return recvs, reqs

    def recv_forward(self, tensor_shapes, is_first_stage: bool):
        unwrap = False
        if is_single_shape(tensor_shapes):
            unwrap = True
            tensor_shapes = [tensor_shapes]
        out = []
        for shape in tensor_shapes:
            if is_first_stage:
                out.append(None)
            else:
                t, _, _ = self._communicate(None, None, True, False, shape)
                out.append(t)
        return out[0] if unwrap else out

    def recv_backward(self, tensor_shapes, is_last_stage: bool):
        unwrap = False
        if is_single_shape(tensor_shapes):
            unwrap = True
            tensor_shapes = [tensor_shapes]
        out = []
        for shape in tensor_shapes:
            if is_last_stage:
                out.append(None)
            else:
                _, t, _ = self._communicate(None, None, False, True, shape)
                out.append(t)
        return out[0] if unwrap else out

    def send_forward(self, output_tensors, is_last_stage: bool):
        if is_last_stage:
            return
        if not isinstance(output_tensors, list):
            output_tensors = [output_tensors]
        for t in output_tensors:
            self._communicate(t, None, False, False, t.shape)

    def send_backward(self, input_tensor_grads, is_first_stage: bool):
        if is_first_stage:
            return
        if not isinstance(input_tensor_grads, list):
            input_tensor_grads = [input_tensor_grads]
        for t in input_tensor_grads:
            self._communicate(None, t, False, False, t.shape)

    def send_forward_recv_backward(self, output_tensors, tensor_shapes, is_last_stage: bool):
        unwrap_output_tensors = False
        if not isinstance(output_tensors, list):
            unwrap_output_tensors = True
            output_tensors = [output_tensors]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        assert len(output_tensors) == len(
            tensor_shapes
        ), f"send_forward_recv_backward length mismatch: {len(output_tensors)} vs {len(tensor_shapes)}"
        out = []
        for t, s in zip(output_tensors, tensor_shapes):
            if is_last_stage:
                # Last stage has no forward peer and should not send anything.
                out.append(None)
            else:
                assert t is not None, "send_forward_recv_backward got None output tensor"
                assert tuple(t.shape) == tuple(s), (
                    "send_forward_recv_backward shape mismatch: "
                    f"tensor={tuple(t.shape)} expected={tuple(s)}"
                )
                _, grad, _ = self._communicate(t, None, False, True, s)
                out.append(grad)
        return out[0] if unwrap_output_tensors else out

    def send_backward_recv_forward(self, input_tensor_grads, tensor_shapes, is_first_stage: bool):
        unwrap_input_tensor_grads = False
        if not isinstance(input_tensor_grads, list):
            unwrap_input_tensor_grads = True
            input_tensor_grads = [input_tensor_grads]
        if not isinstance(tensor_shapes, list):
            tensor_shapes = [tensor_shapes]
        assert len(input_tensor_grads) == len(
            tensor_shapes
        ), f"send_backward_recv_forward length mismatch: {len(input_tensor_grads)} vs {len(tensor_shapes)}"
        out = []
        for t, s in zip(input_tensor_grads, tensor_shapes):
            if is_first_stage:
                # First stage has no backward peer and should not send anything.
                out.append(None)
            else:
                assert t is not None, "send_backward_recv_forward got None grad tensor"
                assert tuple(t.shape) == tuple(s), (
                    "send_backward_recv_forward shape mismatch: "
                    f"tensor={tuple(t.shape)} expected={tuple(s)}"
                )
                inp, _, _ = self._communicate(None, t, True, False, s)
                out.append(inp)
        return out[0] if unwrap_input_tensor_grads else out


class OctoPipeP2PCommunicator(NvshmemP2PCommunicator):
    """Comp-driven NVSHMEM P2P helper for OctoPipe schedules.

    This exposes a non-blocking ``try_recv_tensor`` path so the schedule can
    check local/staged payloads before deciding to wait.  The underlying slot
    protocol and route metadata are inherited from ``NvshmemP2PCommunicator``.
    """

    def _route_kwargs(self, sender_sid, recver_sid, mid):
        return {"sender_sid": sender_sid, "recver_sid": recver_sid, "mid": mid}

    def try_recv_tensor(
        self,
        tensor_shapes,
        recv_src_rank,
        *,
        sender_sid=None,
        recver_sid=None,
        mid=None,
        requires_grad: bool = True,
    ):
        unwrap_tensor_shapes = False
        if is_single_shape(tensor_shapes):
            unwrap_tensor_shapes = True
            tensor_shapes = [tensor_shapes]

        src_group_rank = self._resolve_peer_group_rank(recv_src_rank)
        route_key = self._msg_route_key(sender_sid=sender_sid, recver_sid=recver_sid, mid=mid)
        self._validate_route_key(route_key)

        def _shape_tuple(s):
            return tuple(s) if not isinstance(s, torch.Size) else tuple(s)

        if src_group_rank == self._rank:
            key = (self._rank, self._rank, route_key)
            q = self._local_peer_queues.get(key)
            if q is None or len(q) == 0:
                return None
            payload = q.pop(0)
            outs = []
            if len(payload) != len(tensor_shapes):
                raise RuntimeError(
                    f"OctoPipe local recv length mismatch: {len(payload)} vs {len(tensor_shapes)}"
                )
            for t, s in zip(payload, tensor_shapes):
                if tuple(t.shape) != tuple(s):
                    raise RuntimeError(
                        f"OctoPipe local recv shape mismatch: tensor={tuple(t.shape)} expected={tuple(s)}"
                    )
                out = torch.empty(
                    _shape_tuple(s),
                    dtype=self.config.pipeline_dtype,
                    device=torch.cuda.current_device(),
                    requires_grad=requires_grad,
                )
                with torch.no_grad():
                    out.copy_(t)
                outs.append(out)
            return outs[0] if unwrap_tensor_shapes else outs

        outs = []
        expected_key = (src_group_rank, self._rank, route_key)
        expected = self._recv_peer_seq.get(expected_key, 0) + 1
        expected_tag = self._ready_tag(route_key, expected)

        for s in tensor_shapes:
            out = torch.empty(
                _shape_tuple(s),
                dtype=self.config.pipeline_dtype,
                device=torch.cuda.current_device(),
                requires_grad=requires_grad,
            )
            if self._try_pop_pending_payload(out, src_group_rank, route_key, expected):
                self._recv_peer_seq[expected_key] = expected
                outs.append(out)
                expected += 1
                expected_tag = self._ready_tag(route_key, expected)
                continue

            slot = self._slot_for_route(route_key, expected, src_group_rank=src_group_rank)
            ready = int(self._ready_flags[slot][0].item())
            if ready != expected_tag:
                return None

            expected_bytes = self._shape_nbytes(s)
            if self._validate_expected_meta:
                self._validate_slot_meta(slot, src_group_rank, route_key, expected, expected_bytes)
            local = self._slot_slice(slot, expected_bytes)
            with torch.no_grad():
                out.view(torch.uint8).reshape(-1).copy_(local, non_blocking=True)
                self._release_slot(slot)
            self._recv_peer_seq[expected_key] = expected
            outs.append(out)
            expected += 1
            expected_tag = self._ready_tag(route_key, expected)

        return outs[0] if unwrap_tensor_shapes else outs

    def recv_tensor_blocking(self, tensor_shapes, recv_src_rank, **route_kwargs):
        result = self.try_recv_tensor(tensor_shapes, recv_src_rank, **route_kwargs)
        if result is not None:
            return result
        recvs, handles = self.recv_tensor_async(tensor_shapes, recv_src_rank, **route_kwargs)
        for handle in handles:
            handle.wait()
        return recvs
