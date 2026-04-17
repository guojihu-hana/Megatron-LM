# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import os
from typing import List, Optional, Tuple, Union

import torch
import torch.distributed as dist

from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.utils import nvtx_decorator

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
                    torch.distributed.isend, send_prev_shape_tensor, self.prev_rank
                )
                ops.append(send_prev_op)
            if recv_prev_shape_tensor is not None:
                recv_prev_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_prev_shape_tensor, self.prev_rank
                )
                ops.append(recv_prev_op)
            if send_next_shape_tensor is not None:
                send_next_op = torch.distributed.P2POp(
                    torch.distributed.isend, send_next_shape_tensor, self.next_rank
                )
                ops.append(send_next_op)
            if recv_next_shape_tensor is not None:
                recv_next_op = torch.distributed.P2POp(
                    torch.distributed.irecv, recv_next_shape_tensor, self.next_rank
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

        if not config.variable_seq_lengths:
            recv_prev_shape = tensor_shape
            recv_next_shape = tensor_shape
        else:
            recv_prev_shape, recv_next_shape = self._communicate_shapes(
                tensor_send_next, tensor_send_prev, recv_prev, recv_next
            )

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


class NvshmemP2PCommunicator:
    """NVSHMEM symmetric-memory P2P for pipeline stages (1F1B and OctoPipe).

    OctoPipe uses ``send_tensor_async`` / ``recv_tensor_async`` with peer ranks given as
    **global ranks in the PP process group** (same convention as ``P2PCommunicator``).
    Internally, peers are mapped to NVSHMEM PE indices ``0 .. pp_size-1``.

    Enabled via ``MEGATRON_NVSHMEM_P2P=1``; default NCCL path is unchanged when unset.
    """

    def __init__(self, pp_group, config: object):
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
        # Ring buffer slots per peer channel to avoid payload overwrite.
        self._peer_slots = int(os.environ.get("MEGATRON_NVSHMEM_PEER_SLOTS", "8"))
        assert self._peer_slots >= 2, "MEGATRON_NVSHMEM_PEER_SLOTS must be >= 2"
        self._trace_max = int(os.environ.get("MEGATRON_NVSHMEM_TRACE_MAX", "0"))
        self._trace_count = 0
        self._warned_remote_ready_fallback = False
        self._init_nvshmem()

    def _init_nvshmem(self):
        import nvshmem.core as nvshmem_core
        from cuda.core.experimental import Device

        os.environ.setdefault("NVSHMEM_REMOTE_TRANSPORT", "none")
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
        torch.distributed.barrier(group=self.pp_group)

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

        # Allocate point-to-point channels for arbitrary PP-rank communication.
        for src in range(self._world):
            for slot in range(self._peer_slots):
                for channel in (
                    f"peer_data_from_{src}_slot_{slot}",
                    f"peer_ready_from_{src}_slot_{slot}",
                ):
                    mkey = (channel, shape, str(self.config.pipeline_dtype))
                    if mkey not in self._mailboxes:
                        if "ready" in channel:
                            t = self._nv.tensor((1,), dtype=torch.int64)
                            t.zero_()
                            self._mailboxes[mkey] = t
                        else:
                            self._mailboxes[mkey] = self._nv.tensor(shape, dtype=self.config.pipeline_dtype)
        for channel in ("peer_tmp_data", "peer_tmp_ready"):
            mkey = (channel, shape, str(self.config.pipeline_dtype))
            if mkey not in self._mailboxes:
                if "ready" in channel:
                    t = self._nv.tensor((1,), dtype=torch.int64)
                    t.zero_()
                    self._mailboxes[mkey] = t
                else:
                    self._mailboxes[mkey] = self._nv.tensor(shape, dtype=self.config.pipeline_dtype)
        torch.distributed.barrier(group=self.pp_group)
        self._shape_initialized.add(key)

    def register_workload_routes(self, workloads):
        # No-op: keep API compatibility with schedules.py.
        del workloads

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
            self._nv.quiet(stream=self._stream)
            # NVSHMEM-only ready handshake: write sequence id to receiver's ready flag.
            seq = self._send_fwd_seq.get(shape_key, 0) + 1
            self._send_fwd_seq[shape_key] = seq
            ready_src = self._get_mailbox(tensor_send_next.shape, "tmp_fwd_ready")
            ready_src.fill_(seq)
            ready_dst = self._get_mailbox(tensor_send_next.shape, "fwd_ready")
            self._nv.put(dst=ready_dst, src=ready_src, remote_pe=self._next_pe, stream=self._stream)
            self._nv.quiet(stream=self._stream)
        if tensor_send_prev is not None:
            assert self._prev_pe is not None
            send_src = self._get_mailbox(tensor_send_prev.shape, "tmp_send_bwd")
            send_src.copy_(tensor_send_prev)
            dst = self._get_mailbox(tensor_send_prev.shape, "bwd")
            self._nv.put(dst=dst, src=send_src, remote_pe=self._prev_pe, stream=self._stream)
            self._nv.quiet(stream=self._stream)
            # NVSHMEM-only ready handshake: write sequence id to receiver's ready flag.
            seq = self._send_bwd_seq.get(shape_key, 0) + 1
            self._send_bwd_seq[shape_key] = seq
            ready_src = self._get_mailbox(tensor_send_prev.shape, "tmp_bwd_ready")
            ready_src.fill_(seq)
            ready_dst = self._get_mailbox(tensor_send_prev.shape, "bwd_ready")
            self._nv.put(dst=ready_dst, src=ready_src, remote_pe=self._prev_pe, stream=self._stream)
            self._nv.quiet(stream=self._stream)

        tensor_recv_prev = None
        tensor_recv_next = None
        if recv_prev:
            expected = self._recv_fwd_seq.get(shape_key, 0) + 1
            ready_local = self._get_mailbox(tensor_shape, "fwd_ready")
            # Busy-wait on local symmetric flag; peer updates it with nvshmem.put.
            while int(ready_local.item()) < expected:
                pass
            self._recv_fwd_seq[shape_key] = expected
            local = self._get_mailbox(tensor_shape, "fwd")
            tensor_recv_prev = local.clone().requires_grad_(True)
        if recv_next:
            expected = self._recv_bwd_seq.get(shape_key, 0) + 1
            ready_local = self._get_mailbox(tensor_shape, "bwd_ready")
            # Busy-wait on local symmetric flag; peer updates it with nvshmem.put.
            while int(ready_local.item()) < expected:
                pass
            self._recv_bwd_seq[shape_key] = expected
            local = self._get_mailbox(tensor_shape, "bwd")
            tensor_recv_next = local.clone().requires_grad_(True)
        return tensor_recv_prev, tensor_recv_next, None

    class _NvshmemDoneHandle:
        def wait(self):
            return None

    class _NvshmemPeerRecvHandle:
        """Deferred NVSHMEM peer recv — matches NCCL ``irecv`` + ``wait()`` semantics for OctoPipe."""

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

    def _peek_remote_ready(self, tensor_shape: Shape, src_group_rank: int, slot: int, remote_pe: int) -> int:
        """Read remote PE's ready flag for src+slot. Returns int64 tag (0 means empty)."""
        ch_ready = f"peer_ready_from_{src_group_rank}_slot_{slot}"
        remote_ready = self._get_mailbox(tensor_shape, ch_ready)
        tmp_ready = self._get_mailbox(tensor_shape, "peer_tmp_ready")
        try:
            # Preferred: probe receiver-side occupancy via remote GET.
            self._nv.get(dst=tmp_ready, src=remote_ready, remote_pe=remote_pe, stream=self._stream)
            self._nv.quiet(stream=self._stream)
            return int(tmp_ready.item())
        except Exception:
            # Fallback when Python binding lacks get(): preserves previous behavior.
            if not self._warned_remote_ready_fallback:
                self._warned_remote_ready_fallback = True
                self._trace_event(
                    "warn remote-ready-probe-fallback=local-item "
                    "(nvshmem.get unavailable in current python binding)"
                )
            return int(remote_ready.item())

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
        if dst_group_rank == self._rank:
            raise RuntimeError("NVSHMEM peer send does not support self-send")
        shape_key = self._shape_key(t.shape)
        self._ensure_mailboxes_for_shape(t.shape)

        ch_data_tmp = "peer_tmp_data"
        ch_ready_tmp = "peer_tmp_ready"

        tmp_data = self._get_mailbox(t.shape, ch_data_tmp)
        tmp_data.copy_(t)
        seq_key = (self._rank, dst_group_rank, shape_key, route_key)
        seq = self._send_peer_seq.get(seq_key, 0) + 1
        preferred_slot = self._peer_slot(route_key, seq)
        slot = None
        dst_ready = None
        spin = 0
        max_spin = int(os.environ.get("MEGATRON_NVSHMEM_SEND_WAIT_MAX_SPIN", "0"))
        while slot is None:
            # Probe for any free slot to avoid head-of-line blocking.
            for offset in range(self._peer_slots):
                cand_slot = (preferred_slot + offset) % self._peer_slots
                seen_tag = self._peek_remote_ready(
                    t.shape, src_group_rank=self._rank, slot=cand_slot, remote_pe=dst_group_rank
                )
                if seen_tag == 0:
                    slot = cand_slot
                    ch_ready_cand = f"peer_ready_from_{self._rank}_slot_{cand_slot}"
                    dst_ready = self._get_mailbox(t.shape, ch_ready_cand)
                    break
            if max_spin > 0:
                spin += 1
            if max_spin > 0 and spin >= max_spin:
                raise RuntimeError(
                    "NVSHMEM peer send wait timeout: no free slot on receiver. "
                    f"src_pe={self._rank} dst_pe={dst_group_rank} preferred_slot={preferred_slot} "
                    f"route_key={route_key} seq={seq}"
                )
        ch_data_dst = f"peer_data_from_{self._rank}_slot_{slot}"
        ch_ready_dst = f"peer_ready_from_{self._rank}_slot_{slot}"
        self._trace_event(
            f"send src={self._rank} dst={dst_group_rank} route={route_key} seq={seq} slot={slot} pref={preferred_slot}"
        )
        self._send_peer_seq[seq_key] = seq

        dst_data = self._get_mailbox(t.shape, ch_data_dst)
        self._nv.put(dst=dst_data, src=tmp_data, remote_pe=dst_group_rank, stream=self._stream)
        self._nv.quiet(stream=self._stream)
        tmp_ready = self._get_mailbox(t.shape, ch_ready_tmp)
        tmp_ready.fill_(self._ready_tag(route_key, seq))
        self._nv.put(dst=dst_ready, src=tmp_ready, remote_pe=dst_group_rank, stream=self._stream)
        self._nv.quiet(stream=self._stream)

    def _recv_peer_tensor_into(
        self, out: torch.Tensor, tensor_shape: Shape, src_group_rank: int, route_key=None
    ):
        """Block until peer data is ready, then copy symmetric mailbox into ``out``."""
        if src_group_rank == self._rank:
            raise RuntimeError("NVSHMEM peer recv does not support self-recv")
        shape_key = self._shape_key(tensor_shape)
        self._ensure_mailboxes_for_shape(tensor_shape)

        expected_key = (src_group_rank, self._rank, shape_key, route_key)
        expected = self._recv_peer_seq.get(expected_key, 0) + 1
        expected_tag = self._ready_tag(route_key, expected)
        pending_key = (src_group_rank, self._rank, shape_key, route_key, expected)
        pending = self._pending_peer_payloads.pop(pending_key, None)
        if pending is not None:
            with torch.no_grad():
                out.copy_(pending)
            self._trace_event(
                f"recv-pending src={src_group_rank} dst={self._rank} route={route_key} seq={expected}"
            )
            self._recv_peer_seq[expected_key] = expected
            return
        spin = 0
        matched_slot = None
        matched_ready = None
        # Diagnostic timeout is opt-in. Default (0) waits indefinitely.
        max_spin = int(os.environ.get("MEGATRON_NVSHMEM_WAIT_MAX_SPIN", "0"))
        while matched_slot is None:
            for slot in range(self._peer_slots):
                ch_ready = f"peer_ready_from_{src_group_rank}_slot_{slot}"
                ready_local = self._get_mailbox(tensor_shape, ch_ready)
                seen_tag = int(ready_local.item())
                if seen_tag == 0:
                    continue
                if seen_tag == expected_tag:
                    matched_slot = slot
                    matched_ready = ready_local
                    break
                # Receive-and-stash unexpected message to free the slot.
                s_sid, r_sid, m_sid, q_seq = self._decode_ready_tag(seen_tag)
                stash_route = self._route_from_tag(s_sid, r_sid, m_sid, route_key)
                stash_key = (src_group_rank, self._rank, shape_key, stash_route, q_seq)
                if stash_key not in self._pending_peer_payloads:
                    if self._pending_peer_payloads_max <= 0 or len(self._pending_peer_payloads) < self._pending_peer_payloads_max:
                        ch_data_seen = f"peer_data_from_{src_group_rank}_slot_{slot}"
                        local_seen = self._get_mailbox(tensor_shape, ch_data_seen)
                        self._pending_peer_payloads[stash_key] = local_seen.clone()
                        self._trace_event(
                            f"stash src={src_group_rank} dst={self._rank} route={stash_route} seq={q_seq} slot={slot}"
                        )
                        with torch.no_grad():
                            ready_local.zero_()
                    else:
                        if not self._pending_overflow_warned:
                            self._pending_overflow_warned = True
                            self._trace_event(
                                f"warn pending-overflow size={len(self._pending_peer_payloads)} "
                                f"limit={self._pending_peer_payloads_max}; stop stashing unexpected payloads"
                            )
            if max_spin > 0:
                spin += 1
            if max_spin > 0 and spin >= max_spin:
                seen_tag = 0
                seen_slot = -1
                for slot in range(self._peer_slots):
                    ch_ready = f"peer_ready_from_{src_group_rank}_slot_{slot}"
                    cand = int(self._get_mailbox(tensor_shape, ch_ready).item())
                    if cand != 0:
                        seen_tag = cand
                        seen_slot = slot
                        break
                s_sid, r_sid, m_sid, q_seq = self._decode_ready_tag(seen_tag)
                if isinstance(route_key, tuple) and len(route_key) == 2:
                    exp_s, exp_r, exp_m = route_key[0], route_key[1], 0
                elif isinstance(route_key, tuple) and len(route_key) == 3:
                    exp_s, exp_r, exp_m = route_key
                else:
                    exp_s, exp_r, exp_m = (None, None, None)
                raise RuntimeError(
                    "NVSHMEM peer recv wait timeout: sender not reached yet or mailbox collision. "
                    f"src_pe={src_group_rank} dst_pe={self._rank} slot={seen_slot} "
                    f"expected_tag={expected_tag} expected_route=({exp_s},{exp_r},{exp_m}) expected_seq={expected} "
                    f"seen_tag={seen_tag} seen_route=({s_sid},{r_sid},{m_sid}) seen_seq={q_seq}"
                )
        self._trace_event(
            f"recv src={src_group_rank} dst={self._rank} route={route_key} seq={expected} slot={matched_slot}"
        )
        self._recv_peer_seq[expected_key] = expected

        ch_data = f"peer_data_from_{src_group_rank}_slot_{matched_slot}"
        local = self._get_mailbox(tensor_shape, ch_data)
        # Avoid in-place on a leaf that requires_grad (autograd forbids copy_ on such leaves).
        with torch.no_grad():
            out.copy_(local)
            # Mark this slot as reusable after payload is consumed.
            matched_ready.zero_()

    def _recv_peer_tensor(self, tensor_shape: Shape, src_group_rank: int, route_key=None):
        out = torch.empty(
            tuple(tensor_shape) if not isinstance(tensor_shape, torch.Size) else tensor_shape,
            dtype=self.config.pipeline_dtype,
            device=torch.cuda.current_device(),
            requires_grad=True,
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
                    requires_grad=True,
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
                requires_grad=True,
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
