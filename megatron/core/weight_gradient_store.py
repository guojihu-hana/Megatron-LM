
import queue
# from megatron.training import get_args
# from megatron.core import parallel_state
from contextlib import contextmanager

class WeightGradStore:

    should_split_bw = False
    cache = []
    weight_grad_queue = None  # lazy init

    @classmethod
    def lazy_init(cls):
        if cls.weight_grad_queue is not None:
            return
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        # num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        num_chunks = 1
        # chunk id => seq id => Queue
        num_seq_splits = 1
        cls.weight_grad_queue = [[queue.Queue() for _ in range(num_seq_splits)] for _ in range(num_chunks)]

    @classmethod
    def is_supported(cls):
        """If not supported, fallback to original schedule."""
        # args = get_args()
        # if args.pipeline_model_parallel_size <= 1:
        #     return False
        # # if args.virtual_pipeline_model_parallel_size is not None:
        # #     return False
        # if args.overlap_grad_reduce:
        #     # the logic of overlapping grad reduce should be changed
        #     return False
        # if not args.gradient_accumulation_fusion:
        #     return False
        # # if args.transformer_impl == 'transformer_engine':
        # #     # hard to capture weight gradient computation for transformer_engine
        # #     return False
        return True

    @classmethod
    def split_bw(cls):
        if not cls.is_supported():
            return False
        return cls.should_split_bw

    @classmethod
    def enable_split_bw(cls):
        cls.should_split_bw = True

    @classmethod
    def disable_split_bw(cls):
        cls.should_split_bw = False

    @classmethod
    @contextmanager
    def set_split_bw(cls, enabled: bool):
        prev = cls.should_split_bw
        cls.should_split_bw = enabled
        try:
            yield
        finally:
            cls.should_split_bw = prev

    @classmethod
    def put(cls, weight, pre_func, func):
        assert cls.split_bw()
        # func(*pre_func(async_op=False))
        cls.cache.append((weight, pre_func, func))
        return

    @classmethod
    def queue_size(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        return WeightGradStore.weight_grad_queue[chunk][seq_split_idx].qsize()

    @classmethod
    def flush(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        # Or W later will consume empty computation and leak the non-empty computation.
        if not cls.split_bw():
            assert len(cls.cache) == 0
            return
        cls.weight_grad_queue[chunk][seq_split_idx].put(cls.cache)
        cls.cache = []

    @classmethod
    def pop(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        if cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
            for weight, pre_func, func in stored_grads:
                func(*pre_func(async_op=False))
        # else:
        #     rank = parallel_state.get_pipeline_model_parallel_rank()
        #     raise Exception(f"Pop empty queue. rank {rank}")

    @classmethod
    def clear(cls, model, chunk=0, seq_split_idx=0):
        cls.lazy_init()
        weight_grad_tasks = []
        while cls.weight_grad_queue[chunk][seq_split_idx].qsize() > 0:
            stored_grads = cls.weight_grad_queue[chunk][seq_split_idx].get()
            if len(weight_grad_tasks) == 0:
                for _ in stored_grads:
                    weight_grad_tasks.append([])
            else:
                assert len(weight_grad_tasks) == len(stored_grads)
            for i, task in enumerate(stored_grads):
                weight_grad_tasks[i].append(task)

        for i in range(len(weight_grad_tasks)):
            tasks = weight_grad_tasks[i]
            param = None
            for j in range(len(tasks)):
                weight, pre_func, func = tasks[j]
                if param is None:
                    param = weight
                assert param.storage().data_ptr() == weight.storage().data_ptr()
                func(*pre_func(async_op=False))
                tasks[j] = None  # release memory
            
            weight_grad_tasks[i] = None  # release memory