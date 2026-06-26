
import queue
from contextlib import contextmanager

# from megatron.training import get_args
# from megatron.core import parallel_state

class WeightGradStore:

    should_split_bw = False
    cache = {}
    tagged_tasks = {}
    weight_grad_queue = None  # lazy init

    @classmethod
    def lazy_init(cls, num_chunks=1, num_seq_splits=1):
        # Lazy init to make sure parallel_state and get_args() have been initialized.
        # num_chunks = parallel_state.get_virtual_pipeline_model_parallel_world_size() or 1
        if cls.weight_grad_queue is None:
            cls.weight_grad_queue = []
        while len(cls.weight_grad_queue) < num_chunks:
            cls.weight_grad_queue.append([])
        for chunk_queues in cls.weight_grad_queue:
            while len(chunk_queues) < num_seq_splits:
                chunk_queues.append(queue.Queue())

    @classmethod
    def _ensure_queue(cls, chunk=0, seq_split_idx=0):
        cls.lazy_init(num_chunks=chunk + 1, num_seq_splits=seq_split_idx + 1)
        return cls.weight_grad_queue[chunk][seq_split_idx]

    @staticmethod
    def _cache_key(chunk=0, seq_split_idx=0, tag=None):
        return chunk, seq_split_idx, tag

    @staticmethod
    def _task_key(chunk=0, seq_split_idx=0, tag=None):
        return chunk, seq_split_idx, tag

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
        cls.put_task(
            lambda: func(*pre_func(async_op=False)),
            description=getattr(weight, "shape", None),
        )
        return

    @classmethod
    def put_task(cls, task, description=None, chunk=0, seq_split_idx=0, tag=None):
        """Cache a delayed weight-gradient task for the current bwd split."""
        assert cls.split_bw()
        if not callable(task):
            raise TypeError("WeightGradStore task must be callable")
        key = cls._cache_key(chunk, seq_split_idx, tag)
        cls.cache.setdefault(key, []).append((task, description))

    @classmethod
    def queue_size(cls, chunk=0, seq_split_idx=0):
        return cls._ensure_queue(chunk, seq_split_idx).qsize()

    @classmethod
    def flush(cls, chunk=0, seq_split_idx=0, tag=None):
        cls._ensure_queue(chunk, seq_split_idx)
        # Or W later will consume empty computation and leak the non-empty computation.
        if not cls.split_bw():
            assert all(len(tasks) == 0 for tasks in cls.cache.values())
            assert all(len(tasks) == 0 for tasks in cls.tagged_tasks.values())
            return
        key = cls._cache_key(chunk, seq_split_idx, tag)
        tasks = cls.cache.pop(key, [])
        if tasks:
            if tag is None:
                cls.weight_grad_queue[chunk][seq_split_idx].put(tasks)
            else:
                task_key = cls._task_key(chunk, seq_split_idx, tag)
                if task_key in cls.tagged_tasks:
                    raise RuntimeError(
                        f"Duplicate WeightGradStore tagged task "
                        f"(chunk={chunk}, seq_split_idx={seq_split_idx}, tag={tag})"
                    )
                cls.tagged_tasks[task_key] = tasks

    @classmethod
    def pop(cls, chunk=0, seq_split_idx=0, strict=True, tag=None):
        q = cls._ensure_queue(chunk, seq_split_idx)
        if tag is not None:
            task_key = cls._task_key(chunk, seq_split_idx, tag)
            stored_tasks = cls.tagged_tasks.pop(task_key, None)
            if stored_tasks is None:
                if strict and cls.split_bw():
                    raise RuntimeError(
                        f"WeightGradStore pop on missing tagged task "
                        f"(chunk={chunk}, seq_split_idx={seq_split_idx}, tag={tag})"
                    )
                return
            for task, _description in stored_tasks:
                task()
            return

        if q.qsize() == 0:
            if strict and cls.split_bw():
                raise RuntimeError(
                    f"WeightGradStore pop on empty queue "
                    f"(chunk={chunk}, seq_split_idx={seq_split_idx})"
                )
            return
        stored_tasks = q.get()
        for task, _description in stored_tasks:
            task()

    @classmethod
    def pending_count(cls, chunk=0, seq_split_idx=0, tag=None):
        q = cls._ensure_queue(chunk, seq_split_idx)
        if tag is not None:
            key = cls._cache_key(chunk, seq_split_idx, tag)
            task_key = cls._task_key(chunk, seq_split_idx, tag)
            return int(bool(cls.cache.get(key))) + int(bool(cls.tagged_tasks.get(task_key)))

        pending = q.qsize()
        for key, tasks in cls.cache.items():
            cached_chunk, cached_seq_split_idx, _tag = key
            if cached_chunk == chunk and cached_seq_split_idx == seq_split_idx and tasks:
                pending += 1
        for task_key, tasks in cls.tagged_tasks.items():
            task_chunk, task_seq_split_idx, _tag = task_key
            if task_chunk == chunk and task_seq_split_idx == seq_split_idx and tasks:
                pending += 1
        return pending

    @classmethod
    def reset(cls):
        cls.should_split_bw = False
        cls.cache = {}
        cls.tagged_tasks = {}
        cls.weight_grad_queue = None

    @classmethod
    def clear(cls, model=None, chunk=0, seq_split_idx=0, tag=None):
        """Drain all queued and cached tasks for a chunk."""
        q = cls._ensure_queue(chunk, seq_split_idx)
        if tag is None:
            while q.qsize() > 0:
                stored_tasks = q.get()
                for task, _description in stored_tasks:
                    task()

        for key in list(cls.cache):
            cached_chunk, cached_seq_split_idx, cached_tag = key
            if (
                cached_chunk == chunk
                and cached_seq_split_idx == seq_split_idx
                and (tag is None or cached_tag == tag)
            ):
                for task, _description in cls.cache.pop(key, []):
                    task()

        for task_key in list(cls.tagged_tasks):
            task_chunk, task_seq_split_idx, task_tag = task_key
            if (
                task_chunk == chunk
                and task_seq_split_idx == seq_split_idx
                and (tag is None or task_tag == tag)
            ):
                for task, _description in cls.tagged_tasks.pop(task_key, []):
                    task()

        if tag is not None:
            return
