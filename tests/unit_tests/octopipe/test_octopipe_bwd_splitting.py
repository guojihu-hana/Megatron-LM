from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel import schedules
from megatron.core.weight_gradient_store import WeightGradStore


def setup_function():
    WeightGradStore.reset()


def teardown_function():
    WeightGradStore.reset()


def test_prepare_octopipe_bwd_splitting_requires_w_workload():
    config = SimpleNamespace(octopipe_bwd_splitting=True)

    with pytest.raises(ValueError, match="requires at least one 'w' workload"):
        schedules._prepare_octopipe_bwd_splitting(
            config,
            workloads=[{"op": "comp", "type": "b", "mid": 0, "sid": 0}],
        )


def test_prepare_octopipe_bwd_splitting_enables_store_when_w_present():
    config = SimpleNamespace(octopipe_bwd_splitting=True)

    enabled = schedules._prepare_octopipe_bwd_splitting(
        config,
        workloads=[
            {"op": "comp", "type": "b", "mid": 0, "sid": 0},
            {"op": "comp", "type": "w", "mid": 0, "sid": 0},
        ],
    )

    assert enabled
    assert WeightGradStore.split_bw()


def test_register_octopipe_wgrad_task_flushes_chunk_backward_dw():
    calls = []
    model_chunk = SimpleNamespace(backward_dw=lambda: calls.append("backward_dw"))
    WeightGradStore.enable_split_bw()

    schedules._register_octopipe_wgrad_task(model_chunk, chunk=0, tag=(3, 7))

    assert WeightGradStore.queue_size(chunk=0) == 0
    assert WeightGradStore.pending_count(chunk=0, tag=(3, 7)) == 1
    WeightGradStore.pop(chunk=0, tag=(3, 7))
    assert calls == ["backward_dw"]


def test_register_octopipe_wgrad_task_traverses_chunk_submodules_without_duplicates():
    calls = []

    class Leaf(torch.nn.Module):
        def __init__(self, name):
            super().__init__()
            self.name = name

        def backward_dw(self):
            calls.append(self.name)

    class Parent(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.child = Leaf("child")

        def backward_dw(self):
            calls.append("parent")

    class Chunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = Parent()
            self.second = Leaf("second")

    WeightGradStore.enable_split_bw()
    schedules._register_octopipe_wgrad_task(Chunk(), chunk=0, tag=(0, 0))

    WeightGradStore.pop(chunk=0, tag=(0, 0))

    assert calls == ["second", "parent"]
