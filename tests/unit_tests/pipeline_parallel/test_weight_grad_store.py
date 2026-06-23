from megatron.core.weight_gradient_store import WeightGradStore


def setup_function():
    WeightGradStore.reset()


def teardown_function():
    WeightGradStore.reset()


def test_put_task_flush_pop_executes_fifo_groups():
    executed = []

    WeightGradStore.enable_split_bw()
    WeightGradStore.put_task(lambda: executed.append("b0-w0"))
    WeightGradStore.put_task(lambda: executed.append("b0-w1"))
    WeightGradStore.flush()
    WeightGradStore.put_task(lambda: executed.append("b1-w0"))
    WeightGradStore.flush()

    assert WeightGradStore.queue_size() == 2

    WeightGradStore.pop()
    assert executed == ["b0-w0", "b0-w1"]

    WeightGradStore.pop()
    assert executed == ["b0-w0", "b0-w1", "b1-w0"]


def test_pop_strict_raises_on_empty_queue():
    WeightGradStore.enable_split_bw()

    try:
        WeightGradStore.pop()
    except RuntimeError as exc:
        assert "empty" in str(exc)
    else:
        raise AssertionError("Expected strict pop to fail on an empty queue")


def test_reset_clears_cache_queue_and_split_state():
    WeightGradStore.enable_split_bw()
    WeightGradStore.put_task(lambda: None)
    WeightGradStore.flush()

    WeightGradStore.reset()

    assert not WeightGradStore.split_bw()
    assert WeightGradStore.queue_size() == 0


def test_clear_drains_queued_and_cached_tasks():
    executed = []

    WeightGradStore.enable_split_bw()
    WeightGradStore.put_task(lambda: executed.append("queued"))
    WeightGradStore.flush()
    WeightGradStore.put_task(lambda: executed.append("cached"))

    WeightGradStore.clear()

    assert executed == ["queued", "cached"]
    assert WeightGradStore.pending_count() == 0
