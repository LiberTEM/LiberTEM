import sys
import time
from itertools import chain
import gc

import numpy as np
import pytest

from libertem.udf.masks import ApplyMasksUDF


# based on https://code.activestate.com/recipes/577504/
def total_size(o):
    def dict_handler(d):
        return chain.from_iterable(d.items())

    def object_handler(o):
        if hasattr(o, '__dict__'):
            return dict_handler(o.__dict__)
        else:
            return tuple()

    all_handlers = {
        tuple: iter,
        list: iter,
        # deque: iter, Triggers RuntimeError with Dask client
        dict: dict_handler,
        set: iter,
        frozenset: iter,
        object: object_handler,
    }
    seen = set()                      # track which object id's have already been seen
    default_size = sys.getsizeof(0)       # estimate sizeof object without __sizeof__

    def sizeof(o):
        if id(o) in seen:       # do not double count the same object
            return 0
        seen.add(id(o))
        s = sys.getsizeof(o, default_size)

        for typ, handler in all_handlers.items():
            if isinstance(o, typ):
                s += sum(map(sizeof, handler(o)))
                break
        return s

    return sizeof(o)


def worker_memory(client):
    info = client.scheduler_info()
    memory = 0
    for name, w_info in info['workers'].items():
        memory += w_info['metrics']['memory']
    return memory


@pytest.mark.slow
@pytest.mark.parametrize(
    'ctx_select', ('dask', 'inline')
)
def test_executor_memleak(local_cluster_ctx, lt_ctx_fast, default_raw, ctx_select):
    if ctx_select == 'dask':
        ctx = local_cluster_ctx
        rounds = 5

        def get_worker_mem(ctx):
            return worker_memory(ctx.executor.client)

    elif ctx_select == 'inline':
        ctx = lt_ctx_fast
        rounds = 1

        def get_worker_mem(ctx):
            return 0

    mask_count = 8*1014*1024 // np.prod(default_raw.shape.sig)

    mask_shape = (mask_count, *default_raw.shape.sig)
    masks = np.zeros(mask_shape)

    # Intentionally "bad" factory function: make it large by closing over masks
    def mask_factory():
        return masks

    udf = ApplyMasksUDF(
        mask_factories=mask_factory,
        mask_count=mask_count,
        mask_dtype=masks.dtype,
        use_torch=False
    )

    # warm-up
    for _ in range(2):
        for _ in ctx.run_udf_iter(dataset=default_raw, udf=udf):
            pass

    cumulative_worker_delta = 0
    cumulative_executor_delta = 0

    for round in range(rounds):
        gc.collect()

        # Allow to settle
        time.sleep(1)

        ctx.executor.run_each_worker(gc.collect)

        executor_size_before = total_size(ctx)
        worker_mem_before = get_worker_mem(ctx)

        executor_size_during = None

        for res in ctx.run_udf_iter(dataset=default_raw, udf=udf):
            if executor_size_during is None:
                executor_size_during = total_size(ctx)
                worker_mem_during = get_worker_mem(ctx)

        gc.collect()

        # Allow to settle
        time.sleep(1)

        ctx.executor.run_each_worker(gc.collect)

        # Allow to settle
        time.sleep(1)

        executor_size_after = total_size(ctx)
        worker_mem_after = get_worker_mem(ctx)

        active_use = worker_mem_during - worker_mem_before

        # Memory use does increase slowly. Just make sure it is not caused by keeping
        # a big array around
        worker_delta = worker_mem_after - worker_mem_before
        executor_delta = executor_size_after - executor_size_before

        print(f"Round {round}")
        print(f"Memory use during UDF run: {active_use}.")
        print(f"Memory increase worker: {worker_delta}.")
        print(f"Memory increase executor: {executor_delta}.")

        cumulative_worker_delta += worker_delta
        cumulative_executor_delta += executor_delta

    worker_count = len(ctx.executor.get_available_workers())

    assert cumulative_worker_delta/rounds/worker_count < sys.getsizeof(masks) * 0.1
    assert cumulative_executor_delta/rounds < sys.getsizeof(masks) * 0.1
