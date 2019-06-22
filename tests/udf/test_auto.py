import functools

import numpy as np

from libertem.udf import run_auto

from utils import MemoryDataSet, _mk_random


def test_auto(lt_ctx):
    data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                            num_partitions=2, sig_dims=2)

    logsum_result = run_auto(ctx=lt_ctx, dataset=dataset, f=functools.partial(np.sum, axis=-1))

    naive_result = data.sum(axis=-1)

    assert np.allclose(logsum_result['result'].data, naive_result)
