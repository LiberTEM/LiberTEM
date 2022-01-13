import numpy as np

from libertem.udf.sumsigudf import SumSigUDF

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_sum(lt_ctx, delayed_ctx):
    data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                            num_partitions=2, sig_dims=2)

    sum_result = lt_ctx.run_udf(udf=SumSigUDF(), dataset=dataset)
    sum_delayed = delayed_ctx.run_udf(udf=SumSigUDF(), dataset=dataset)

    naive_result = data.sum(axis=(2, 3))

    assert np.allclose(sum_result['intensity'].data, naive_result)
    assert np.allclose(sum_delayed['intensity'].data, naive_result)
