import numpy as np
import sparse
import pytest

from libertem.udf.sum import SumUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.common.sparse import as_format, NUMPY

from utils import _mk_random


@pytest.mark.parametrize(
    "sparse_input", [True, False]
)
def test_sum(lt_ctx, delayed_ctx, sparse_input):
    if sparse_input:
        data = sparse.random((16, 8, 32, 64))
    else:
        data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                            num_partitions=2, sig_dims=2)

    sum_result = lt_ctx.run_udf(udf=SumUDF(), dataset=dataset)
    sum_delayed = delayed_ctx.run_udf(udf=SumUDF(), dataset=dataset)

    naive_result = as_format(data.sum(axis=(0, 1)), NUMPY)

    assert np.allclose(sum_result['intensity'].data, naive_result)
    assert np.allclose(sum_delayed['intensity'].data, naive_result)
