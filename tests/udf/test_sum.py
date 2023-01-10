import numpy as np
import pytest

from libertem.udf.sum import SumUDF
from libertem.io.dataset.memory import MemoryDataSet
from sparseconverter import (
    for_backend, NUMPY, get_device_class, SPARSE_BACKENDS, SPARSE_COO
)
from utils import _mk_random, set_device_class


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(SumUDF().get_backends())
)
def test_sum(lt_ctx, delayed_ctx, backend):
    with set_device_class(get_device_class(backend)):
        if backend in SPARSE_BACKENDS:
            data = _mk_random((16, 8, 32, 64), array_backend=SPARSE_COO)
        else:
            data = _mk_random((16, 8, 32, 64), array_backend=NUMPY)

        dataset = MemoryDataSet(
            data=data,
            tileshape=(8, 17, 23),
            num_partitions=2,
            sig_dims=2,
            array_backends=(backend, ) if backend is not None else None
        )

        sum_result = lt_ctx.run_udf(udf=SumUDF(), dataset=dataset)
        sum_delayed = delayed_ctx.run_udf(udf=SumUDF(), dataset=dataset)

        naive_result = for_backend(data.sum(axis=(0, 1)), NUMPY)

        assert np.allclose(sum_result['intensity'].data, naive_result)
        assert np.allclose(sum_delayed['intensity'].data, naive_result)
