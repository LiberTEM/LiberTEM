import numpy as np
import pytest

import libertem.udf.logsum as logsum
from sparseconverter import (
    SPARSE_BACKENDS, for_backend, NUMPY, SPARSE_COO, get_device_class
)

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random, set_device_class


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(logsum.LogsumUDF().get_backends())
)
def test_logsum(lt_ctx, delayed_ctx, backend):
    with set_device_class(get_device_class(backend)):
        if backend in SPARSE_BACKENDS:
            data = _mk_random((16, 8, 32, 64), array_backend=SPARSE_COO)
        else:
            data = _mk_random((16, 8, 32, 64), array_backend=NUMPY)

        dataset = MemoryDataSet(
            data=data, tileshape=(8, 32, 64),
            num_partitions=2,
            sig_dims=2,
            array_backends=(backend, ) if backend is not None else None
        )

        logsum_result = logsum.run_logsum(ctx=lt_ctx, dataset=dataset)
        logsum_delayed = logsum.run_logsum(ctx=delayed_ctx, dataset=dataset)

        minima = np.min(data, axis=(2, 3))
        naive_result = for_backend(
            np.log(data - minima[..., np.newaxis, np.newaxis] + 1).sum(axis=(0, 1)),
            NUMPY
        )

        assert np.allclose(logsum_result['logsum'].data, naive_result)
        assert np.allclose(logsum_delayed['logsum'].data, naive_result)
