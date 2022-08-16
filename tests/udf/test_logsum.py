import numpy as np
import pytest

import libertem.udf.logsum as logsum
from libertem.common.sparse import as_format, NUMPY, SPARSE_COO

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random, set_backend


@pytest.mark.parametrize(
    "format", [NUMPY, SPARSE_COO]
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_logsum(lt_ctx, delayed_ctx, format, backend):
    with set_backend(backend):
        data = _mk_random((16, 8, 32, 64), format=format)

        dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                                num_partitions=2, sig_dims=2)

        logsum_result = logsum.run_logsum(ctx=lt_ctx, dataset=dataset)
        logsum_delayed = logsum.run_logsum(ctx=delayed_ctx, dataset=dataset)

        minima = np.min(data, axis=(2, 3))
        naive_result = as_format(
            np.log(data - minima[..., np.newaxis, np.newaxis] + 1).sum(axis=(0, 1)),
            NUMPY
        )

        assert np.allclose(logsum_result['logsum'].data, naive_result)
        assert np.allclose(logsum_delayed['logsum'].data, naive_result)
