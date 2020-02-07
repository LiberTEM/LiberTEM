import warnings

import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_job_deprecation(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    def ones():
        return np.ones((16, 16))

    with warnings.catch_warnings(record=True) as w:
        lt_ctx.create_pick_job(dataset=dataset, origin=(7, 8))
        lt_ctx.create_mask_job(dataset=dataset, factories=[ones])
        assert len(w) == 4
        assert issubclass(w[0].category, FutureWarning)
        assert issubclass(w[1].category, DeprecationWarning)
        assert issubclass(w[2].category, FutureWarning)
        assert issubclass(w[3].category, DeprecationWarning)
