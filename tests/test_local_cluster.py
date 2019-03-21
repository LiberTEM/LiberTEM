import numpy as np
import pytest

from libertem import api
from utils import _naive_mask_apply, _mk_random


@pytest.mark.functional
def test_start_local(hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with api.Context() as ctx:
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
