import os
import tempfile

import pytest
import h5py
import numpy as np

from libertem.io.dataset.hdf5 import H5DataSet

from utils import _naive_mask_apply


@pytest.fixture
def hdf5():
    f, tmpfn = tempfile.mkstemp(suffix=".h5")
    os.close(f)
    with h5py.File(tmpfn, "w") as f:
        yield f
    os.unlink(tmpfn)


@pytest.fixture
def hdf5_ds_1(hdf5):
    hdf5.create_dataset("data", data=np.ones((5, 5, 16, 16)))
    return H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16), target_size=512*1024*1024
    )


def test_hdf5_apply_masks_1(lt_ctx, hdf5_ds_1):
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    with hdf5_ds_1.get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
