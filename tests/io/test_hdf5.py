import numpy as np

from utils import _naive_mask_apply, _mk_random


def test_hdf5_apply_masks_1(lt_ctx, hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
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
