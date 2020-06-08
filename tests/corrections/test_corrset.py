import numpy as np
import pytest
import sparse

from libertem.corrections import CorrectionSet
from libertem.udf.sum import SumUDF


@pytest.mark.parametrize("gain,dark", [
    (np.zeros((128, 128)), np.ones((128, 128))),
    (np.zeros((128, 128)), None),
])
def test_correction_set_zero_gain(lt_ctx, default_raw, gain, dark):
    udf = SumUDF()

    corr = CorrectionSet(dark=dark, gain=gain)
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=udf,
        corrections=corr
    )
    assert np.allclose(res['intensity'], 0)


@pytest.mark.parametrize("gain,dark", [
    (np.ones((128, 128)), np.ones((128, 128))),
    (None, np.ones((128, 128))),
])
def test_correction_set_dark_one(lt_ctx, default_raw, default_raw_data, gain, dark):
    udf = SumUDF()

    corr = CorrectionSet(dark=dark, gain=gain)
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=udf,
        corrections=corr
    )
    assert np.allclose(res['intensity'], np.sum(default_raw_data - 1, axis=(0, 1)))


def test_patch_pixels(lt_ctx, default_raw, default_raw_data):
    udf = SumUDF()

    # test with empty excluded_pixels array
    corr = CorrectionSet(excluded_pixels=np.array([
        (), ()
    ]).astype(np.int64), gain=np.ones((128, 128)))
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=udf,
        corrections=corr
    )
    assert np.allclose(res['intensity'], np.sum(default_raw_data, axis=(0, 1)))


def test_patch_pixels_only_excluded_pixels(lt_ctx, default_raw, default_raw_data):
    udf = SumUDF()
    excluded_pixels = sparse.COO(
        np.zeros((128, 128))
    )
    corr = CorrectionSet(excluded_pixels=excluded_pixels)
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=udf,
        corrections=corr
    )
    assert np.allclose(res['intensity'], np.sum(default_raw_data, axis=(0, 1)))
