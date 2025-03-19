import pytest
import numpy as np


@pytest.mark.parametrize(
    "slice_", [
        np.s_[0],
        np.s_[5, 5],
        np.s_[0:5],
        np.s_[0:5, 2:7],
    ]
)
def test_roi_helper(default_raw, slice_):
    ref_roi = np.zeros(default_raw.shape.nav, dtype=bool)
    ref_roi[slice_] = True
    assert np.allclose(
        default_raw.roi[slice_],
        ref_roi,
    )
