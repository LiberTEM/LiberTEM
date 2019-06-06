import numpy as np

import libertem.masks as m


def test_background_substraction():
    mask = m.background_substraction(10, 10, 20, 20, 5, 3)
    assert(np.allclose(np.sum(mask), 0))


def test_radial_bins():
    bins = m.radial_bins(35, 37, 80, 80, n_bins=42)
    assert np.allclose(1, bins.sum(axis=0).todense())
    assert bins.shape == (42, 80, 80)


def test_radial_bins_dense():
    bins = m.radial_bins(40, 41, 80, 80, n_bins=2)
    assert np.allclose(1, bins.sum(axis=0))
    assert bins.shape == (2, 80, 80)
