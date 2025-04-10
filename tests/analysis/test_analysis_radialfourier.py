import pytest

import numpy as np
from numpy.testing import assert_allclose

from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.generate import cbed_frame

from utils import _mk_random


@pytest.fixture
def ds_random():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        num_partitions=2,
    )
    return dataset


def test_smoke(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, ri=0, ro=10, n_bins=2, max_order=23
    )
    lt_ctx.run(analysis)


def test_smoke_small(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, ri=0, ro=1, n_bins=1, max_order=23
    )
    results = lt_ctx.run(analysis)
    # access result to actually create result list:
    results['absolute_0_0']


def test_smoke_large(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, n_bins=1, max_order=23
    )
    lt_ctx.run(analysis)


def test_smoke_two(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, cx=0, cy=0, ri=0, ro=2, n_bins=2, max_order=2
    )
    results = lt_ctx.run(analysis)
    # access result to actually create result list:
    results.complex_1_2


def test_smoke_defaults(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random
    )
    lt_ctx.run(analysis)


@pytest.mark.with_numba
def test_sparse(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, use_sparse=True,
    )
    lt_ctx.run(analysis)


def test_sparse_multi(ds_random, lt_ctx):
    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds_random, use_sparse=True,
        n_bins=3, max_order=7,
    )
    lt_ctx.run(analysis)


def test_symmetries(lt_ctx):
    (d1, i1, p1) = cbed_frame(all_equal=True, radius=3, indices=np.array([(1, 0)]))
    (d2, i2, p2) = cbed_frame(all_equal=True, radius=3, indices=np.array([(-1, 0)]))
    (d3, i3, p3) = cbed_frame(all_equal=True, radius=3, indices=np.array([(1, 0), (-1, 0)]))
    (d4, i4, p4) = cbed_frame(
        all_equal=True, radius=3, indices=np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])
    )

    data = np.zeros((2, 2, *d1[0].shape))
    data[0, 0] = d1[0]
    data[0, 1] = d2[0]
    data[1, 0] = d3[0]
    data[1, 1] = d4[0]

    ds = MemoryDataSet(data=data)

    r = np.linalg.norm(p2[0] - p1[0]) / 2
    cy, cx = (p2[0] + p1[0]) / 2

    analysis = lt_ctx.create_radial_fourier_analysis(
        dataset=ds, cy=cy, cx=cx, ri=0, ro=r+4, n_bins=2, max_order=8
    )

    results = lt_ctx.run(analysis)

    c_0_0 = results.complex_0_0.raw_data
    c_1_0 = results.complex_1_0.raw_data

    assert_allclose(np.abs(c_0_0), 0)
    assert_allclose(np.abs(c_1_0), data.sum(axis=(2, 3)))

    c_0_1 = results.complex_0_1.raw_data
    c_1_1 = results.complex_1_1.raw_data

    assert_allclose(np.abs(c_0_1), 0, atol=1e-6, rtol=1e-6)

    assert_allclose(np.abs(c_1_1[1, 0]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.abs(c_1_1[1, 1]), 0, atol=1e-6, rtol=1e-6)

    assert np.all(np.abs(c_1_1[0, 0]) > 0)
    assert np.all(np.abs(c_1_1[0, 1]) > 0)
    assert_allclose(np.angle(c_1_1[0, 0]), np.pi/2)
    assert_allclose(np.angle(c_1_1[0, 1]), -np.pi/2)

    c_0_2 = results.complex_0_2.raw_data
    c_1_2 = results.complex_1_2.raw_data

    assert_allclose(np.abs(c_0_2), 0, atol=1e-6, rtol=1e-6)

    # 2-fold suppressed for 4-fold symmetry
    assert_allclose(np.abs(c_1_2[1, 1]), 0, atol=1e-6, rtol=1e-6)

    assert np.all(np.abs(c_1_2[0, 0]) > 0)
    assert np.all(np.abs(c_1_2[0, 1]) > 0)
    assert np.all(np.abs(c_1_2[1, 0]) > 0)
    # Discontinuity at this point, can be pi or -pi
    assert_allclose(np.abs(np.angle(c_1_2[0, 0])), np.pi)
    assert_allclose(np.abs(np.angle(c_1_2[0, 1])), np.pi)
    assert_allclose(np.abs(np.angle(c_1_2[1, 0])), np.pi)

    c_0_3 = results.complex_0_3.raw_data
    c_1_3 = results.complex_1_3.raw_data

    assert_allclose(np.abs(c_0_3), 0, atol=1e-6, rtol=1e-6)
    # odd harmonics suppressed in 2-fold symmetry
    assert_allclose(np.abs(c_1_3[1]), 0, atol=1e-6, rtol=1e-6)

    assert np.all(np.abs(c_1_3[0, 0]) > 0)
    assert np.all(np.abs(c_1_3[0, 1]) > 0)
    assert_allclose(np.angle(c_1_3[0, 0]), -np.pi/2)
    assert_allclose(np.angle(c_1_3[0, 1]), np.pi/2)

    c_0_4 = results.complex_0_4.raw_data
    c_1_4 = results.complex_1_4.raw_data

    assert_allclose(np.abs(c_0_4), 0, atol=1e-6, rtol=1e-6)

    assert np.all(np.abs(c_1_4) > 0)

    assert_allclose(np.angle(c_1_4[0, 0]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_4[0, 1]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_4[1, 0]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_4[1, 1]), 0, atol=1e-6, rtol=1e-6)

    c_0_5 = results.complex_0_5.raw_data
    c_1_5 = results.complex_1_5.raw_data

    assert_allclose(np.abs(c_0_5), 0, atol=1e-6, rtol=1e-6)

    # odd harmonics suppressed in 2-fold symmetry
    assert_allclose(np.abs(c_1_5[1]), 0, atol=1e-6, rtol=1e-6)

    c_0_7 = results.complex_0_5.raw_data
    c_1_7 = results.complex_1_5.raw_data

    assert_allclose(np.abs(c_0_7), 0, atol=1e-6, rtol=1e-6)

    # odd harmonics suppressed in 2-fold symmetry
    assert_allclose(np.abs(c_1_7[1]), 0, atol=1e-6, rtol=1e-6)

    c_0_8 = results.complex_0_4.raw_data
    c_1_8 = results.complex_1_4.raw_data

    assert_allclose(np.abs(c_0_8), 0, atol=1e-6, rtol=1e-6)

    assert np.all(np.abs(c_1_8) > 0)

    assert_allclose(np.angle(c_1_8[0, 0]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_8[0, 1]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_8[1, 0]), 0, atol=1e-6, rtol=1e-6)
    assert_allclose(np.angle(c_1_8[1, 1]), 0, atol=1e-6, rtol=1e-6)
