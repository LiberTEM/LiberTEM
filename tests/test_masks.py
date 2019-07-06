import numpy as np

import libertem.masks as m


def test_background_subtraction():
    mask = m.background_subtraction(10, 10, 20, 20, 5, 3)
    assert(np.allclose(np.sum(mask), 0))


def test_radial_bins():
    bins = m.radial_bins(35, 37, 80, 80, n_bins=42)
    assert np.allclose(1, bins.sum(axis=0).todense())
    assert bins.shape == (42, 80, 80)


def test_radial_bins_dense():
    bins = m.radial_bins(40, 41, 80, 80, n_bins=2)
    assert np.allclose(1, bins.sum(axis=0))
    assert bins.shape == (2, 80, 80)


def test_oval_radial_background_balance():
    radius, angle = m.polar_map(
        centerX=10, centerY=10,
        imageSizeX=21, imageSizeY=21,
        stretchY=1.5,
        angle=np.pi/4
    )
    template = m.radial_gradient_background_subtraction(
        r=radius,
        r0=5,
        r_outer=7
    )
    template = m.balance(template)
    outside = radius > 7
    inside = (radius < 5) * (radius > 0)

    assert np.allclose(template.sum(), 0)
    assert np.allclose(template[outside], 0)
    assert np.all(template[inside] > 0)

    assert np.allclose(template[0, :], 0)
    assert np.allclose(template[20, :], 0)
    assert np.allclose(template[:, 0], 0)
    assert np.allclose(template[:, 20], 0)

    diag = int(np.sqrt(5))

    # Confirm stretched gradient in 45 diagonal
    assert template[10+diag, 10+diag] > 0
    assert template[10-diag, 10+diag] < template[10+diag, 10+diag]


def test_oval_radial_background_symmetry():
    radius, angle = m.polar_map(
        centerX=10, centerY=10,
        imageSizeX=21, imageSizeY=21,
        stretchY=1.5,
        angle=0.
    )

    radius2, angle2 = m.polar_map(
        centerX=10, centerY=10,
        imageSizeX=21, imageSizeY=21,
        stretchY=1.5,
        angle=np.pi
    )

    radius3, angle3 = m.polar_map(
        centerX=10, centerY=10,
        imageSizeX=21, imageSizeY=21,
        stretchY=1.5,
        angle=np.pi/4
    )

    radius4, angle4 = m.polar_map(
        centerX=10, centerY=10,
        imageSizeX=21, imageSizeY=21,
        stretchY=1.5,
        angle=-np.pi/4
    )

    assert np.allclose(radius, radius2)
    assert np.allclose(radius3, np.flip(radius4, axis=1))
