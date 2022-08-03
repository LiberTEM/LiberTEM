import numpy as np
import pytest

import libertem.masks as m


def test_background_subtraction():
    mask = m.background_subtraction(10, 10, 20, 20, 5, 3)
    assert np.allclose(np.sum(mask), 0)


@pytest.mark.parametrize(
    'antialiased', (True, False)
)
def test_radial_gradient(antialiased):
    template = m.radial_gradient(
        centerX=30, centerY=33,
        imageSizeX=65,
        imageSizeY=66,
        radius=17,
        antialiased=antialiased
    )

    assert template.shape == (66, 65)

    for i in range(17):
        assert np.allclose(template[33, 30+i], i/17)
        assert np.allclose(template[33, 30-i], i/17)
        assert np.allclose(template[33+i, 30], i/17)
        assert np.allclose(template[33-i, 30], i/17)

    for i in range(18, 30):
        assert np.allclose(template[33, 30+i], 0)
        assert np.allclose(template[33, 30-i], 0)
        assert np.allclose(template[33+i, 30], 0)
        assert np.allclose(template[33-i, 30], 0)


def test_sparse_template_multi_stack():
    template = np.ones((2, 3))
    stack = m.sparse_template_multi_stack(
        mask_index=(0, 1, 2),
        offsetY=(13, 14, 15),
        offsetX=(15, 14, 13),
        template=template,
        imageSizeY=32,
        imageSizeX=32
    )
    t1 = np.zeros((32, 32))
    t1[13:15, 15:18] = 1

    t2 = np.zeros((32, 32))
    t2[14:16, 14:17] = 1

    t3 = np.zeros((32, 32))
    t3[15:17, 13:16] = 1

    assert np.allclose(stack[0].todense(), t1)
    assert np.allclose(stack[1].todense(), t2)
    assert np.allclose(stack[2].todense(), t3)


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


def test_rectmask():
    rect = np.ones([3, 3])
    testrect = np.zeros([5, 5])
    rect1 = 1*testrect
    rect1[0:3, 0:3] = rect
    rect2 = 1*testrect
    rect2[0:3, 2:5] = rect
    rect3 = 1*testrect
    rect3[2:5, 0:3] = rect
    rect4 = 1*testrect
    rect4[2:5, 2:5] = rect
    assert np.allclose(m.rectangular(2, 2, 3, 3, 5, 5), rect4)
    assert np.allclose(m.rectangular(2, 2, -3, 3, 5, 5), rect3)
    assert np.allclose(m.rectangular(2, 2, 3, -3, 5, 5), rect2)
    assert np.allclose(m.rectangular(2, 2, -3, -3, 5, 5), rect1)


def test_empty_rectmask():
    assert not np.any(m.rectangular(2, 2, 0, 3, 5, 5))
    assert not np.any(m.rectangular(2, 2, 3, 0, 5, 5))
    assert not np.any(m.rectangular(2, 2, 0, 0, 5, 5))
