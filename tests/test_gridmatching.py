import numpy as np

import libertem.analysis.gridmatching as grm

from utils import _fullgrid


def test_consistency(zero, a, b, points, indices):
    coefficients = np.array((a, b))
    result = zero + np.dot(indices, coefficients)
    assert np.allclose(result, points)


def test_calc_coords(zero, a, b, points, indices):
    result = grm.calc_coords(zero, a, b, indices)
    assert np.allclose(result, points)


def test_within_frame():
    points = np.array([
        (0, 0),
        (1, 1),
        (9, 19),
        (19, 9),
        (8, 19),
        (8, 18),
        (9, 18),
        (18, 8)
    ])
    expected = np.array([
        False,
        True,
        False,
        False,
        False,
        True,
        False,
        False
    ])
    result = grm.within_frame(points, r=1, fy=10, fx=20)
    assert np.all(result == expected)


def test_fastmatch(zero, a, b):
    grid = _fullgrid(zero, a, b, 5)
    matcher = grm.Matcher()
    match = matcher.fastmatch(centers=grid, zero=zero, a=a, b=b)
    assert np.allclose(zero, match.zero)
    assert np.allclose(a, match.a)
    assert np.allclose(b, match.b)
    assert len(match) == len(grid)
    assert np.allclose(match.calculated_refineds, grid)


def test_affinematch(zero, a, b):
    grid = _fullgrid(zero, a, b, 5)
    indices = grm.get_indices(grid, zero, a, b)
    matcher = grm.Matcher()
    match = matcher.affinematch(centers=grid, indices=indices)
    assert np.allclose(zero, match.zero)
    assert np.allclose(a, match.a)
    assert np.allclose(b, match.b)
    assert len(match) == len(grid)
    assert np.allclose(match.calculated_refineds, grid)


def test_get_transformation(points):
    points2 = points * np.array((3, 7)) + np.array((-2, -3))
    trans = grm.get_transformation(points, points2)
    target = np.array([
        (3, 0, 0),
        (0, 7, 0),
        (-2, -3, 1)
    ])
    assert np.allclose(trans, target)


def test_do_transformation(points):
    m = np.array([
        (3, 0, 0),
        (0, 7, 0),
        (-2, -3, 1)
    ])
    points2 = grm.do_transformation(m, points)
    target = points * np.array((3, 7)) + np.array((-2, -3))
    assert np.allclose(points2, target)


def test_find_center(points):
    m = np.array([
        (3, 0, 0),
        (0, 7, 0),
        (0, 0, 1)
    ])
    target_center = np.array((2., 3.))
    points2 = grm.do_transformation(m, points, center=target_center)
    trans = grm.get_transformation(points, points2)
    center = grm.find_center(trans)

    assert np.allclose(center, target_center)
