import pytest
import numpy as np

try:
    import libertem.analysis.fullmatch as fm
except ModuleNotFoundError as e:
    fm = None
    missing = e.name

from utils import _fullgrid


def test_sizefilter():
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    polars = np.array([
        (0.1, 2),
        (1, 0.3),
        (2, 0.5)
    ])

    assert np.allclose(fm.size_filter(polars, 0, np.inf), polars)
    assert len(fm.size_filter(polars, 3, np.inf)) == 0
    assert np.allclose(fm.size_filter(polars, 0.9, 1.1), polars[1])


def test_angle_ckeck():
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    polar_1 = np.array([
        (1, 0),
        (17, np.pi/10),
        (8, np.pi*9/10),
        (1, np.pi),
        (0.1, 2*np.pi),
        (1, 2*np.pi*19/20),
        (1, -np.pi/10),
        (1, -np.pi*9/10),
        (1, np.pi*11/10),
        (1, np.pi/2)
    ])
    polar_2 = np.repeat(np.array([(1, 0)]), len(polar_1), axis=0)
    check = fm.angle_check(polar_1, polar_2, np.pi/5)
    assert not check[:-1].any()
    assert check[-1]
    check_2 = fm.angle_check(polar_2, polar_1, np.pi/5)
    assert not check_2[:-1].any()
    assert check_2[-1]


def test_fullmatch_two_residual(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)

    grid_1 = _fullgrid(zero, a, b, 5)

    random = np.array([
        (0.3, 0.4),
        (-0.33, 12.5),
    ])

    grid = np.vstack((grid_1, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
        'tolerance': 0.1
    }
    matcher = fm.FullMatcher(**parameters)
    (matches, unmatched, weak) = matcher.full_match(grid, zero=zero)

    print(matches[0])

    assert len(matches) == 1

    assert len(unmatched) == len(random)
    assert len(weak) == 0

    match1 = matches[0]

    assert np.allclose(zero, match1.zero)
    assert (np.allclose(a, match1.a) or np.allclose(b, match1.a)
           or np.allclose(-a, match1.a) or np.allclose(-b, match1.a))
    assert (np.allclose(a, match1.b) or np.allclose(b, match1.b)
           or np.allclose(-a, match1.b) or np.allclose(-b, match1.b))
    assert len(match1) == len(grid_1)
    assert np.allclose(match1.calculated_refineds, grid_1)


def test_fullmatch_weak(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)

    grid_1 = _fullgrid(zero, a, b, 7)

    random = np.array([
        (0.3, 0.5),
    ])

    grid = np.vstack((grid_1, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
        'tolerance': 0.05,
        'min_weight': 0.1
    }

    values = np.ones(len(grid))
    elevations = np.ones(len(grid))

    # The  minimum weight is 0.1, so this should be ignored
    elevations[0] = 0.01

    matcher = fm.FullMatcher(**parameters)
    (matches, unmatched, weak) = matcher.full_match(
        centers=grid, refineds=grid, peak_values=values, peak_elevations=elevations, zero=zero
    )

    assert len(matches) == 1

    assert len(unmatched) == len(random)
    assert len(weak) == 1

    # We have kicked out the first point with a low weight
    # so we make sure it is gone from the match although it
    # would be at the right position
    assert len(matches[0]) == len(grid_1) - 1


def test_fullmatch_cand(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)

    grid_1 = _fullgrid(zero, a, b, 7)

    random = np.array([
        (0.3, 0.5),
    ])

    grid = np.vstack((grid_1, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
        'tolerance': 0.05
    }

    matcher = fm.FullMatcher(**parameters)
    (matches, unmatched, weak) = matcher.full_match(
        centers=grid, zero=zero, cand=[a, b, np.array((23, 42))]
    )

    # Make sure the odd candidate didn't catch anything
    assert len(matches) == 1
    assert len(unmatched) == len(random)
