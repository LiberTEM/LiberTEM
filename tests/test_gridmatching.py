import pytest
import numpy as np

import libertem.analysis.gridmatching as grm

try:
    import libertem.analysis.fullmatch as fm
except ModuleNotFoundError as e:
    fm = None
    missing = e.name


@pytest.fixture
def points():
    return np.array([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, -1),
        (-1, 0),
        (-1, -1)
    ])


@pytest.fixture
def indices():
    return np.array([
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, -1),
        (-1, -1)
    ])


@pytest.fixture
def zero():
    return np.array([0, 0])


@pytest.fixture
def a():
    return np.array([0, 1])


@pytest.fixture
def b():
    return np.array([1, 0])


def _fullgrid(zero, a, b, index, skip_zero=False):
    i, j = np.mgrid[-index:index, -index:index]
    indices = np.concatenate(np.array((i, j)).T)
    if skip_zero:
        select = (np.not_equal(indices[:, 0], 0) + np.not_equal(indices[:, 1], 0))
        indices = indices[select]
    return grm.calc_coords(zero, a, b, indices)


def test_consistency(zero, a, b, points, indices):
    coefficients = np.array((a, b))
    result = zero + np.dot(indices, coefficients)
    assert(np.allclose(result, points))


def test_calc_coords(zero, a, b, points, indices):
    result = grm.calc_coords(zero, a, b, indices)
    assert(np.allclose(result, points))


def test_polar():
    data = np.array([
        (0, 1),
        (1, 0),
        (-2, 0)
    ])
    expected = np.array([
        (1, 0),
        (1, np.pi/2),
        (2, -np.pi/2)
    ])

    result = grm.make_polar(data)
    assert(np.allclose(expected, result))


def test_conversion(points):
    assert(np.allclose(points, grm.make_cartesian(grm.make_polar(points))))


def test_sizefilter():
    polars = np.array([
        (0.1, 2),
        (1, 0.3),
        (2, 0.5)
    ])

    assert(np.allclose(grm.size_filter(polars, 0, np.float('inf')), polars))
    assert(len(grm.size_filter(polars, 3, np.float('inf'))) == 0)
    assert(np.allclose(grm.size_filter(polars, 0.9, 1.1), polars[1]))


def test_angle_ckeck():
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
    check = grm.angle_check(polar_1, polar_2, np.pi/5)
    assert(not check[:-1].any())
    assert(check[-1])
    check_2 = grm.angle_check(polar_2, polar_1, np.pi/5)
    assert(not check_2[:-1].any())
    assert(check_2[-1])


def test_fastmatch(zero, a, b):
    grid = _fullgrid(zero, a, b, 5)
    correlation_result = grm.CorrelationResult(
        centers=grid,
        refineds=grid,
        peak_values=np.ones(len(grid)),
        peak_elevations=np.ones(len(grid))
    )
    match = grm.Match.fastmatch(
        correlation_result=correlation_result, zero=zero, a=a, b=b)
    assert(np.allclose(zero, match.zero))
    assert(np.allclose(a, match.a))
    assert(np.allclose(b, match.b))
    assert(len(match) == len(grid))
    assert(np.allclose(match.calculated_refineds, grid))


def test_fullmatch_three_residual(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    aa = np.array([1.27, 1.2])
    bb = np.array([1.27, -1.2])

    grid_1 = _fullgrid(zero, a, b, 7)
    grid_2 = _fullgrid(zero, aa, bb, 4, skip_zero=True)

    random = np.array([
        (0.3, 0.5),
        (-0.3, 12.5),
        (-0.3, -17.5),
    ])

    grid = np.vstack((grid_1, grid_2, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
    }

    (matches, unmatched, weak) = fm.full_match(grid, zero=zero, parameters=parameters)

    assert(len(matches) == 2)

    assert(len(unmatched) == len(random))
    assert(len(weak) == 0)

    match1 = matches[0]

    assert(np.allclose(zero, match1.zero))
    assert(np.allclose(a, match1.a) or np.allclose(b, match1.a)
           or np.allclose(-a, match1.a) or np.allclose(-b, match1.a))
    assert(np.allclose(a, match1.b) or np.allclose(b, match1.b)
           or np.allclose(-a, match1.b) or np.allclose(-b, match1.b))
    assert(len(match1) == len(grid_1))
    assert(np.allclose(match1.calculated_refineds, grid_1))

    match2 = matches[1]

    assert(np.allclose(zero, match2.zero))
    assert(np.allclose(aa, match2.a) or np.allclose(bb, match2.a)
           or np.allclose(-aa, match2.a) or np.allclose(-bb, match2.a))
    assert(np.allclose(aa, match2.b) or np.allclose(bb, match2.b)
           or np.allclose(-aa, match2.b) or np.allclose(-bb, match2.b))
    # We always match the zero point for each lattice
    assert(len(match2) == len(grid_2) + 1)
    # We filter out the zero point, which is added in the matching routine to each matching cycle
    skip_zero = np.array([
        any(match2.indices[i] != np.array((0, 0))) for i in range(len(match2))
    ], dtype=np.bool)
    # We calculate by hand because the built-in method can't skip the zero point
    assert(np.allclose(grm.calc_coords(
        match2.zero, match2.a, match2.b, match2.indices[skip_zero]), grid_2))


def test_fullmatch_one_residual(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    aa = np.array([1.27, 1.2])
    bb = np.array([1.27, -1.2])

    grid_1 = _fullgrid(zero, a, b, 7)
    grid_2 = _fullgrid(zero, aa, bb, 4, skip_zero=True)

    random = np.array([
        (0.3, 0.5),
    ])

    grid = np.vstack((grid_1, grid_2, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
    }
    (matches, unmatched, weak) = fm.full_match(grid, zero=zero, parameters=parameters)

    assert(len(matches) == 2)

    assert(len(unmatched) == len(random))
    assert(len(weak) == 0)


def test_fullmatch_no_residual(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    aa = np.array([1.27, 1.2])
    bb = np.array([1.27, -1.2])

    grid_1 = _fullgrid(zero, a, b, 7)
    grid_2 = _fullgrid(zero, aa, bb, 4, skip_zero=True)

    grid = np.vstack((grid_1, grid_2))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
    }
    (matches, unmatched, weak) = fm.full_match(grid, zero=zero, parameters=parameters)

    assert(len(matches) == 2)

    assert(len(unmatched) == 0)
    assert(len(weak) == 0)


def test_fullmatch_weak(zero, a, b):
    if fm is None:
        pytest.skip("Failed to load optional dependency %s." % missing)
    aa = np.array([1.27, 1.2])
    bb = np.array([1.27, -1.2])

    grid_1 = _fullgrid(zero, a, b, 7)
    grid_2 = _fullgrid(zero, aa, bb, 4, skip_zero=True)
    
    random = np.array([
        (0.3, 0.5),
    ])
    
    grid = np.vstack((grid_1, grid_2, random))

    parameters = {
        'min_delta': 0.3,
        'max_delta': 3,
    }

    values = np.ones(len(grid))
    elevations = np.ones(len(grid))

    # The default minimum weight is 0.1
    elevations[0] = 0.01

    correlation_result = grm.CorrelationResult(
        centers=grid,
        refineds=grid,
        peak_values=values,
        peak_elevations=elevations
    )

    (matches, unmatched, weak) = fm.FullMatch.full_match(
        correlation_result=correlation_result, zero=zero, parameters=parameters)

    assert(len(matches) == 2)

    assert(len(unmatched) == len(random))
    assert(len(weak) == 1)

    # We have kicked out the first point with a low weight
    # so we make sure it is gone from the match although it
    # would be at the right position
    assert(len(matches[0]) == len(grid_1) - 1)
