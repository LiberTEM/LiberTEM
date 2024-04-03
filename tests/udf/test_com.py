import numpy as np
import pytest
from numpy.testing import assert_allclose
import random

import libertem.udf.com as com
from sparseconverter import (
    SPARSE_BACKENDS, NUMPY, SPARSE_COO, get_device_class,
    CUPY_SCIPY_CSC
)

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random, set_device_class


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(com.CoMUDF().get_backends())
)
def test_com(lt_ctx, delayed_ctx, backend):
    with set_device_class(get_device_class(backend)):
        if backend in SPARSE_BACKENDS:
            data = _mk_random((16, 8, 32, 64), array_backend=SPARSE_COO)
        else:
            data = _mk_random((16, 8, 32, 64), array_backend=NUMPY)

        dataset = MemoryDataSet(
            data=data, tileshape=(8, 32, 64),
            num_partitions=2,
            sig_dims=2,
            array_backends=(backend, ) if backend is not None else None
        )

        com_result = lt_ctx.run_udf(udf=com.CoMUDF(), dataset=dataset)
        com_delayed = delayed_ctx.run_udf(udf=com.CoMUDF(), dataset=dataset)

        com_analysis = lt_ctx.create_com_analysis(dataset=dataset)
        analysis_result = lt_ctx.run(com_analysis)

        if backend is CUPY_SCIPY_CSC:
            pytest.xfail(
                'For unknown reasons the dot product seems to be unstable '
                'for cupyx.scipy.csc_matrix'
            )

        assert_allclose(
            com_result['field'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
        )
        assert_allclose(
            com_result['field_y'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 0],
        )
        assert_allclose(
            com_result['field_x'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 1],
        )
        assert_allclose(
            com_result['field'],
            np.stack((com_result['field_y'], com_result['field_x']), axis=-1),
        )
        assert_allclose(
            com_result['magnitude'],
            analysis_result.magnitude.raw_data,
        )
        assert_allclose(
            com_result['divergence'],
            analysis_result.divergence.raw_data,
        )
        assert_allclose(
            com_result['curl'],
            analysis_result.curl.raw_data,
        )

        assert_allclose(
            com_delayed['field'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
        )
        assert_allclose(
            com_delayed['field_y'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 0],
        )
        assert_allclose(
            com_delayed['field_x'],
            np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 1],
        )
        assert_allclose(
            com_delayed['field'],
            np.stack((com_delayed['field_y'], com_delayed['field_x']), axis=-1),
        )
        assert_allclose(
            com_delayed['magnitude'],
            analysis_result.magnitude.raw_data,
        )
        assert_allclose(
            com_delayed['divergence'],
            analysis_result.divergence.raw_data,
        )
        assert_allclose(
            com_delayed['curl'],
            analysis_result.curl.raw_data,
        )


@pytest.mark.parametrize('repeat', range(10))
def test_com_params(lt_ctx, repeat):
    data = _mk_random((16, 8, 32, 64))
    dataset = MemoryDataSet(data=data)

    cy = random.choice((None, 2.1, 17))
    cx = random.choice((None, 3.7, 23))
    ro = random.choice((float('inf'), 10.2, 19))
    ri = random.choice((0., 3.1, 9))
    if np.isinf(ro):
        # Non-zero ri not supported by analysis API
        ri = 0.
    scan_rotation = random.choice((0., 13.3, 233))
    flip_y = random.choice((True, False))

    com_udf = com.CoMUDF.with_params(
        cy=cy,
        cx=cx,
        r=ro,
        ri=ri,
        scan_rotation=scan_rotation,
        flip_y=flip_y,
    )
    com_params = com_udf.params.com_params

    com_dict = {}
    com_dict['cy'] = com_params.cy
    com_dict['cx'] = com_params.cx
    com_dict['scan_rotation'] = com_params.scan_rotation
    com_dict['flip_y'] = com_params.flip_y
    com_dict['mask_radius'] = com_params.r
    com_dict['mask_radius_inner'] = com_params.ri

    # Test the analysis default None parameters
    if np.isinf(com_dict['mask_radius']) and random.choice((True, False)):
        com_dict['mask_radius'] = None

    if (com_dict['mask_radius'] is None
       or (np.isclose(com_dict['mask_radius_inner'], 0.) and random.choice((True, False)))):
        com_dict['mask_radius_inner'] = None

    com_result = lt_ctx.run_udf(udf=com.CoMUDF(com_params), dataset=dataset)
    com_analysis = lt_ctx.create_com_analysis(dataset=dataset, **com_dict)
    analysis_result = lt_ctx.run(com_analysis)

    assert_allclose(
        com_result['field'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
    )
    assert_allclose(
        com_result['field_y'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 0],
    )
    assert_allclose(
        com_result['field_x'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 1],
    )
    assert_allclose(
        com_result['field'],
        np.stack((com_result['field_y'], com_result['field_x']), axis=-1),
    )
    assert_allclose(
        com_result['magnitude'],
        analysis_result.magnitude.raw_data,
    )
    assert_allclose(
        com_result['divergence'],
        analysis_result.divergence.raw_data,
    )
    assert_allclose(
        com_result['curl'],
        analysis_result.curl.raw_data,
    )


def test_com_invalid_annulus_params():
    with pytest.raises(ValueError):
        com.CoMUDF.with_params(
            r=5.,
            ri=10.,
        )


def test_invalid_sig_shape(lt_ctx):
    ds = lt_ctx.load(
        'memory',
        data=np.zeros((2, 2, 4, 4, 4)),
        sig_dims=3,
        num_partitions=1,  # avoid a warning
    )
    udf = com.CoMUDF()
    with pytest.raises(ValueError):
        lt_ctx.run_udf(ds, udf)


@pytest.mark.parametrize('dims', (1, 3))
def test_invalid_nav_shape(lt_ctx, dims):
    nav_shape = (2,) * dims
    ds = lt_ctx.load(
        'memory',
        data=np.zeros(nav_shape + (4, 4)),
        sig_dims=2,
        num_partitions=1,  # avoid a warning
    )
    udf = com.CoMUDF()
    with pytest.raises(ValueError):
        lt_ctx.run_udf(ds, udf)


@pytest.mark.parametrize('repeat', range(10))
def test_com_roi(lt_ctx, repeat):

    data = _mk_random((16, 8, 32, 64))
    dataset = MemoryDataSet(data=data)

    roi = np.random.choice([True, False], dataset.shape.nav)

    com_result = lt_ctx.run_udf(udf=com.CoMUDF(), dataset=dataset, roi=roi)

    com_analysis = lt_ctx.create_com_analysis(dataset=dataset)
    analysis_result = lt_ctx.run(com_analysis, roi=roi)

    assert_allclose(
        com_result['field'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
    )
    assert_allclose(
        com_result['field_y'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 0],
    )
    assert_allclose(
        com_result['field_x'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1)[..., 1],
    )
    assert_allclose(
        com_result['field'],
        np.stack((com_result['field_y'], com_result['field_x']), axis=-1),
    )
    assert_allclose(
        com_result['magnitude'],
        analysis_result.magnitude.raw_data,
    )

    # The Analysis may return non-NaN values outside of the ROI
    # if all neighbors are within the ROI because of the way how np.gradient works.
    # It can directly return the result for the entire nav buffer, i.e. also set
    # values outside of the ROI.
    # The UDF cannot set values outside of the ROI to exactly replicate the Analysis
    # because of the way how get_results() and UDF result buffers for kind='nav' work.
    # Instead of equality for the entire result, we test for equality within the ROI
    # and only NaN outside of the ROI for the UDF result.
    inverted_roi = np.invert(roi)
    assert_allclose(
        com_result['divergence'].data[roi],
        analysis_result.divergence.raw_data[roi],
    )
    assert np.all(np.isnan(com_result['divergence'].data[inverted_roi]))
    assert_allclose(
        com_result['curl'].data[roi],
        analysis_result.curl.raw_data[roi],
    )
    assert np.all(np.isnan(com_result['curl'].data[inverted_roi]))


@pytest.mark.parametrize(
    'use_roi', (True, False)
)
@pytest.mark.parametrize(
    'regression', (-1, 0, 1, [[0, 0], [0, 0], [0, 0]])
)
def test_com_regression_neutral(lt_ctx, use_roi, regression):
    data = np.zeros((23, 42, 23, 42))
    for yy in range(23):
        for xx in range(42):
            data[yy, xx, 3, 3] = 1
    # Make sure the first partition extends in nav y direction,
    # so num_partitions smaller than y nav shape
    ds = lt_ctx.load('memory', data=data, num_partitions=17)
    if use_roi:
        roi = np.random.choice([True, False], ds.shape.nav)
    else:
        roi = None

    udf = com.CoMUDF.with_params(cy=3, cx=3, regression=regression)
    for iter_res in lt_ctx.run_udf_iter(dataset=ds, udf=udf, roi=roi):
        res = iter_res.buffers[0]
        assert_allclose(res['regression'], 0, atol=1e-13)
        assert_allclose(res['field'].raw_data, 0, atol=1e-13)
        assert_allclose(res['field_y'].raw_data, 0, atol=1e-13)
        assert_allclose(res['field_x'].raw_data, 0, atol=1e-13)


@pytest.mark.parametrize(
    'use_roi', (True, False)
)
@pytest.mark.parametrize(
    'regression', (0, 1, [[-2, 1], [0, 0], [0, 0]])
)
def test_com_regression_offset(lt_ctx, use_roi, regression):
    data = np.zeros((23, 42, 23, 42))
    for yy in range(23):
        for xx in range(42):
            data[yy, xx, 1, 4] = 1
    ds = lt_ctx.load('memory', data=data, num_partitions=17)
    if use_roi:
        roi = np.random.choice([True, False], ds.shape.nav)
    else:
        roi = None

    udf = com.CoMUDF.with_params(cy=3, cx=3, regression=regression)
    res = lt_ctx.run_udf(dataset=ds, udf=udf, roi=roi)
    for iter_res in lt_ctx.run_udf_iter(dataset=ds, udf=udf, roi=roi):
        res = iter_res.buffers[0]
        assert_allclose(res['regression'].data, [
            [-2, 1],  # constant offset in y and x direction
            [0, 0],
            [0, 0]
        ], atol=1e-13)
        assert_allclose(res['field'].raw_data, 0, atol=1e-13)
        assert_allclose(res['field_y'].raw_data, 0, atol=1e-13)
        assert_allclose(res['field_x'].raw_data, 0, atol=1e-13)


@pytest.mark.parametrize(
    'use_roi', (True, False)
)
@pytest.mark.parametrize(
    'regression', (1, [[-2, 42], [2, 0], [0, -1]])
)
def test_com_regression_linear(lt_ctx, use_roi, regression):
    data = np.zeros((23, 42, 48, 46))
    for yy in range(23):
        for xx in range(42):
            data[yy, xx, 2*yy+1, 45-xx] = 1
    ds = lt_ctx.load('memory', data=data, num_partitions=7)
    if use_roi:
        roi = np.random.choice([True, False], ds.shape.nav)
    else:
        roi = None

    udf = com.CoMUDF.with_params(cy=3, cx=3, regression=regression)
    for iter_res in lt_ctx.run_udf_iter(dataset=ds, udf=udf, roi=roi):
        res = iter_res.buffers[0]
        damage = iter_res.damage
        assert_allclose(res['regression'].data, [[-2, 42], [2, 0], [0, -1]], atol=1e-13)
        # The linear gradient is corrected in valid data
        assert_allclose(res['field'].data[damage], 0, atol=1e-12)
        # Regression only applied to valid data, other parts remain at their
        # default value, 0 in this case
        assert_allclose(res['field'].raw_data[~damage.raw_data], 0, atol=1e-12)
        assert_allclose(res['field_y'].data[damage], 0, atol=1e-12)
        assert_allclose(res['field_x'].data[damage], 0, atol=1e-12)


@pytest.mark.parametrize(
    'use_roi', (True, False)
)
@pytest.mark.parametrize(
    'regression', (1, [[-2, 1], [2, 1], [3, 4]])
)
def test_com_regression_linear_2(lt_ctx, use_roi, regression):
    data = np.zeros((5, 8, 64, 64))
    for yy in range(5):
        for xx in range(8):
            # dy/dy: 2, dx/dy: 1
            # dy/dx: 3, dx/dx: 4
            data[yy, xx, 1+2*yy+3*xx, 4+yy+4*xx] = 1
    ds = lt_ctx.load('memory', data=data, num_partitions=3)
    if use_roi:
        roi = np.ones(ds.shape.nav, dtype=bool)
        roi[2:4, 5:7] = False
    else:
        roi = None

    udf = com.CoMUDF.with_params(cy=3, cx=3, regression=regression)
    for iter_res in lt_ctx.run_udf_iter(dataset=ds, udf=udf, roi=roi):
        res = iter_res.buffers[0]
        assert_allclose(res['regression'].data, [[-2, 1], [2, 1], [3, 4]], atol=1e-13)
        assert_allclose(res['field'].raw_data, 0, atol=1e-12)
        assert_allclose(res['field_y'].raw_data, 0, atol=1e-12)
        assert_allclose(res['field_x'].raw_data, 0, atol=1e-12)


def test_invalid_regression_val(lt_ctx, npy_8x8x8x8_ds):
    udf = com.CoMUDF.with_params(regression=5)
    with pytest.raises(ValueError):
        lt_ctx.run_udf(npy_8x8x8x8_ds, udf)
