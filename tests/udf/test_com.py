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
    'backend', (None, ) + tuple(com.COMUDF().get_backends())
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

        com_result = lt_ctx.run_udf(udf=com.COMUDF(), dataset=dataset)
        com_delayed = delayed_ctx.run_udf(udf=com.COMUDF(), dataset=dataset)

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

    com_udf = com.COMUDF.with_params(
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

    com_result = lt_ctx.run_udf(udf=com.COMUDF(com_params), dataset=dataset)
    com_analysis = lt_ctx.create_com_analysis(dataset=dataset, **com_dict)
    analysis_result = lt_ctx.run(com_analysis)

    assert_allclose(
        com_result['field'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
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


@pytest.mark.parametrize('repeat', range(10))
def test_com_roi(lt_ctx, repeat):

    data = _mk_random((16, 8, 32, 64))
    dataset = MemoryDataSet(data=data)

    roi = np.random.choice([True, False], dataset.shape.nav)

    com_result = lt_ctx.run_udf(udf=com.COMUDF(), dataset=dataset, roi=roi)

    com_analysis = lt_ctx.create_com_analysis(dataset=dataset)
    analysis_result = lt_ctx.run(com_analysis, roi=roi)

    assert_allclose(
        com_result['field'],
        np.flip(np.moveaxis(analysis_result.field.raw_data, 0, -1), axis=-1),
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
