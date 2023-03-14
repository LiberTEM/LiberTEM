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


def test_com_params(lt_ctx):
    for i in range(10):
        data = _mk_random((16, 8, 32, 64))
        dataset = MemoryDataSet(data=data)

        cy = random.choice((None, 2.1, 17))
        cx = random.choice((None, 3.7, 23))
        r = random.choice((None, 10.2, 19))
        ri = random.choice((None, 3.1, 9))
        scan_rotation = random.choice((None, 13.3, 233))
        flip_y = random.choice((None, True, False))

        com_params = com.COMParams(
            cy=cy,
            cx=cx,
            r=r,
            ri=ri,
            scan_rotation=scan_rotation,
            flip_y=flip_y,
        )
        com_dict = {
            p: getattr(com_params, p) for p in (
                'cy', 'cx', 'scan_rotation', 'flip_y'
            ) if getattr(com_params, p) is not None
        }

        if com_params.r is not None:
            com_dict['mask_radius'] = com_params.r
        if com_params.ri is not None:
            com_dict['mask_radius_inner'] = com_params.ri

        if com_params.r is None and com_params.ri is not None:
            continue  # not supported by COMAnalysis

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


def test_com_roi(lt_ctx):
    for i in range(10):
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
        for y in range(com_result['divergence'].data.shape[0]):
            for x in range(com_result['divergence'].data.shape[1]):
                res = com_result['divergence'].data[y, x]
                ref = analysis_result.divergence.raw_data[y, x]
                if not np.allclose(ref, res, equal_nan=True):
                    print(y, x, ref, res)

        assert_allclose(
            com_result['divergence'],
            analysis_result.divergence.raw_data,
        )
        assert_allclose(
            com_result['curl'],
            analysis_result.curl.raw_data,
        )
