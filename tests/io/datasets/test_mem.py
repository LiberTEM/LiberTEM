import numpy as np
import pytest

from libertem.io.dataset.base import TilingScheme, DataSetException
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.stddev import StdDevUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect
from sparseconverter import (
    CUPY_SCIPY_CSC, NUMPY, SCIPY_CSR, SPARSE_COO, SPARSE_GCXS,
    CUPY, get_backend, get_device_class
)
from libertem.udf.base import UDFProtocol

from utils import _mk_random, dataset_correction_verification, set_device_class, roi_as_sparse


d = detect()
has_cupy = d['cudas'] and d['has_cupy']


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(UDFProtocol.BACKEND_ALL)
)
def test_get_macrotile(backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16))
        ds = MemoryDataSet(
            data=data,
            tileshape=(16, 16, 16),
            num_partitions=2,
            array_backends=(backend, ) if backend is not None else None,
        )

        p = next(ds.get_partitions())
        mt = p.get_macrotile()
        assert tuple(mt.shape) == (128, 16, 16)

        if backend is None:
            assert get_backend(data) == get_backend(mt.data)
        else:
            get_backend(mt.data) == backend


@pytest.mark.parametrize(
    'backend', [None, NUMPY, SPARSE_COO, CUPY, CUPY_SCIPY_CSC]
)
@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_correction(lt_ctx, with_roi, backend, as_sparse):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16))
        ds = MemoryDataSet(
            data=data,
            tileshape=(16, 16, 16),
            num_partitions=2,
            array_backends=(backend, ) if backend is not None else None,
        )

        if with_roi:
            roi = np.zeros(ds.shape.nav, dtype=bool)
            roi[:1] = True
            if as_sparse:
                roi = roi_as_sparse(roi)
        else:
            roi = None

        dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@pytest.mark.parametrize(
    'backend', [None, SPARSE_GCXS, CUPY_SCIPY_CSC]
)
def test_positive_sync_offset(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        udf = SumSigUDF()
        data = _mk_random(size=(8, 8, 8, 8))
        sync_offset = 2

        ds_with_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=sync_offset,
            array_backends=(backend, ) if backend is not None else None
        )

        p0 = next(ds_with_offset.get_partitions())
        assert p0._start_frame == 2
        assert p0.slice.origin == (0, 0, 0)

        tileshape = Shape(
            (2,) + tuple(ds_with_offset.shape.sig),
            sig_dims=ds_with_offset.shape.sig.dims
        )
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=tileshape,
            dataset_shape=ds_with_offset.shape,
        )

        for p in ds_with_offset.get_partitions():
            for t in p.get_tiles(tiling_scheme=tiling_scheme, array_backend=backend):
                if backend is None:
                    assert get_backend(t.data) == get_backend(data)
                else:
                    assert get_backend(t.data) == backend

        assert p.slice.origin == (48, 0, 0)
        assert p.slice.shape[0] == 16

        ds_with_no_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=0,
            array_backends=(backend, ) if backend is not None else None
        )
        result = lt_ctx.run_udf(dataset=ds_with_no_offset, udf=udf)
        result = result['intensity'].raw_data[sync_offset:]

        result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
        result_with_offset = result_with_offset['intensity'].raw_data[
            :ds_with_offset._meta.image_count - sync_offset
        ]

        assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    'backend', [NUMPY, SCIPY_CSR, CUPY]
)
def test_negative_sync_offset(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        udf = SumSigUDF()
        data = _mk_random(size=(8, 8, 8, 8))
        sync_offset = -2

        ds_with_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=sync_offset,
            array_backends=(backend, ) if backend is not None else None
        )

        p0 = next(ds_with_offset.get_partitions())
        assert p0._start_frame == -2
        assert p0.slice.origin == (0, 0, 0)

        tileshape = Shape(
            (2,) + tuple(ds_with_offset.shape.sig),
            sig_dims=ds_with_offset.shape.sig.dims
        )
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=tileshape,
            dataset_shape=ds_with_offset.shape,
        )

        for p in ds_with_offset.get_partitions():
            for t in p.get_tiles(tiling_scheme=tiling_scheme, array_backend=backend):
                if backend is None:
                    assert get_backend(t.data) == get_backend(data)
                else:
                    assert get_backend(t.data) == backend

        assert p.slice.origin == (48, 0, 0)
        assert p.slice.shape[0] == 16

        ds_with_no_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=0,
            array_backends=(backend, ) if backend is not None else None
        )
        result = lt_ctx.run_udf(dataset=ds_with_no_offset, udf=udf)
        result = result['intensity'].raw_data[
            :ds_with_no_offset._meta.image_count - abs(sync_offset)
        ]

        result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
        result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

        assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
@pytest.mark.parametrize(
    'backend', [None, SPARSE_GCXS, CUPY_SCIPY_CSC]
)
def test_positive_sync_offset_with_roi(lt_ctx, backend, as_sparse):
    with set_device_class(get_device_class(backend)):
        roi = np.random.choice([False], (8, 8))
        roi[0:1] = True
        if as_sparse:
            roi = roi_as_sparse(roi)

        udf = SumSigUDF()

        data = np.random.randn(8, 8, 8, 8).astype("float32")
        ds = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=0,
            array_backends=(backend, ) if backend is not None else None
        )
        result = lt_ctx.run_udf(dataset=ds, udf=udf)
        result = result['intensity'].raw_data

        sync_offset = 2

        ds_with_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=sync_offset,
            array_backends=(backend, ) if backend is not None else None
        )
        result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
        result_with_offset = result_with_offset['intensity'].raw_data
        roi = np.random.choice([False], (8, 8))
        roi[0:1] = True

        result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
        result_with_offset = result_with_offset['intensity'].raw_data

        assert np.allclose(result[sync_offset:8 + sync_offset], result_with_offset)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
@pytest.mark.parametrize(
    'backend', [None, SCIPY_CSR, CUPY]
)
def test_negative_sync_offset_with_roi(lt_ctx, backend, as_sparse):
    with set_device_class(get_device_class(backend)):
        udf = SumSigUDF()

        data = np.random.randn(8, 8, 8, 8).astype("float32")
        ds = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=0,
            array_backends=(backend, ) if backend is not None else None
        )
        result = lt_ctx.run_udf(dataset=ds, udf=udf)
        result = result['intensity'].raw_data

        sync_offset = -2

        ds_with_offset = MemoryDataSet(
            data=data,
            tileshape=(2, 8, 8),
            num_partitions=4,
            sync_offset=sync_offset,
            array_backends=(backend, ) if backend is not None else None
        )

        roi = np.random.choice([False], (8, 8))
        roi[0:1] = True
        if as_sparse:
            roi = roi_as_sparse(roi)

        result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
        result_with_offset = result_with_offset['intensity'].raw_data

        assert np.allclose(result[:8 + sync_offset], result_with_offset[abs(sync_offset):])


def test_scheme_too_large():
    data = _mk_random(size=(16, 16, 16, 16))
    ds = MemoryDataSet(
        data=data,
        # tileshape=(16, 16, 16),
        num_partitions=2,
    )

    partitions = ds.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, array_backend=NUMPY)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + ds.shape.sig)


@pytest.mark.parametrize(
    'nav_shape', (None, (3, 7), (14*8 + 1, ), (13, 19, 23))
)
@pytest.mark.parametrize(
    'sig_shape', (None, (19, 23), (19 * 23, ))
)
@pytest.mark.parametrize(
    'sync_offset', (0, -3, 7, 13*7, -14*8)
)
@pytest.mark.parametrize(
    'sig_dims', (None, 1, 2)
)
def test_sig_nav_dims_sync(nav_shape, sig_shape, sync_offset, sig_dims, prime_raw_data, lt_ctx):
    if sig_shape is None and sig_dims is None and nav_shape is not None and len(nav_shape) < 2:
        pytest.xfail("This one doesn't reshape nav as the test requires.")
    print('nav_shape', nav_shape)
    print('sig_shape', sig_shape)
    print('sync_offset', sync_offset)
    print('sig_dims', sig_dims)
    throws_value_error = (
        sig_shape is not None and sig_dims is not None and len(sig_shape) != sig_dims
    )
    if nav_shape is not None:
        expected_nav_shape = nav_shape
    elif sig_dims is not None:
        expected_nav_shape = prime_raw_data.shape[:-sig_dims]
    elif sig_shape is not None:
        expected_nav_shape = prime_raw_data.shape[:-len(sig_shape)]
    else:
        expected_nav_shape = prime_raw_data.shape[:-2]

    if sig_shape is not None:
        expected_sig_shape = sig_shape
    elif sig_dims is not None:
        expected_sig_shape = prime_raw_data.shape[-sig_dims:]
    elif nav_shape is not None:
        expected_sig_shape = prime_raw_data.shape[len(nav_shape):]
    else:
        expected_sig_shape = prime_raw_data.shape[-2:]

    if throws_value_error:
        with pytest.raises(ValueError):
            ds = lt_ctx.load(
                'memory',
                data=prime_raw_data,
                nav_shape=nav_shape,
                sig_shape=sig_shape,
                sync_offset=sync_offset,
                sig_dims=sig_dims
            )
    else:
        ds = lt_ctx.load(
            'memory',
            data=prime_raw_data,
            nav_shape=nav_shape,
            sig_shape=sig_shape,
            sync_offset=sync_offset,
            sig_dims=sig_dims
        )
        assert tuple(ds.shape.sig) == expected_sig_shape
        assert tuple(ds.shape.nav) == expected_nav_shape

        # FIXME also check correctness of results
        udf = StdDevUDF()
        lt_ctx.run_udf(dataset=ds, udf=udf)


def test_exception_no_datashape(lt_ctx_fast):
    with pytest.raises(DataSetException):
        lt_ctx_fast.load('memory', tileshape=(5, 6, 7))


def test_num_partitions(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8))
    ds = lt_ctx.load(
        "memory",
        num_partitions=7,
        data=data,
    )
    assert len(list(ds.get_partitions())) == 7
