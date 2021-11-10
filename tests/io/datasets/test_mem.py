import numpy as np
import pytest

from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random
from utils import dataset_correction_verification


def test_get_macrotile():
    data = _mk_random(size=(16, 16, 16, 16))
    ds = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )

    p = next(ds.get_partitions())
    mt = p.get_macrotile()
    assert tuple(mt.shape) == (128, 16, 16)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(lt_ctx, with_roi):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_positive_sync_offset(lt_ctx):
    udf = SumSigUDF()
    data = _mk_random(size=(8, 8, 8, 8))
    sync_offset = 2

    ds_with_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=sync_offset,
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
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    ds_with_no_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=0,
    )
    result = lt_ctx.run_udf(dataset=ds_with_no_offset, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(lt_ctx):
    udf = SumSigUDF()
    data = _mk_random(size=(8, 8, 8, 8))
    sync_offset = -2

    ds_with_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=sync_offset,
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
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    ds_with_no_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=0,
    )
    result = lt_ctx.run_udf(dataset=ds_with_no_offset, udf=udf)
    result = result['intensity'].raw_data[:ds_with_no_offset._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_positive_sync_offset_with_roi(lt_ctx):
    udf = SumSigUDF()

    data = np.random.randn(8, 8, 8, 8).astype("float32")
    ds = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=0,
    )
    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = 2

    ds_with_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=sync_offset,
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[sync_offset:8 + sync_offset], result_with_offset)


def test_negative_sync_offset_with_roi(lt_ctx):
    udf = SumSigUDF()

    data = np.random.randn(8, 8, 8, 8).astype("float32")
    ds = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=0,
    )
    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = -2

    ds_with_offset = MemoryDataSet(
        data=data,
        tileshape=(2, 8, 8),
        num_partitions=4,
        sync_offset=sync_offset,
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[:8 + sync_offset], result_with_offset[abs(sync_offset):])


def test_scheme_too_large():
    data = _mk_random(size=(16, 16, 16, 16))
    ds = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
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
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + ds.shape.sig)
