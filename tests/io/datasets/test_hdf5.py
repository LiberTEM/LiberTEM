import os
import json
import threading
import itertools
from unittest import mock
import sys
from typing import Optional

import cloudpickle
import numpy as np
from numpy.testing import assert_allclose
import pytest
import h5py

from libertem.io.dataset.hdf5 import H5DataSet
from libertem.analysis.sum import SumAnalysis
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.auto import AutoUDF
from libertem.io.dataset.base import TilingScheme, DataSetException
from libertem.common import Shape
from libertem.common.math import flat_nonzero
from libertem.io.dataset.base import Negotiator
from libertem.udf import UDF

from utils import _naive_mask_apply, _mk_random, PixelsumUDF
from utils import dataset_correction_verification, roi_as_sparse


def test_hdf5_apply_masks_1(lt_ctx, hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


def test_hdf5_3d_apply_masks(lt_ctx, hdf5_ds_3d):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_3d.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data.reshape((1, 17, 16, 16)))
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_3d, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


def test_hdf5_5d_apply_masks(lt_ctx, hdf5_ds_5d):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_5d.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data.reshape((1, 135, 16, 16))).reshape((3, 5, 9))
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_5d, factories=[lambda: mask]
    )
    results = lt_ctx.run(analysis)

    print(results.mask_0.raw_data.shape, expected.shape)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(hdf5_ds_1, lt_ctx, with_roi):
    ds = hdf5_ds_1

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_read_1(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_read_2(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_read_3(lt_ctx, random_hdf5):
    # try with smaller partitions:
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data",
        target_size=4096
    )
    ds = ds.initialize(lt_ctx.executor)
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            print(t.tile_slice)


def test_pickle_ds(lt_ctx, hdf5_ds_1):
    pickled = cloudpickle.dumps(hdf5_ds_1)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is not None

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024


def test_cloudpickle(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", target_size=512*1024*1024
    )

    pickled = cloudpickle.dumps(ds)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is None
    assert loaded._shape is None
    repr(loaded)

    ds = ds.initialize(lt_ctx.executor)

    pickled = cloudpickle.dumps(ds)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is not None
    assert loaded._shape is not None
    loaded.shape
    loaded.dtype
    repr(loaded)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_1(hdf5, lt_ctx, as_sparse):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    p = next(ds.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    if as_sparse:
        roi = roi_as_sparse(roi)
    for tile in p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi):
        print("tile:", tile)
        tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.shape.nav.size == 1
    assert tuple(tiles[0].tile_slice.shape.sig) == (16, 16)
    assert tiles[0].tile_slice.origin == (0, 0, 0)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_3(hdf5, lt_ctx, as_sparse):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
        target_size=12800*2,
    )
    ds = ds.initialize(lt_ctx.executor)
    roi = np.zeros(ds.shape.flatten_nav().nav, dtype=bool)
    roi[24] = 1

    tileshape = Shape(
        (16,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    tiles = []
    if as_sparse:
        roi = roi_as_sparse(roi)
    for p in ds.get_partitions():
        for tile in p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi):
            print("tile:", tile)
            tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.shape.nav.size == 1
    assert tuple(tiles[0].tile_slice.shape.sig) == (16, 16)
    assert tiles[0].tile_slice.origin == (0, 0, 0)
    assert np.allclose(tiles[0].data, hdf5['data'][4, 4])


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_4(hdf5, lt_ctx, as_sparse):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
        target_size=12800*2,
    )
    ds = ds.initialize(lt_ctx.executor)
    roi = np.random.choice(size=ds.shape.flatten_nav().nav, a=[True, False])

    sum_udf = lt_ctx.create_sum_analysis(dataset=ds)
    if as_sparse:
        roi = roi_as_sparse(roi)
    sumres = lt_ctx.run(sum_udf, roi=roi)['intensity']

    assert np.allclose(
        sumres,
        np.sum(hdf5['data'][:].reshape(25, 16, 16)[flat_nonzero(roi), ...], axis=0)
    )


def test_roi_5(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
        target_size=12800*2,
    )
    ds = ds.initialize(lt_ctx.executor)
    roi = np.random.choice(size=ds.shape.flatten_nav().nav, a=[True, False])

    udf = SumSigUDF()
    sumres = lt_ctx.run_udf(dataset=ds, udf=udf, roi=roi)['intensity']

    assert np.allclose(
        sumres.raw_data,
        np.sum(hdf5['data'][:][roi.reshape(5, 5), ...], axis=(1, 2))
    )


def test_pick(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    assert len(ds.shape) == 4
    print(ds.shape)
    pick = lt_ctx.create_pick_analysis(dataset=ds, x=2, y=3)
    lt_ctx.run(pick)


def test_diags(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    print(ds.diagnostics)


def test_check_valid(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    assert ds.check_valid()


def test_timeout_1(hdf5, lt_ctx):
    with mock.patch('h5py.File.visititems', side_effect=TimeoutError("too slow")):
        params = H5DataSet.detect_params(hdf5.filename, executor=lt_ctx.executor)["parameters"]
        assert list(params.keys()) == ["path"]

        ds = H5DataSet(
            path=hdf5.filename, ds_path="data",
        )
        ds = ds.initialize(lt_ctx.executor)
        diags = ds.diagnostics
        print(diags)


def test_timeout_2(hdf5, lt_ctx):
    print(threading.enumerate())
    with mock.patch('libertem.io.dataset.hdf5.current_time', side_effect=[1, 60]):
        params = H5DataSet.detect_params(hdf5.filename, executor=lt_ctx.executor)["parameters"]
        assert list(params.keys()) == ["path"]

    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)

    print(threading.enumerate())
    with mock.patch('libertem.io.dataset.hdf5.current_time', side_effect=[30, 90]):
        diags = ds.diagnostics
        print(diags)


@pytest.mark.parametrize("mnp", [None, 1, 4])
def test_roi_2(random_hdf5, lt_ctx, mnp):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data",
        min_num_partitions=mnp,
    )
    ds = ds.initialize(lt_ctx.executor)

    roi = {
        "shape": "disk",
        "cx": 2,
        "cy": 2,
        "r": 1,
    }
    analysis = SumAnalysis(dataset=ds, parameters={
        "roi": roi,
    })

    print(analysis.get_roi())

    results = lt_ctx.run(analysis)

    # let's draw a circle!
    mask = np.full((5, 5), False)
    mask[1, 2] = True
    mask[2, 1:4] = True
    mask[3, 2] = True

    print(mask)

    assert mask.shape == (5, 5)
    assert mask.dtype == bool

    reader = ds.get_reader()
    with reader.get_h5ds() as h5ds:
        data = np.array(h5ds)

        # applying the mask flattens the first two dimensions, so we
        # only sum over axis 0 here:
        expected = data[mask, ...].sum(axis=(0,))

        assert expected.shape == (16, 16)
        assert results.intensity.raw_data.shape == (16, 16)

        # is not equal to results without mask:
        assert not np.allclose(results.intensity.raw_data, data.sum(axis=(0, 1)))
        # ... but rather like `expected`:
        assert np.allclose(results.intensity.raw_data, expected)


def test_cache_key_json_serializable(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    json.dumps(ds.get_cache_key())


def test_no_tileshape(lt_ctx, hdf5_3d):
    ds = lt_ctx.load('HDF5', path=hdf5_3d.filename, ds_path='/data')

    udf = PixelsumUDF()
    lt_ctx.run_udf(udf=udf, dataset=ds)


def test_scheme_idx(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    p = next(ds.get_partitions())

    sig_shape = tuple(ds.shape.sig)
    tileshape = Shape(
        (16,) + sig_shape[:-1] + (sig_shape[-1] // 2,),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)

    for tile, expected_idx in zip(tiles, itertools.cycle([0, 1])):
        print(tile.scheme_idx, tile.tile_slice)
        assert tile.scheme_idx == expected_idx


def test_hdf5_with_2d_shape(lt_ctx, hdf5_2d):
    # regression test for error message:
    # sync_offset should be in (0, 0), which is (-image_count, image_count)
    with pytest.raises(DataSetException) as e:
        hdf5_ds_2d = lt_ctx.load("hdf5", path=hdf5_2d.filename, ds_path="data")

        # these lines are currently not reached yet, but will be if we decide to support 2D HDF5
        assert hdf5_ds_2d.shape.to_tuple() == (1, 16, 16)  # navigation shape is extended
        udf = PixelsumUDF()
        lt_ctx.run_udf(udf=udf, dataset=hdf5_ds_2d)

    assert e.match("2D HDF5 files are currently not supported")


@pytest.mark.parametrize('chunks', [
    (1, 3, 16, 16),
    (1, 6, 16, 16),
    (1, 4, 16, 16),
    (1, 16, 16, 16),
])
def test_chunked(lt_ctx, tmpdir_factory, chunks):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'hdf5-test-chunked.h5')
    data = _mk_random((16, 16, 16, 16), dtype=np.float32)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, chunks=chunks)

    ds = lt_ctx.load("hdf5", path=filename)
    udf = PixelsumUDF()
    res = lt_ctx.run_udf(udf=udf, dataset=ds)['pixelsum']
    assert np.allclose(
        res,
        np.sum(data, axis=(2, 3))
    )


@pytest.fixture(scope='module')
def shared_random_data():
    return _mk_random(size=(16, 16, 256, 256), dtype='float32')


@pytest.mark.parametrize('udf', [
    SumSigUDF(),
    AutoUDF(f=lambda frame: frame.sum()),
])
@pytest.mark.parametrize('chunks', [
    (3, 3, 32, 32),
    (3, 6, 32, 32),
    (3, 4, 32, 32),
    (1, 4, 32, 32),
    (1, 16, 32, 32),

    (3, 3, 256, 256),
    (3, 6, 256, 256),
    (3, 4, 256, 256),
    (1, 4, 256, 256),
    (1, 16, 256, 256),

    (3, 3, 128, 256),
    (3, 6, 128, 256),
    (3, 4, 128, 256),
    (1, 4, 128, 256),
    (1, 16, 128, 256),

    (3, 3, 32, 128),
    (3, 6, 32, 128),
    (3, 4, 32, 128),
    (1, 4, 32, 128),
    (1, 16, 32, 128),
])
def test_chunked_weird(lt_ctx, tmpdir_factory, chunks, udf, shared_random_data):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'weirdly-chunked-256-256.h5')
    data = shared_random_data

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, chunks=chunks)

    ds = lt_ctx.load("hdf5", path=filename)

    base_shape = ds.get_base_shape(roi=None)
    print(base_shape)

    res = lt_ctx.run_udf(dataset=ds, udf=udf)
    assert len(res) == 1
    res = next(iter(res.values()))
    assert_allclose(
        res,
        np.sum(data, axis=(2, 3))
    )

    os.unlink(filename)


@pytest.mark.parametrize('in_dtype', [
    np.float32,
    np.float64,
    np.uint16,
])
@pytest.mark.parametrize('read_dtype', [
    np.float32,
    np.float64,
    np.uint16,
])
@pytest.mark.parametrize('use_roi', [
    True, False
])
def test_hdf5_result_dtype(lt_ctx, tmpdir_factory, in_dtype, read_dtype, use_roi):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'result-dtype-checks.h5')
    data = _mk_random((2, 2, 4, 4), dtype=in_dtype)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)

    ds = lt_ctx.load("hdf5", path=filename)

    if use_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool).reshape((-1,))
        roi[0] = 1
    else:
        roi = None
    udfs = [SumSigUDF()]  # need to have at least one UDF
    p = next(ds.get_partitions())
    neg = Negotiator()
    tiling_scheme = neg.get_scheme(
        udfs=udfs,
        dataset=ds,
        approx_partition_shape=p.shape,
        read_dtype=read_dtype,
        roi=roi,
        corrections=None,
    )
    tile = next(p.get_tiles(tiling_scheme=tiling_scheme, roi=roi, dest_dtype=read_dtype))
    assert tile.dtype == np.dtype(read_dtype)


class UDFWithLargeDepth(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": 128,
            "total_size": UDF.TILE_SIZE_BEST_FIT,
        }


def test_hdf5_tileshape_negotation(lt_ctx, tmpdir_factory):
    # try to hit the third case in _get_subslices:
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'tileshape-neg-test.h5')
    data = _mk_random((4, 100, 256, 256), dtype=np.uint16)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, chunks=(2, 32, 32, 32))

    ds = lt_ctx.load("hdf5", path=filename)

    udfs = [UDFWithLargeDepth()]
    p = next(ds.get_partitions())
    neg = Negotiator()
    tiling_scheme = neg.get_scheme(
        udfs=udfs,
        dataset=ds,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None,
        corrections=None,
    )
    assert len(tiling_scheme) > 1
    next(p.get_tiles(tiling_scheme=tiling_scheme, roi=None, dest_dtype=np.float32))


def test_scheme_too_large(hdf5_ds_1):
    partitions = hdf5_ds_1.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(hdf5_ds_1.shape.sig),
        sig_dims=hdf5_ds_1.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=hdf5_ds_1.shape,
    )

    # tile shape is clamped to partition shape.
    # in case of hdf5, it is even smaller than the
    # partition, as the depth from the negotiation
    # is overridden:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert t.tile_slice.shape[0] <= hdf5_ds_1.shape[0]


def test_hdf5_macrotile(lt_ctx, tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, 'macrotile-1.h5')
    data = _mk_random((128, 128, 4, 4), dtype=np.float32)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)

    ds = lt_ctx.load("hdf5", path=filename)
    ds.set_num_cores(4)

    partitions = ds.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile()
    assert m0.tile_slice.origin == (0, 0, 0)
    assert m0.tile_slice.shape == p0.shape

    p1 = next(partitions)
    m1 = p1.get_macrotile()
    assert m1.tile_slice.origin == (p0.shape[0], 0, 0)
    assert m1.tile_slice.shape == p1.shape


def test_hdf5_macrotile_roi(lt_ctx, hdf5_ds_1):
    roi = np.random.choice(size=hdf5_ds_1.shape.flatten_nav().nav, a=[True, False])
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = data.reshape(hdf5_ds_1.shape.flatten_nav())[roi]
    partitions = hdf5_ds_1.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile(roi=roi)
    assert_allclose(
        m0.data,
        expected
    )


def test_hdf5_macrotile_empty_roi(lt_ctx, hdf5_ds_1):
    roi = np.zeros(hdf5_ds_1.shape.flatten_nav().nav, dtype=bool)
    partitions = hdf5_ds_1.get_partitions()
    p0 = next(partitions)
    m0 = p0.get_macrotile(roi=roi)
    assert m0.shape == (0, 16, 16)
    assert_allclose(
        m0.data,
        0,
    )


def test_hdf5_filters(local_cluster_ctx, lt_ctx, tmpdir_factory):
    # Make sure it is either preloaded already
    # or not installed
    if 'hdf5plugin' not in sys.modules:
        with pytest.raises(ImportError):
            import hdf5plugin
    else:
        import hdf5plugin
        datadir = tmpdir_factory.mktemp('filtered')
        filename = os.path.join(datadir, 'filtered.h5')
        with h5py.File(filename, "w") as f:
            f.create_dataset(
                "data",
                data=np.ones((16, 16, 16, 16)),
                **hdf5plugin.LZ4()
            )
        for ctx in (local_cluster_ctx, lt_ctx):
            ds = ctx.load('HDF5', path=filename)
            res = ctx.run_udf(dataset=ds, udf=SumSigUDF())
            assert np.allclose(res['intensity'].raw_data, np.prod(ds.shape.sig))


def _place_results(
    data: np.ndarray,
    nav_shape: tuple[int],
    sync_offset: int = 0,
    roi: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Places ravelled data into an array of nav_shape
    following the sync_offset, roi and reshaping conventions
    of LiberTEM.

    roi.shape must equal nav_shape
    data.size can be smaller or larger than nav_shape.size
    the data will be clipped or filled with zeros appropriately
    """
    data = data.ravel()
    output_size = np.prod(nav_shape)
    output = np.zeros((output_size,), dtype=data.dtype)
    if sync_offset <= 0:
        nel = min(output[abs(sync_offset):].size, data.size)
        output[abs(sync_offset): abs(sync_offset) + nel] = data[:nel]
    else:
        nel = min(data[sync_offset:].size, output.size)
        output[:nel] = data[sync_offset:sync_offset + nel]
    output = output.reshape(nav_shape)
    if roi is not None:
        output[np.logical_not(roi)] = np.nan
    return output


@pytest.mark.parametrize(
    'file_nav_shape, nav_chunks, nav_shape,', [
        ((9,), (3,), (3, 3,)),
        ((4, 4), (2, 2), (2, 8)),
        ((4, 4, 2), (2, 4, 2), (32,)),
        ((16,), (4,), (3, 3,)),
        ((9, 9), (3, 9), (4, 4,)),
        ((9, 9), (3, 9), (16, 16)),
        ((125,), (10,), (5, 5)),
        ((125,), (10,), (100, 100)),
    ])
@pytest.mark.parametrize('roi', [
    None, True,
])
@pytest.mark.parametrize('sync_offset', [
    0, -5, 5,
])
def test_nav_reshape(lt_ctx, tmpdir_factory, file_nav_shape,
                     nav_chunks, nav_shape, roi, sync_offset):
    sig_shape = (16, 16)
    sig_chunks = (16, 16)
    sig_dims = len(sig_shape)
    frame_size_bytes = np.prod(sig_shape) * np.dtype(np.float32).itemsize
    # Choose a deliberately small partition size so that some
    # partition + sync_offset combinations mean some parts are unneeded
    target_size_bytes = 4 * frame_size_bytes

    datadir = tmpdir_factory.mktemp('data')
    file_shape = file_nav_shape + sig_shape
    filename = os.path.join(datadir, f'data_{"_".join(str(s) for s in file_shape)}.h5')
    data = np.random.uniform(size=file_shape).astype(np.float32)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data, chunks=nav_chunks + sig_chunks)

    if roi:
        roi = np.random.choice([True, False], size=nav_shape).astype(bool)

    ds = lt_ctx.load(
        "hdf5",
        path=filename,
        nav_shape=nav_shape,
        sig_dims=sig_dims,
        target_size=target_size_bytes,
        sync_offset=sync_offset,
    )
    res = lt_ctx.run_udf(dataset=ds, udf=SumSigUDF(), roi=roi)

    full_sum_result = data.sum(axis=tuple(range(-1, -sig_dims - 1, -1))).ravel()
    sum_result = _place_results(
        full_sum_result,
        nav_shape,
        sync_offset=sync_offset,
        roi=roi,
    )

    assert np.allclose(
        res['intensity'].data,
        sum_result,
        equal_nan=roi is not None,
    )

    os.unlink(filename)


def test_sig_reshape_unsupported(lt_ctx, tmpdir_factory):
    file_shape = (8, 16, 16)
    sig_shape = (3, 3)
    sig_dims = 2

    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, f'data_{"_".join(str(s) for s in file_shape)}.h5')
    data = np.random.uniform(size=file_shape).astype(np.float32)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)

    with pytest.raises(DataSetException):
        lt_ctx.load(
            "hdf5",
            path=filename,
            sig_shape=sig_shape,
            sig_dims=sig_dims,
        )

    os.unlink(filename)


def test_sync_offset_beyond_ds(lt_ctx, tmpdir_factory):
    file_shape = (8, 16, 16)
    sync_offset = 10

    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, f'data_{"_".join(str(s) for s in file_shape)}.h5')
    data = np.random.uniform(size=file_shape).astype(np.float32)

    with h5py.File(filename, "w") as f:
        f.create_dataset("data", data=data)

    with pytest.raises(DataSetException):
        lt_ctx.load(
            "hdf5",
            path=filename,
            sync_offset=sync_offset,
        )

    os.unlink(filename)
