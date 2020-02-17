import pickle
import json
from unittest import mock
import threading

import cloudpickle
import numpy as np
import pytest

from libertem.io.dataset.hdf5 import H5DataSet
from libertem.analysis.sum import SumAnalysis
from libertem.udf.sumsigudf import SumSigUDF

from utils import _naive_mask_apply, _mk_random


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_hdf5_apply_masks_1(lt_ctx, hdf5_ds_1, TYPE):
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)
    analysis = lt_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )
    analysis.TYPE = TYPE
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


def test_read_1(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    for p in ds.get_partitions():
        for t in p.get_tiles():
            print(t.tile_slice)


def test_read_2(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 3, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    for p in ds.get_partitions():
        for t in p.get_tiles():
            print(t.tile_slice)


def test_read_3(lt_ctx, random_hdf5):
    # try with smaller partitions:
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 2, 16, 16),
        target_size=4096
    )
    ds = ds.initialize(lt_ctx.executor)
    for p in ds.get_partitions():
        for t in p.get_tiles():
            print(t.tile_slice)


def test_pickle_ds(lt_ctx, hdf5_ds_1):
    pickled = pickle.dumps(hdf5_ds_1)
    loaded = pickle.loads(pickled)

    assert loaded._dtype is not None

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024


def test_cloudpickle(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16), target_size=512*1024*1024
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


def test_roi_1(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    p = next(ds.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    for tile in p.get_tiles(dest_dtype="float32", roi=roi):
        print("tile:", tile)
        tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.shape.nav.size == 1
    assert tuple(tiles[0].tile_slice.shape.sig) == (16, 16)
    assert tiles[0].tile_slice.origin == (0, 0, 0)


def test_roi_3(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16),
        target_size=12800*2,
    )
    ds = ds.initialize(lt_ctx.executor)
    roi = np.zeros(ds.shape.flatten_nav().nav, dtype=bool)
    roi[24] = 1

    tiles = []
    for p in ds.get_partitions():
        for tile in p.get_tiles(dest_dtype="float32", roi=roi):
            print("tile:", tile)
            tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.shape.nav.size == 1
    assert tuple(tiles[0].tile_slice.shape.sig) == (16, 16)
    assert tiles[0].tile_slice.origin == (0, 0, 0)
    assert np.allclose(tiles[0].data, hdf5['data'][4, 4])


def test_roi_4(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16),
        target_size=12800*2,
    )
    ds = ds.initialize(lt_ctx.executor)
    roi = np.random.choice(size=ds.shape.flatten_nav().nav, a=[True, False])

    sum_udf = lt_ctx.create_sum_analysis(dataset=ds)
    sumres = lt_ctx.run(sum_udf, roi=roi)['intensity']

    assert np.allclose(
        sumres,
        np.sum(hdf5['data'][:].reshape(25, 16, 16)[roi, ...], axis=0)
    )


def test_roi_5(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16),
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


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick(hdf5, lt_ctx, TYPE):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 3, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    assert len(ds.shape) == 4
    print(ds.shape)
    pick = lt_ctx.create_pick_analysis(dataset=ds, x=2, y=3)
    pick.TYPE = TYPE
    lt_ctx.run(pick)


def test_diags(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    print(ds.diagnostics)


def test_check_valid(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    assert ds.check_valid()


def test_timeout_1(hdf5, lt_ctx):
    with mock.patch('h5py.File.visititems', side_effect=TimeoutError("too slow")):
        params = H5DataSet.detect_params(hdf5.filename, executor=lt_ctx.executor)
        assert list(params.keys()) == ["path"]

        ds = H5DataSet(
            path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
        )
        ds = ds.initialize(lt_ctx.executor)
        diags = ds.diagnostics
        print(diags)


def test_timeout_2(hdf5, lt_ctx):
    print(threading.enumerate())
    with mock.patch('libertem.io.dataset.hdf5.current_time', side_effect=[1, 30]):
        params = H5DataSet.detect_params(hdf5.filename, executor=lt_ctx.executor)
        assert list(params.keys()) == ["path"]

    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)

    print(threading.enumerate())
    with mock.patch('libertem.io.dataset.hdf5.current_time', side_effect=[30, 60]):
        diags = ds.diagnostics
        print(diags)


@pytest.mark.parametrize("mnp", [None, 1, 4])
def test_roi_2(random_hdf5, lt_ctx, mnp):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16),
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
    assert mask.dtype == np.bool

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
        path=hdf5.filename, ds_path="data", tileshape=(1, 3, 16, 16)
    )
    ds = ds.initialize(lt_ctx.executor)
    json.dumps(ds.get_cache_key())


def test_auto_tileshape(chunked_hdf5, lt_ctx):
    ds = H5DataSet(
        path=chunked_hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(lt_ctx.executor)
    p = next(ds.get_partitions())
    t = next(p.get_tiles(dest_dtype="float32", target_size=4*1024))
    assert tuple(p._get_tileshape("float32", 4*1024)) == (1, 4, 16, 16)
    assert tuple(t.tile_slice.shape) == (4, 16, 16)
