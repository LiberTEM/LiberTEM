import pickle
from unittest import mock

import cloudpickle
import numpy as np
import pytest

from libertem.io.dataset.hdf5 import H5DataSet, unravel_nav
from libertem.common import Slice, Shape
from libertem.analysis.sum import SumAnalysis

from utils import _naive_mask_apply, _mk_random


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


def test_read_1(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize()
    for p in ds.get_partitions():
        for t in p.get_tiles():
            print(t.tile_slice)


def test_read_2(lt_ctx, hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 3, 16, 16)
    )
    ds = ds.initialize()
    for p in ds.get_partitions():
        for t in p.get_tiles():
            print(t.tile_slice)


def test_read_3(lt_ctx, random_hdf5):
    # try with smaller partitions:
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 2, 16, 16),
        target_size=4096
    )
    ds = ds.initialize()
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

    ds.initialize()

    pickled = cloudpickle.dumps(ds)
    loaded = cloudpickle.loads(pickled)

    assert loaded._dtype is not None
    assert loaded._shape is not None
    loaded.shape
    loaded.dtype
    repr(loaded)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 1 * 1024


def test_flatten_roundtrip():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((2, 16, 16, 16), sig_dims=2)
    )
    sflat = Slice(
        origin=(0, 0, 0),
        shape=Shape((32, 16, 16), sig_dims=2)
    )
    assert s.flatten_nav((16, 16, 16, 16)) == sflat
    assert unravel_nav(sflat, (16, 16, 16, 16)) == s


def test_flatten_roundtrip_2():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((2, 16, 32, 64), sig_dims=2)
    )
    sflat = Slice(
        origin=(0, 0, 0),
        shape=Shape((32, 32, 64), sig_dims=2)
    )
    assert s.flatten_nav((8, 16, 36, 64)) == sflat
    assert unravel_nav(sflat, (8, 16, 32, 64)) == s


def test_roi_1(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize()
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


def test_pick(hdf5, lt_ctx):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 3, 16, 16)
    )
    ds.initialize()
    assert len(ds.shape) == 4
    print(ds.shape)
    pick = lt_ctx.create_pick_analysis(dataset=ds, x=2, y=3)
    lt_ctx.run(pick)


def test_diags(hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize()
    print(ds.diagnostics)


def test_check_valid(hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
    )
    ds = ds.initialize()
    assert ds.check_valid()


def test_timeout_1(hdf5):
    with mock.patch('h5py.File.visititems', side_effect=TimeoutError("too slow")):
        params = H5DataSet.detect_params(hdf5.filename)
        assert list(params.keys()) == ["path"]

        ds = H5DataSet(
            path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
        )
        ds = ds.initialize()
        diags = ds.diagnostics
        print(diags)


def test_timeout_2(hdf5):
    with mock.patch('time.time', side_effect=[1, 30, 30, 60]):
        params = H5DataSet.detect_params(hdf5.filename)
        assert list(params.keys()) == ["path"]

        ds = H5DataSet(
            path=hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16)
        )
        ds = ds.initialize()
        diags = ds.diagnostics
        print(diags)


@pytest.mark.parametrize("mnp", [None, 1, 4])
def test_roi_2(random_hdf5, lt_ctx, mnp):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 4, 16, 16),
        min_num_partitions=mnp,
    )
    ds = ds.initialize()

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
