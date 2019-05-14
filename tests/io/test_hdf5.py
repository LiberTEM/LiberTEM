import pickle
import cloudpickle
import numpy as np
from libertem.io.dataset.hdf5 import H5DataSet, unravel_nav
from libertem.common import Slice, Shape

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
