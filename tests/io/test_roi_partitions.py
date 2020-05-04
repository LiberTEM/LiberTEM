import numpy as np
import pytest

from libertem.common import Shape
from libertem.udf.sum import SumUDF
from libertem.io.dataset.roi import RoiDataSet
from libertem.io.dataset.base import TilingScheme
from libertem.io.dataset.memory import MemoryDataSet


def test_smoke_udf(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)

    lt_ctx.run_udf(dataset=roi_ds, udf=SumUDF())


def test_smoke_simple(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    tileshape = Shape(
        (16,) + tuple(default_raw.shape.sig),
        sig_dims=default_raw.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)
    for p in roi_ds.get_partitions():
        for tile in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


def test_get_macrotile(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)
    for p in roi_ds.get_partitions():
        p.get_macrotile()


def test_get_macrotile_from_overlapping_parts(lt_ctx):
    data = np.random.randn(16, 16, 32, 32).astype("float32")
    wrapped_ds = MemoryDataSet(
        data=data,
        num_partitions=27, sig_dims=2,
    )

    # make a single roi partition from all partitions of `wrapped_ds`
    roi = np.ones((16, 16), dtype=np.bool)

    roi_ds = RoiDataSet(wrapped=wrapped_ds, rois=[roi])
    parts = list(roi_ds.get_partitions())
    assert len(parts) == 1
    p = parts[0]
    macrotile = p.get_macrotile()
    assert np.allclose(macrotile, data)


def test_get_tiles_from_overlapping_parts(lt_ctx):
    data = np.random.randn(16, 16, 32, 32).astype("float32")
    wrapped_ds = MemoryDataSet(
        data=data,
        num_partitions=27, sig_dims=2,
    )

    # make a single roi partition from all partitions of `wrapped_ds`
    roi = np.ones((16, 16), dtype=np.bool)

    roi_ds = RoiDataSet(wrapped=wrapped_ds, rois=[roi])
    parts = list(roi_ds.get_partitions())
    assert len(parts) == 1
    p = parts[0]

    # shape: whole roi_ds
    tileshape = Shape(
        (16 * 16,) + tuple(wrapped_ds.shape.sig),
        sig_dims=wrapped_ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=wrapped_ds.shape,
    )

    tiles = list(p.get_tiles(tiling_scheme=tiling_scheme))
    assert len(tiles) == 1
    macrotile = tiles[0]
    assert np.allclose(
        macrotile,
        data.reshape(wrapped_ds.shape.flatten_nav())
    )


def test_nested_rois_get_tiles(lt_ctx):
    raise NotImplementedError()


def test_nested_rois_get_macrotile(lt_ctx):
    raise NotImplementedError()
