import numpy as np
import pytest

from libertem.io.dataset import get_extensions
from libertem.io.dataset.base import (
    _roi_to_nd_indices, DataSetException
)
from libertem.common import Shape, Slice
from libertem.io.dataset.memory import MemoryDataSet
from libertem.io.dataset.base import TilingScheme

from utils import _mk_random


def test_sweep_stackheight():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        num_partitions=2,
    )
    for stackheight in range(1, 256):
        tileshape = Shape(
            (stackheight,) + tuple(dataset.shape.sig),
            sig_dims=dataset.shape.sig.dims
        )
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=tileshape,
            dataset_shape=dataset.shape,
        )
        print("testing with stackheight", stackheight)
        for p in dataset.get_const_partitions(partition_size=128):
            for tile in p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32"):
                pass


def test_roi_to_nd_indices():
    roi = np.full((5, 5), False)
    roi[1, 2] = True
    roi[2, 1:4] = True
    roi[3, 2] = True

    part_slice = Slice(
        origin=(2, 0, 0, 0),
        shape=Shape((2, 5, 16, 16), sig_dims=2)
    )

    assert list(_roi_to_nd_indices(roi, part_slice)) == [
        (2, 1), (2, 2), (2, 3),
                (3, 2)
    ]

    part_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((5, 5, 16, 16), sig_dims=2)
    )

    assert list(_roi_to_nd_indices(roi, part_slice)) == [
                (1, 2),
        (2, 1), (2, 2), (2, 3),         # NOQA: E131
                (3, 2)
    ]


def test_get_extensions():
    exts = get_extensions()
    assert len(exts) >= 15
    assert "mib" in exts
    assert "gtg" in exts
    # etc...


def test_filetype_auto(hdf5, lt_ctx):
    ds = lt_ctx.load("auto", path=hdf5.filename)
    assert ds.ds_path == "data"


def test_filetype_auto_fail_no_path(lt_ctx):
    with pytest.raises(TypeError):
        lt_ctx.load("auto")


def test_filetype_auto_fail_file_does_not_exist(lt_ctx):
    with pytest.raises(DataSetException) as e:
        lt_ctx.load("auto", path="/does/not/exist/believe_me")
    assert e.match("could not determine DataSet type for file")
