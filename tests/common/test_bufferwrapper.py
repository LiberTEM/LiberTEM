import pytest
import numpy as np

from libertem.io.dataset.base import Partition
from libertem.io.dataset.memory import MemoryDataSet
from libertem.common.buffers import (
    BufferWrapper, AuxBufferWrapper, reshaped_view, PlaceholderBufferWrapper,
)
from libertem.common import Shape
from libertem.udf.base import UDFData

from utils import _mk_random


def test_new_for_partition():
    auxdata = _mk_random(size=(16, 16), dtype="float32")
    buf = AuxBufferWrapper(kind="nav", dtype="float32")
    buf.set_buffer(auxdata)

    dataset = MemoryDataSet(data=_mk_random(size=(16, 16, 16, 16), dtype="float32"),
                            tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    assert auxdata.shape == tuple(dataset.shape.nav)

    roi = _mk_random(size=dataset.shape.nav, dtype="bool")

    for idx, partition in enumerate(dataset.get_partitions()):
        print("partition number", idx)
        new_buf = buf.new_for_partition(partition, roi=roi)
        ps = partition.slice.get(nav_only=True)
        roi_part = roi.reshape(-1)[ps]

        assert np.product(new_buf._data.shape) == roi_part.sum()

        # old buffer stays the same:
        assert np.allclose(buf._data, auxdata.reshape(-1))
        assert buf._data_coords_global
        assert not new_buf._data_coords_global

        # new buffer is sliced to partition and has ROI applied:
        assert new_buf._data.shape[0] <= buf._data.shape[0]
        assert new_buf._data.shape[0] <= partition.shape[0]

        # let's try and manually apply the ROI to `auxdata`:
        assert np.allclose(
            new_buf._data,
            auxdata.reshape(-1)[ps][roi_part]
        )


def test_buffer_extra_shape_1():
    buffer = BufferWrapper(kind='nav', extra_shape=(2, 3))
    assert buffer._extra_shape == (2, 3)


def test_buffer_extra_shape_2():
    shape_obj = Shape(shape=(12, 13, 14, 15), sig_dims=2)
    buffer = BufferWrapper(kind='nav', extra_shape=shape_obj)
    assert buffer._extra_shape == (12, 13, 14, 15)


def test_reshaped_view():
    data = np.zeros((2, 5))
    view = data[:, :3]
    with pytest.raises(AttributeError):
        reshaped_view(view, (-1, ))
    view_2 = reshaped_view(data, (-1, ))
    view_2[0] = 1
    assert data[0, 0] == 1
    assert np.all(data[0, 1:] == 0)
    assert np.all(data[1:] == 0)


def test_result_buffer_decl():
    buf = PlaceholderBufferWrapper(kind='sig', dtype=np.float32)
    with pytest.raises(ValueError):
        # no array associated with this bufferwrapper:
        np.array(buf)


class PlaceholderPartition(Partition):
    def __init__(
        self, meta, partition_slice, tiles, start_frame: int, num_frames: int,
    ):
        self._tiles = tiles
        self._start_frame = start_frame
        self._num_frames = num_frames
        super().__init__(
            meta=meta,
            partition_slice=partition_slice,
            io_backend=None,
        )

    def get_tiles(self, tiling_scheme, dest_dtype=np.float32, roi=None):
        assert roi is None

        # FIXME: stop after processing `num_frames`
        for tile in self._tiles:
            yield tile

    def need_decode(self, read_dtype, roi, corrections):
        return True

    def get_base_shape(self, roi):
        return (930, 16)

    def adjust_tileshape(self, tileshape, roi):
        return tileshape  # FIXME

    def set_corrections(self, corrections):
        self._corrections = corrections


def test_sig_slicing():
    from libertem.common import Shape, Slice
    from libertem.udf.sum import SumUDF
    from libertem.io.dataset.base import DataTile, DataSetMeta
    from libertem.udf.base import UDFMeta
    from libertem.executor.base import Environment

    partition_slice = Slice(
        origin=(0, 0, 256),
        shape=Shape((4000, 1860, 256), sig_dims=2),
    )

    dataset_shape = Shape(
        (4000, 1860, 2048), sig_dims=2,
    )

    dsmeta = DataSetMeta(
        shape=dataset_shape,
        raw_dtype=np.uint16,
        image_count=4000,
    )

    partition = PlaceholderPartition(
        meta=dsmeta,
        partition_slice=partition_slice,
        start_frame=0,
        num_frames=4000,
        tiles=[],
    )

    roi = None
    corrections = None  # FIXME?
    device_class = 'cpu'
    dtype = np.uint16
    env = Environment(threads_per_worker=2)

    tile_slice = Slice(
        origin=(0, 0, 480),
        shape=Shape((1, 930, 16), sig_dims=2),
    )
    data = np.zeros((1, 930, 16), dtype=np.uint16)
    tile = DataTile(
        data,
        tile_slice=tile_slice,
        scheme_idx=30,
    )

    udf = SumUDF()

    meta = UDFMeta(
        partition_shape=partition.slice.adjust_for_roi(roi).shape,
        dataset_shape=dataset_shape,
        roi=roi,
        dataset_dtype=partition.dtype,
        input_dtype=dtype,
        tiling_scheme=None,
        corrections=corrections,
        device_class=device_class,
        threads_per_worker=env.threads_per_worker,
    )

    udf.set_meta(meta)
    udf.init_result_buffers()
    udf.allocate_for_part(partition=partition, roi=None)
    udf.init_task_data()

    udf.set_contiguous_views_for_tile(
        partition=partition,
        tile=tile,
    )
    udf.process_tile(tile)


def test_sig_slicing_2():
    from libertem.common import Shape, Slice
    from libertem.io.dataset.base import DataTile, DataSetMeta

    partition_slice = Slice(
        origin=(0, 0, 256),
        shape=Shape((4000, 1860, 256), sig_dims=2),
    )

    dataset_shape = Shape(
        (4000, 1860, 2048), sig_dims=2,
    )

    dsmeta = DataSetMeta(
        shape=dataset_shape,
        raw_dtype=np.uint16,
        image_count=4000,
    )

    partition = PlaceholderPartition(
        meta=dsmeta,
        partition_slice=partition_slice,
        start_frame=0,
        num_frames=4000,
        tiles=[],
    )

    tile_slice = Slice(
        origin=(0, 0, 480),
        shape=Shape((1, 930, 16), sig_dims=2),
    )
    data = np.zeros((1, 930, 16), dtype=np.uint16)
    tile = DataTile(
        data,
        tile_slice=tile_slice,
        scheme_idx=30,
    )
    roi = None

    buf = BufferWrapper(
        kind='sig',
        extra_shape=(),
        dtype=np.uint16,
        where=None,
        use=None,
    )
    buf.set_shape_partition(partition, roi)
    buf.allocate(lib=None)
    view = buf.get_contiguous_view_for_tile(partition, tile)
    assert view.size > 0


def test_sig_slicing_views_for_partition():
    from libertem.common import Shape, Slice
    from libertem.io.dataset.base import DataTile, DataSetMeta

    partition_slice = Slice(
        origin=(0, 0, 256),
        shape=Shape((4000, 1860, 256), sig_dims=2),
    )

    dataset_shape = Shape(
        (4000, 1860, 2048), sig_dims=2,
    )

    dsmeta = DataSetMeta(
        shape=dataset_shape,
        raw_dtype=np.uint16,
        image_count=4000,
    )

    partition = PlaceholderPartition(
        meta=dsmeta,
        partition_slice=partition_slice,
        start_frame=0,
        num_frames=4000,
        tiles=[],
    )

    tile_slice = Slice(
        origin=(0, 0, 480),
        shape=Shape((1, 930, 16), sig_dims=2),
    )
    data = np.zeros((1, 930, 16), dtype=np.uint16)
    tile = DataTile(
        data,
        tile_slice=tile_slice,
        scheme_idx=30,
    )
    roi = None

    buf = BufferWrapper(
        kind='sig',
        extra_shape=(),
        dtype=np.uint16,
        where=None,
        use=None,
    )
    buf.set_shape_ds(dataset_shape, roi)
    buf.allocate(lib=None)
    view_p = buf.get_view_for_partition(partition)
    view_p[:] = 1

    # FIXME: this works for now, as the dataset parameter is not used yet. we
    # may need to stub out the dataset, too, in the future
    view_ds = buf.get_view_for_dataset(dataset=None)

    assert np.allclose(view_ds[partition.slice.sig.get()], 1)

    # the partition "left" of that is zero:
    partition_slice_0 = Slice(
        origin=(0, 0, 0),
        shape=Shape((4000, 1860, 256), sig_dims=2),
    )
    assert np.allclose(view_ds[partition_slice_0.sig.get()], 0)

    # actually, everything _but_ the give partition is 0:
    view_roi = np.zeros(dataset_shape.sig, dtype=bool)
    view_roi[partition.slice.sig.get()] = 1
    assert np.allclose(view_ds[~view_roi], 0)


def test_sig_slicing_views_for_partition_2():
    from libertem.common import Shape, Slice
    from libertem.io.dataset.base import DataTile, DataSetMeta

    partition_slice = Slice(
        origin=(0, 0, 256),
        shape=Shape((4000, 1860, 256), sig_dims=2),
    )

    dataset_shape = Shape(
        (4000, 1860, 2048), sig_dims=2,
    )

    dsmeta = DataSetMeta(
        shape=dataset_shape,
        raw_dtype=np.uint16,
        image_count=4000,
    )

    partition = PlaceholderPartition(
        meta=dsmeta,
        partition_slice=partition_slice,
        start_frame=0,
        num_frames=4000,
        tiles=[],
    )

    tile_slice = Slice(
        origin=(0, 0, 480),
        shape=Shape((1, 930, 16), sig_dims=2),
    )
    data = np.zeros((1, 930, 16), dtype=np.uint16)
    tile = DataTile(
        data,
        tile_slice=tile_slice,
        scheme_idx=30,
    )
    roi = None

    # we want to test merging of a partition into the dataset buffer, so we
    # create two UDFData instances:
    ud_ds = UDFData({
        'buf': BufferWrapper(
            kind='sig',
            extra_shape=(),
            dtype=np.uint16,
            where=None,
            use=None,
        )
    })
    for k, buf in ud_ds._get_buffers():
        buf.set_shape_ds(dataset_shape, roi)
    for k, buf in ud_ds._get_buffers(filter_allocated=True):
        buf.allocate()

    # emulate what's happening in run_for_partition:
    ud_p = UDFData({
        'buf': BufferWrapper(
            kind='sig',
            extra_shape=(),
            dtype=np.uint16,
            where=None,
            use=None,
        )
    })
    for k, buf in ud_p._get_buffers():
        buf.set_shape_partition(partition, roi)
    for k, buf in ud_p._get_buffers(filter_allocated=True):
        buf.allocate(lib=None)

    # set the whole partition result to 1:
    ud_p.buf[:] = 1

    # and wrap up:
    ud_p.clear_views()
    ud_p.export()

    # prepare ds result buffer:
    ud_ds.set_view_for_partition(partition)

    # now, simulate a merge:
    dest = ud_ds.get_proxy()
    src = ud_p.get_proxy()
    dest.buf[:] += src.buf

    ud_p.clear_views()
    ud_ds.clear_views()

    # everything _but_ the give partition is 0:
    view_roi = np.zeros(dataset_shape.sig, dtype=bool)
    view_roi[partition.slice.sig.get()] = 1
    assert np.allclose(ud_ds.buf[~view_roi], 0)
    assert np.allclose(ud_ds.buf[view_roi], 1)
