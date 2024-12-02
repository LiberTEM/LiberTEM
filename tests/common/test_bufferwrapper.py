import pytest
import numpy as np

from libertem.io.dataset.memory import (
    MemPartition, MemoryDataSet, FileSet, MemoryFile
)
from libertem.common.buffers import (
    BufferWrapper, AuxBufferWrapper, reshaped_view, PlaceholderBufferWrapper,
)
from libertem.common import Shape, Slice
from libertem.udf.base import UDF

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

        assert np.prod(new_buf._data.shape) == roi_part.sum()

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


class BadMemPartition(MemPartition):
    def get_tiles(self, *args, **kwargs):
        for t in super().get_tiles(*args, **kwargs):
            sl = t.tile_slice
            origin = np.array(sl.origin)
            origin += 1
            new_sl = Slice(origin=tuple(origin), shape=sl.shape)
            t.tile_slice = new_sl
            yield t


class BadMemoryDS(MemoryDataSet):
    def get_partitions(self):
        fileset = FileSet([
            MemoryFile(
                path=None,
                start_idx=0,
                end_idx=self._image_count,
                native_dtype=self.data.dtype,
                sig_shape=self.shape.sig,
                data=self.data.reshape(self.shape.flatten_nav()),
                check_cast=self._check_cast,
            )
        ])

        for part_slice, start, stop in self.get_slices():
            yield BadMemPartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
                tiledelay=self._tiledelay,
                tileshape=self.tileshape,
                force_need_decode=self._force_need_decode,
                io_backend=self.get_io_backend(),
                decoder=self.get_decoder(),
            )


class BaseSigUDF(UDF):
    def get_result_buffers(self):
        return {
            'data': self.buffer(kind='sig')
        }

    def merge(self, dest, src):
        dest.data[:] += src.data


class FrameSigUDF(BaseSigUDF):
    def process_frame(self, frame):
        self.results.data[:] += frame


class TileSigUDF(BaseSigUDF):
    def process_tile(self, tile):
        self.results.data[:] += tile.sum(axis=0)


class PartitionSigUDF(BaseSigUDF):
    def process_partition(self, partition):
        self.results.data[:] += partition.sum(axis=0)


class BaseNavUDF(UDF):
    def get_result_buffers(self):
        return {
            'data': self.buffer(kind='nav')
        }


class FrameNavUDF(BaseNavUDF):
    def process_frame(self, frame):
        self.results.data[:] = frame.sum()


class TileNavUDF(BaseNavUDF):
    def process_tile(self, tile):
        self.results.data[:] = tile.sum(axis=tuple(range(1, len(tile.shape))))


class PartitionNavUDF(BaseNavUDF):
    def process_partition(self, partition):
        self.results.data[:] = partition.sum(
            axis=tuple(range(1, len(partition.shape)))
        )


@pytest.mark.parametrize(
    'tileshape,udf_clss', (
        [
            None, (
                FrameSigUDF, TileSigUDF, PartitionSigUDF, FrameNavUDF,
                TileNavUDF, PartitionNavUDF
            )
        ],
        [(7, 16, 16), (FrameSigUDF, TileSigUDF, FrameNavUDF, TileNavUDF)],
        [(7, 7, 7), (TileSigUDF, TileNavUDF)],
        [(16, 8, 16), (TileSigUDF, TileNavUDF)],
        [
            (8*16, 16, 16), (
                FrameSigUDF, TileSigUDF, PartitionSigUDF, FrameNavUDF,
                TileNavUDF, PartitionNavUDF
            )
        ],
    )
)
def test_buffer_slices(lt_ctx, tileshape, udf_clss):
    data = _mk_random(size=(16, 16, 16, 16))
    bad_ds = BadMemoryDS(
        data=data,
        tileshape=tileshape,
        num_partitions=2,
        sig_dims=2
    )
    ds = MemoryDataSet(
        data=data,
        tileshape=tileshape,
        num_partitions=2,
        sig_dims=2
    )

    for udf_cls in udf_clss:
        _ = lt_ctx.run_udf(dataset=ds, udf=udf_cls())
        with pytest.raises(Exception) as exc_info:
            _ = lt_ctx.run_udf(dataset=bad_ds, udf=udf_cls())
        assert exc_info.errisinstance((AssertionError, IndexError))
