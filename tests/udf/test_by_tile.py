import pytest
import numpy as np

from libertem.udf import UDF
from utils import MemoryDataSet, _mk_random


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile, tile_slice):
        # the last tile contains only two frames:
        assert tile.shape[0] in (7, 2)

        assert len(tile.shape) == 3
        assert tile.shape[1:] == (16, 16)
        assert self.results.pixelsum.shape == (tile.shape[0],)
        axes = tuple(range(1, len(tile.shape)))
        self.results.pixelsum[:] += np.sum(tile, axis=axes)


def test_sum_tiles(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


class PixelsumForCropped(UDF):
    """
    like above but different asserts in process_tile
    """
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile, tile_slice):
        # the last tile contains only two frames:
        assert tile.shape[0] in (7, 2)
        assert len(tile.shape) == 3
        assert self.results.pixelsum.shape == (tile.shape[0],)
        axes = tuple(range(1, len(tile.shape)))
        self.results.pixelsum[:] += np.sum(tile, axis=axes)


def test_mem_cropped(lt_ctx):
    """
    to make sure the memory dataset works fine with cropping:
    """
    data = _mk_random(size=(16, 16, 24, 24), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 7, 7),
                            num_partitions=2, sig_dims=2, crop_frames=True)

    buf = np.zeros((256, 24, 24), dtype="float32")
    for p in dataset.get_partitions():
        for tile in p.get_tiles():
            assert tuple(tile.tile_slice.shape)[0] in (7, 2)
            assert tuple(tile.tile_slice.shape)[1:] in [(7, 7),
                                                        (7, 3),
                                                        (3, 7),
                                                        (3, 3)]
            buf[tile.tile_slice.get()] = tile.data

    assert np.allclose(
        buf.reshape(data.shape), data
    )


def test_cropped(lt_ctx):
    data = _mk_random(size=(16, 16, 24, 24), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 7, 7),
                            num_partitions=2, sig_dims=2, crop_frames=True)

    pixelsum = PixelsumForCropped()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


class FrameCounter(UDF):
    def get_result_buffers(self):
        return {
            'counter': self.buffer(kind="single", dtype="int64"),
        }

    def process_tile(self, tile, tile_slice):
        self.results.counter += 1  # FIXME

    def merge(self, dest, src):
        dest['counter'][:] += src['counter']


@pytest.mark.xfail
def test_frame_counter(lt_ctx):
    data = _mk_random(size=(16, 16, 24, 24), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 7, 7),
                            num_partitions=2, sig_dims=2, crop_frames=True)

    counter = FrameCounter()
    res = lt_ctx.run_udf(dataset=dataset, udf=counter)
    assert 'counter' in res
    print(data.shape, res['counter'].data.shape)
    assert res['counter'].data == 256
