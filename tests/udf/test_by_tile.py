import numpy as np

from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile):
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

    def process_tile(self, tile):
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
                            num_partitions=2, sig_dims=2)

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
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumForCropped()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


def test_roi_extra_dimension_shape(lt_ctx):
    """
    Test sum over the pixels for 2-dimensional dataset

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them

    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    class ExtraShapeUDF(UDF):
        def get_result_buffers(self):
            return {
                'test': self.buffer(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'test2': self.buffer(
                    kind="sig", extra_shape=(2,), dtype="float32"
                ),
                # 'test3': self.buffer(
                #     kind="single", extra_shape=(2,), dtype="float32"
                # )
            }

        def process_tile(self, tile):
            tile_slice = self.meta.slice
            self.results.test[:] = np.ones(tuple(tile_slice.shape.nav)) * (1, 2)
            framecount = np.prod(tuple(tile_slice.shape.nav))
            self.results.test2[:] += np.ones(tuple(tile_slice.shape.sig) + (2, )) * framecount
            # self.results.test3[:] += (framecount, 2*framecount)

        def merge(self, dest, src):
            dest.test[:] = src.test[:]
            dest.test2[:] += src.test2[:]
            # dest.test3[:] += src.test3[:]

    extra = ExtraShapeUDF()
    roi = _mk_random(size=dataset.shape.nav, dtype=bool)
    res = lt_ctx.run_udf(dataset=dataset, udf=extra, roi=roi)

    navcount = np.count_nonzero(roi)

    print(data.shape, res['test'].data.shape)
    assert res['test'].data.shape == tuple(dataset.shape.nav) + (2,)
    assert res['test2'].data.shape == tuple(dataset.shape.sig) + (2,)
    # assert res['test3'].data.shape == (2,)
    assert np.allclose(res['test'].raw_data, (1, 2))
    assert np.allclose(res['test2'].raw_data, navcount)
    # assert np.allclose(res['test3'].raw_data, (navcount, 2*navcount))
