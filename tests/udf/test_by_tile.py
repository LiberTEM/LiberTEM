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

    def process_tile(self, tile):
        # the last tile contains only two frames:
        assert tile.shape[0] in (7, 2)

        assert len(tile.shape) == 3
        assert tile.shape[1:] == (16, 16)
        assert self.results.pixelsum.shape == (tile.shape[0],)
        axes = tuple(range(1, len(tile.shape)))
        self.results.pixelsum[:] = np.sum(tile, axis=axes)


def test_sum_tiles(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))
