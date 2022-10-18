import numpy as np
import pytest

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

    def process_partition(self, partition):
        self.results.pixelsum[:] += np.sum(partition, axis=(-1, -2))


@pytest.mark.parametrize(
    'tileshape', (None, ((16*16)//7, 16, 16))
)
def test_sum_tiles(lt_ctx, tileshape):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    # The dataset can force a different tile size even for process_partition!
    # At least MemoryDataSet does that and we should check that everything is ok...
    dataset = MemoryDataSet(
        data=data, num_partitions=7, sig_dims=2, tileshape=tileshape
    )

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


class TouchUDF(UDF):
    def get_result_buffers(self):
        return {
            'touched': self.buffer(
                kind="nav", dtype="int"
            )
        }

    def process_partition(self, partition):
        print(partition.shape)
        self.results.touched[:] += 1
        assert partition.shape[0] == self.meta.coordinates.shape[0]


@pytest.mark.parametrize(
    'use_roi', (False, True)
)
@pytest.mark.parametrize(
    'tileshape', (None, (7, 16, 16), (8 * 16, 16, 16))
)
def test_partition_roi(lt_ctx, use_roi, tileshape):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, num_partitions=2, sig_dims=2, tileshape=tileshape)
    if use_roi:
        roi = np.random.choice([True, False], dataset.shape.nav)
    else:
        roi = None
    udf = TouchUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
    print(data.shape, res['touched'].data.shape)
    assert np.all(res['touched'].raw_data == 1)
