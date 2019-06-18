import pytest
import numpy as np

from libertem.udf.base import UDFRunner, UDF
from utils import MemoryDataSet, _mk_random
from libertem.web.base import run_blocking


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        assert frame.shape == (16, 16)
        assert self.results.pixelsum.shape == (1,)
        self.results.pixelsum[:] = np.sum(frame)


@pytest.mark.asyncio
async def test_async_run_for_dset(async_executor):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    roi = np.zeros((256,), dtype=bool)
    runner = UDFRunner(pixelsum)

    async for udf_results in runner.run_for_dataset_async(dataset, async_executor, roi=roi):
        pass
    assert "udf_results" in locals(), "must yield at least one result"
