import numpy as np

from libertem.job.sum import SumFramesJob
from libertem.executor.inline import InlineJobExecutor
from utils import MemoryDataSet


def test_sum_dataset_tilesize_1():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))
    expected = data.sum(axis=(0, 1))

    job = SumFramesJob(dataset=dataset)
    executor = InlineJobExecutor()
    result = np.zeros(job.get_result_shape())

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)
    assert np.allclose(result, expected)
