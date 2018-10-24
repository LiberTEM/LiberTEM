import pytest
import numpy as np
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.com import COMAnalysis
from utils import MemoryDataSet


@pytest.fixture
def com_dataset():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    data[0, 0] = np.zeros((16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))
    return dataset


def test_com_with_zero_frames(com_dataset):
    params = {
        "cx": 0,
        "cy": 0,
        "r": 0,
    }
    analysis = COMAnalysis(dataset=com_dataset, parameters=params)
    job = analysis.get_job()

    executor = InlineJobExecutor()
    job_results = np.zeros((3, 16, 16))
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(job_results)

    results = analysis.get_results(job_results=job_results)

    # no inf/nan in center_x and center_y
    assert not np.any(np.isinf(results[3].raw_data))
    assert not np.any(np.isinf(results[4].raw_data))
    assert not np.any(np.isnan(results[3].raw_data))
    assert not np.any(np.isnan(results[4].raw_data))

    # no inf/nan in divergence/magnitude
    assert not np.any(np.isinf(results[1].raw_data))
    assert not np.any(np.isinf(results[2].raw_data))
    assert not np.any(np.isnan(results[1].raw_data))
    assert not np.any(np.isnan(results[2].raw_data))
