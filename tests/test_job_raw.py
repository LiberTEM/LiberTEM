import numpy as np

from libertem.job.raw import PickFrameJob
from libertem.executor.inline import InlineJobExecutor
from libertem.common.slice import Slice
from utils import MemoryDataSet


def test_get_single_frame():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(5, 5, 0, 0), shape=(1, 1, 16, 16)
    ))

    executor = InlineJobExecutor()

    result = np.zeros(job.get_result_shape())
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert result.shape == (1, 1, 16, 16)
    assert np.allclose(result[0, 0], data[5, 5])


def test_get_multiple_frames():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(0, 0, 0, 0), shape=(1, 2, 16, 16)
    ))

    executor = InlineJobExecutor()

    result = np.zeros(job.get_result_shape())
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    print(result[0, 0].astype("uint32"))
    print(data[0, 0])
    print(result[0, 1].astype("uint32"))
    print(data[0, 1])

    assert result.shape == (1, 2, 16, 16)
    assert not np.allclose(result[0, 0], result[0, 1])
    assert np.allclose(result[0, 0], data[0, 0])
    assert np.allclose(result[0, 0:2], data[0, 0:2])


def test_get_multiple_frames_2():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(5, 5, 0, 0), shape=(5, 5, 16, 16)
    ))

    executor = InlineJobExecutor()

    result = np.zeros(job.get_result_shape())
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert result.shape == (5, 5, 16, 16)
    assert not np.allclose(result[0, 0], result[0, 1])
    assert np.allclose(result[0, 0], data[5, 5])
    assert np.allclose(result[0, 0:2], data[5, 5:7])


def test_get_multiple_frame_row():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(5, 0, 0, 0), shape=(1, 16, 16, 16)
    ))

    executor = InlineJobExecutor()

    result = np.zeros(job.get_result_shape())
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert result.shape == (1, 16, 16, 16)
    assert not np.allclose(result[0, 0], result[0, 1])
    assert np.allclose(result[0, 0:16], data[5, 0:16])
