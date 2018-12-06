import numpy as np

from libertem.job.raw import PickFrameJob
from libertem.analysis.raw import PickFrameAnalysis
from libertem.executor.inline import InlineJobExecutor
from libertem.common import Slice, Shape
from utils import MemoryDataSet


def test_get_single_frame(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16),
        sig_dims=2
    )

    job = lt_ctx.create_pick_job(dataset=dataset, x=5, y=5)
    result = lt_ctx.run(job)

    assert result.shape == (16, 16)
    assert np.allclose(result, data[5, 5])


def test_get_multiple_frames(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16),
        sig_dims=2
    )

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(0, 0, 0, 0), shape=Shape((1, 2, 16, 16), sig_dims=2)
    ))

    result = lt_ctx.run(job)

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
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16),
        sig_dims=2
    )

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(5, 5, 0, 0), shape=Shape((5, 5, 16, 16), sig_dims=2)
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
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16),
        sig_dims=2
    )

    job = PickFrameJob(dataset=dataset, slice_=Slice(
        origin=(5, 0, 0, 0), shape=Shape((1, 16, 16, 16), sig_dims=2)
    ))

    executor = InlineJobExecutor()

    result = np.zeros(job.get_result_shape())
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert result.shape == (1, 16, 16, 16)
    assert not np.allclose(result[0, 0], result[0, 1])
    assert np.allclose(result[0, 0:16], data[5, 0:16])


def test_pick_analysis(lt_ctx):
    """
    the other tests cover the pick job, this one uses the analysis
    """
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16),
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 5, "y": 5})
    result = lt_ctx.run(analysis)

    assert result.intensity.raw_data.shape == (16, 16)
    assert np.allclose(result.intensity.raw_data, data[5, 5])
