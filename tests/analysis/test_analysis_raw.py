import numpy as np
import pytest

from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_pick_analysis(lt_ctx):
    """
    the other tests cover the pick job, this one uses the analysis
    """
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 5, "y": 5})
    result = lt_ctx.run(analysis)

    assert result.intensity.raw_data.shape == (16, 16)
    assert np.allclose(result.intensity.raw_data, data[5, 5])
    assert np.allclose(result.intensity_lin.raw_data, data[5, 5])


def test_pick_from_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 5})
    result = lt_ctx.run(analysis)

    assert result.intensity.raw_data.shape == (16, 16)
    assert np.allclose(result.intensity.raw_data, data[5])
    assert np.allclose(result.intensity_lin.raw_data, data[5])


def test_pick_analysis_via_api_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = lt_ctx.create_pick_analysis(dataset=dataset, x=8, y=7)
    result = lt_ctx.run(analysis)

    assert result.intensity.raw_data.shape == (16, 16)
    assert np.allclose(result.intensity.raw_data, data[7, 8])
    assert np.allclose(result.intensity_lin.raw_data, data[7, 8])


def test_pick_analysis_via_api_2_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = lt_ctx.create_pick_analysis(dataset=dataset, x=8)
    result = lt_ctx.run(analysis)

    assert result.intensity.raw_data.shape == (16, 16)
    assert np.allclose(result.intensity.raw_data, data[8])
    assert np.allclose(result.intensity_lin.raw_data, data[8])


def test_pick_analysis_via_api_3_3d_ds_fail_1(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8, "z": 11})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)


def test_pick_analysis_via_api_3_3d_ds_fail_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8, "z": 11})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)


def test_pick_analysis_via_api_3_3d_ds_fail_3(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)


def test_pick_analysis_via_api_3_3d_ds_fail_4(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8, "z": 11})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)


def test_pick_analysis_via_api_3_3d_ds_fail_5(lt_ctx):
    data = _mk_random(size=(16, 256, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2
    )

    analysis = PickFrameAnalysis(dataset=dataset, parameters={"x": 7, "y": 8, "z": 11})
    with pytest.raises(ValueError):
        lt_ctx.run(analysis)
