import numpy as np

from utils import MemoryDataSet


def test_sum_dataset_tilesize_1(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert np.allclose(results.intensity.raw_data, expected)
