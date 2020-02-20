import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_get_macrotile():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )

    p = next(dataset.get_partitions())
    mt = p.get_macrotile()
    assert tuple(mt.shape) == (128, 16, 16)
