import functools

import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_auto(lt_ctx):
    data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                            num_partitions=2, sig_dims=2)

    auto_result = lt_ctx.map(dataset=dataset, f=functools.partial(np.sum, axis=-1))

    naive_result = data.sum(axis=-1)

    assert np.allclose(auto_result.data, naive_result)


def test_auto_weird(lt_ctx):
    data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(data=data, tileshape=(8, 32, 64),
                            num_partitions=2, sig_dims=2)

    def f(frame):
        return [
            "Shape %s" % str(frame.shape),
            dict(shape=frame.shape, sum=frame.sum()),
            lambda x: x,
            MemoryDataSet
        ]

    auto_result = lt_ctx.map(dataset=dataset, f=f)
    item = auto_result.data[0, 0]

    assert len(item) == 4
    assert isinstance(item[0], str)
    assert isinstance(item[1], dict)
    assert callable(item[2])
    assert item[2](1) == 1
    assert isinstance(item[3], type)
