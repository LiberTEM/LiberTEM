import functools

import pytest
import numpy as np
from numpy.testing import assert_allclose

from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.auto import AutoUDF

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


@pytest.mark.parametrize(
    'with_roi', (True, False)
)
def test_auto_monitor(lt_ctx, with_roi):
    data = _mk_random((16, 8, 32, 64))

    dataset = MemoryDataSet(
        data=data,
        tileshape=(8, 32, 64),
        num_partitions=2,
        sig_dims=2
    )

    if with_roi:
        roi = np.random.choice((True, False), size=dataset.shape.nav)
    else:
        roi = None

    f = functools.partial(np.sum, axis=-1)
    udf = AutoUDF(f=f, monitor=True)
    for res in lt_ctx.run_udf_iter(dataset=dataset, udf=udf, roi=roi):
        # Confirm it is the last valid data point in nav space
        valid = np.argwhere(res.damage.raw_data.reshape((-1, )))
        # Not sure if argwhere() is guaranteed to be sorted. Docs say nothing,
        # but I'd expect it should be since it will
        # go through the values and append indices that are True to the output.
        # Adding to the test to hopefully catch discrepancies.
        # In any case, monitoring is ephemeral, so it should not matter too much
        valid.sort(axis=0)
        if len(valid):
            index = valid[-1][0]
        else:
            index = 0
        last_valid = res.buffers[0]['result'].raw_data[index]
        monitor = res.buffers[0]['monitor']
        assert_allclose(last_valid, monitor)
