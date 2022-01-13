import numpy as np

from libertem.udf.raw import PickUDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_pick(lt_ctx, delayed_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    # data = np.ones((16, 16, 16, 16), dtype="float32")
    # data = np.arange(0, 16*16*16*16, dtype="float32").reshape((16, 16, 16, 16))
    dataset = MemoryDataSet(data=data, tileshape=(3, 7, 16),
                            num_partitions=7, sig_dims=2)
    roi = np.random.choice([True, False], size=dataset.shape.nav)
    roi[0] = True

    udf = PickUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
    res_delayed = delayed_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    assert np.allclose(data[roi], res['intensity'].data)
    assert np.allclose(data[roi], res_delayed['intensity'].data)

    assert data.dtype == res['intensity'].data.dtype
    assert data.dtype == res_delayed['intensity'].data.dtype


def test_pick_empty_roi(lt_ctx, delayed_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 7, 7),
                            num_partitions=7, sig_dims=2)
    roi = np.zeros(dataset.shape.nav, dtype=bool)

    udf = PickUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
    res_delayed = delayed_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    assert np.allclose(data[roi], res['intensity'].data)
    assert np.allclose(data[roi], res_delayed['intensity'].data)

    assert data[roi].shape == res['intensity'].data.shape
    assert data[roi].shape == res_delayed['intensity'].data.shape

    assert data.dtype == res['intensity'].data.dtype
    assert data.dtype == res_delayed['intensity'].data.dtype
