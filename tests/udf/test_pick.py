import pytest
import numpy as np

from libertem.udf.raw import PickUDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_pick(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 7, 7),
                            num_partitions=7, sig_dims=2)
    roi = np.random.choice([True, False], size=dataset.shape.nav)

    udf = PickUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    assert np.allclose(data[roi], res['intensity'].data)
    assert data.dtype == res['intensity'].data.dtype


def test_pick_zero_roi(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 7, 7),
                            num_partitions=7, sig_dims=2)
    roi = np.zeros(dataset.shape.nav, dtype=bool)

    udf = PickUDF()
    with pytest.raises(ValueError) as e:
        res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    assert e.match("cannot load data with all-zeros ROI")
