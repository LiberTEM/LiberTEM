import numpy as np
import pytest

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random
from utils import dataset_correction_verification


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


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(lt_ctx, with_roi):
    data = _mk_random(size=(16, 16, 16, 16))
    ds = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)
