import numpy as np

from libertem.udf.base import UDFRunner
from libertem.executor.inline import InlineJobExecutor
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_simple_multi_udf_run():
    data = _mk_random(size=(32, 1860, 2048))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2,
        base_shape=(1, 930, 16),
        force_need_decode=True,
    )

    executor = InlineJobExecutor()
    udfs = [
        SumSigUDF(),
        SumUDF(),
    ]
    res = UDFRunner(udfs=udfs).run_for_dataset(
        dataset=dataset,
        executor=executor,
    )
    sumsigres, sumres = res.buffers
    print(sumsigres, sumres)
    assert np.allclose(
        sumres['intensity'],
        np.sum(data, axis=0)
    )
    assert np.allclose(
        sumsigres['intensity'],
        np.sum(data, axis=(1, 2))
    )
