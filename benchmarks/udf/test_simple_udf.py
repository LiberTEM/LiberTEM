import numpy as np
import pytest

from libertem.udf.base import UDF, NoOpUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect
from libertem.common.backend import set_use_cpu, set_use_cuda


class NoopSigUDF(UDF):
    def get_result_buffers(self):
        return {
            "sigbuf": self.buffer(kind="sig", dtype=int, where="device")
        }

    def process_tile(self, tile):
        pass

    def merge(self, dest, src):
        pass

    def get_backends(self):
        return ('numpy', 'cupy')


@pytest.mark.benchmark(
    group="udf"
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_udf_noncontiguous_tiles(lt_ctx, backend, benchmark):
    if backend == 'cupy':
        d = detect()
        cudas = d['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    data = np.zeros(shape=(30, 3, 256), dtype="float32")
    dataset = MemoryDataSet(
        data=data, tileshape=(3, 2, 2),
        num_partitions=2, sig_dims=2
    )
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        udf = NoopSigUDF()
        res = benchmark(lt_ctx.run_udf, udf=udf, dataset=dataset)
    finally:
        set_use_cpu(0)

    assert np.all(res["sigbuf"].data == 0)


@pytest.mark.benchmark(
    group="udf overheads"
)
def test_overhead(shared_dist_ctx, benchmark):
    ds = shared_dist_ctx.load(
        'memory',
        data=np.zeros((1024, 2)),
        sig_dims=1,
        num_partitions=32
    )
    benchmark(
        shared_dist_ctx.run_udf,
        dataset=ds,
        udf=NoOpUDF()
    )
