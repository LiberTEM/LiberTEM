import numpy as np
import pytest

from libertem import api
from utils import _naive_mask_apply, _mk_random
import libertem.common.backend as bae
from libertem.udf.base import UDF
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.utils.devices import detect


class DebugUDF(UDF):
    def __init__(self, backends=None):
        if backends is None:
            backends = ('cupy', 'numpy')
        super().__init__(backends=backends)

    def get_result_buffers(self):
        return {
            'debug': self.buffer(kind="single", dtype="object"),
            'on_device': self.buffer(kind="sig", dtype=np.float32, where="device"),
            'backend': self.buffer(kind="nav", dtype="object"),
        }

    def preprocess(self):
        self.results.debug[0] = dict()

    def process_partition(self, partition):
        cpu = bae.get_use_cpu()
        cuda = bae.get_use_cuda()
        self.results.debug[0][self.meta.slice] = (cpu, cuda)
        self.results.on_device[:] += self.xp.sum(partition, axis=0)
        self.results.backend[:] = self.meta.backend
        print(f"meta backend {self.meta.backend}")

    def merge(self, dest, src):
        de, sr = dest['debug'][0], src['debug'][0]
        for key, value in sr.items():
            assert key not in de
            de[key] = value

        dest['on_device'][:] += src['on_device']
        dest['backend'][:] = src['backend']

    def get_backends(self):
        return self.params.backends


@pytest.mark.functional
def test_start_local_default(hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
    cudas = detect()['cudas']
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with api.Context() as ctx:
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        _ = ctx.run_udf(udf=DebugUDF(), dataset=hdf5_ds_1)
        if cudas:
            cupy_only = ctx.run_udf(udf=DebugUDF(), dataset=hdf5_ds_1, backends=('cupy',))
        numpy_only = ctx.run_udf(udf=DebugUDF(), dataset=hdf5_ds_1, backends=('numpy',))

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
    if cudas:
        assert np.all(cupy_only['backend'].data == 'cupy')
    assert np.all(numpy_only['backend'].data == 'numpy')



@pytest.mark.functional
def test_start_local_cpuonly(hdf5_ds_1):
    # We don't use all since that might be too many
    cpus = (0, 1)
    hdf5_ds_1.set_num_cores(len(cpus))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    spec = cluster_spec(cpus=cpus, cudas=())
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        udf_res = ctx.run_udf(udf=DebugUDF(), dataset=hdf5_ds_1)
        # No CuPy resources
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugUDF(backends=('cupy',)), dataset=hdf5_ds_1)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
    found = {}

    for val in udf_res['debug'].data[0].values():
        # no CuPy
        assert val[1] is None
        print(val)
        found[val[0]] = True

    # Each CPU got work. We have to see if this
    # actually works always since this depends on the scheduler behavior
    assert set(found.keys()) == set(cpus)

    assert np.all(udf_res['backend'].data == 'numpy')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))


@pytest.mark.functional
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices or no functional CuPy")
def test_start_local_cudaonly(hdf5_ds_1):
    cudas = detect()['cudas']
    # Make sure we have enough partitions
    hdf5_ds_1.set_num_cores(len(cudas))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    spec = cluster_spec(cpus=(), cudas=cudas)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        # Uses ApplyMasksUDF, which supports CuPy
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        udf_res = ctx.run_udf(udf=DebugUDF(), dataset=hdf5_ds_1)
        # No CPU compute resources
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugUDF(backends=('numpy',)), dataset=hdf5_ds_1)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )

    found = {}

    for val in udf_res['debug'].data[0].values():
        # no CPU
        assert val[0] is None
        # Register which GPUs got work
        found[val[1]] = True
        print(val)

    # Test if each GPU got work. We have to see if this
    # actually works always since this depends on the scheduler behavior
    assert set(found.keys()) == set(cudas)

    assert np.all(udf_res['backend'].data == 'cupy')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))
