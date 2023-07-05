import os
import re
from collections import defaultdict

import numpy as np
import pytest
import distributed as dd

from libertem import api
from libertem.udf.base import NoOpUDF
from utils import _naive_mask_apply, _mk_random
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.utils.devices import detect, has_cupy
from libertem.executor.utils.gpu_plan import DEFAULT_RAM_PER_WORKER
from libertem.common.scheduler import Scheduler

from utils import DebugDeviceUDF


def characterize(spec):
    per_resource = defaultdict(lambda: 0)
    per_cpu_id = defaultdict(lambda: 0)
    per_cuda_id = defaultdict(lambda: 0)

    for name, worker in spec.items():
        if 'resources' not in worker['options'] or 'compute' not in worker['options']['resources']:
            per_resource['service'] += 1
        if 'preload' in worker['options']:
            for line in worker['options']['preload']:
                is_cpu = re.match(r'.*worker_setup\(resource="CPU", device=([0-9]*)\)', line)
                if is_cpu:
                    cpu_id = int(is_cpu.groups()[0])
                    per_cpu_id[cpu_id] += 1
                is_cuda = re.match(r'.*worker_setup\(resource="CUDA", device=([0-9]*)\)', line)
                if is_cuda:
                    cuda_id = int(is_cuda.groups()[0])
                    per_cuda_id[cuda_id] += 1

        for resource in 'compute', 'CPU', 'CUDA', 'ndarray':
            if resource in worker['options']['resources']:
                per_resource[resource] += 1
    return dict(per_resource), dict(per_cpu_id), dict(per_cuda_id)


def test_start_local_default(hdf5_ds_1, local_cluster_ctx):
    mask = _mk_random(size=(16, 16))
    d = detect()
    cudas = d['cudas']
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    ctx = local_cluster_ctx
    analysis = ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )

    num_cores_ds = ctx.load('memory', data=np.zeros((2, 3, 4, 5)))

    workers = ctx.executor.get_available_workers()
    scheduler = Scheduler(workers)

    assert num_cores_ds._cores == scheduler.effective_worker_count()

    # Based on ApplyMasksUDF, which is CuPy-enabled
    hybrid = ctx.run(analysis)
    _ = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'cuda', 'numpy')), dataset=hdf5_ds_1)
    if cudas:
        cuda_only = ctx.run_udf(
            udf=DebugDeviceUDF(backends=('cuda', 'numpy')),
            dataset=hdf5_ds_1,
            backends=('cuda',)
        )
        if d['has_cupy']:
            cupy_only = ctx.run_udf(
                udf=DebugDeviceUDF(backends=('cupy', 'numpy')),
                dataset=hdf5_ds_1,
                backends=('cupy', 'cuda')
            )
        else:
            with pytest.raises(RuntimeError):
                cupy_only = ctx.run_udf(
                    udf=DebugDeviceUDF(backends=('cupy', 'numpy')),
                    dataset=hdf5_ds_1,
                    backends=('cupy',)
                )
            cupy_only = None

    numpy_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('numpy',)),
        dataset=hdf5_ds_1
    )

    assert np.allclose(
        hybrid.mask_0.raw_data,
        expected
    )
    if cudas:
        assert np.all(cuda_only['device_class'].data == 'cuda')
        if cupy_only is not None:
            assert np.all(cupy_only['device_class'].data == 'cuda')
    np_only = numpy_only['device_class'].data
    assert np.all((np_only == 'cpu') + (np_only == 'cuda'))


@pytest.mark.slow
def test_start_local_cpuonly(hdf5_ds_1):
    # We don't use all since that might be too many
    cpus = (0, 1)
    hdf5_ds_1.set_num_cores(len(cpus))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    spec = cluster_spec(cpus=cpus, cudas=[], has_cupy=False)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        udf_res = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
        # No CuPy resources
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy',)), dataset=hdf5_ds_1)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )
    found = {}

    for val in udf_res['device_id'].data[0].values():
        print(val)
        # no CUDA
        assert val["cuda"] is None
        found[val["cpu"]] = True

    for val in udf_res['backend'].data[0].values():
        print(val)
        # no CUDA
        assert 'numpy' in val

    # Each CPU got work. We have to see if this
    # actually works always since this depends on the scheduler behavior
    assert set(found.keys()) == set(cpus)

    assert np.all(udf_res['device_class'].data == 'cpu')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))


@pytest.mark.slow
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.skipif(not has_cupy(), reason="No functional CuPy")
def test_start_local_cupyonly(hdf5_ds_1):
    cudas = detect()['cudas']
    # Make sure we have enough partitions
    hdf5_ds_1.set_num_cores(len(cudas))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    spec = cluster_spec(cpus=(), cudas=cudas, has_cupy=True)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        # Uses ApplyMasksUDF, which supports CuPy
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        udf_res = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
        # No CPU compute resources since no CPU workers
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('numpy',)), dataset=hdf5_ds_1)
        cuda_res = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda',)), dataset=hdf5_ds_1)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )

    found = {}

    for val in udf_res['device_id'].data[0].values():
        print(val)
        # no CPU
        assert val["cpu"] is None
        # Register which GPUs got work
        found[val["cuda"]] = True

    for val in udf_res['device_id'].data[0].values():
        print(val)
        # no CPU
        assert val["cpu"] is None
        # Register which GPUs got work
        found[val["cuda"]] = True

    for val in cuda_res['device_id'].data[0].values():
        print(val)
        # no CPU
        assert val["cpu"] is None
        # Register which GPUs got work
        found[val["cuda"]] = True

    for val in udf_res['backend'].data[0].values():
        # use CuPy
        print(val)
        assert 'cupy' in val

    for val in cuda_res['backend'].data[0].values():
        # no CuPy, i.e. NumPy
        print(val)
        assert 'numpy' in val

    # Test if each GPU got work. We have to see if this
    # actually works always since this depends on the scheduler behavior
    assert set(found.keys()) == set(cudas)

    assert np.all(udf_res['device_class'].data == 'cuda')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))


def test_cluster_spec_cpu_int():
    int_spec = cluster_spec(cpus=4, cudas=tuple(), has_cupy=True)
    range_spec = cluster_spec(cpus=range(4), cudas=tuple(), has_cupy=True)
    assert range_spec == int_spec


def test_cluster_spec_cudas_int():
    n_cudas = 4
    cuda_spec = cluster_spec(
        cpus=tuple(),
        cudas=n_cudas,
        has_cupy=True,
    )
    print(cuda_spec)
    num_cudas = 0
    for spec in cuda_spec.values():
        num_cudas += spec.get('options', {}).get('resources', {}).get('CUDA', 0)
    assert num_cudas == n_cudas


def test_cluster_spec_1():
    cuda_info = {
        0: {'mem_info': (2*DEFAULT_RAM_PER_WORKER, 2*DEFAULT_RAM_PER_WORKER), },
        1: {'mem_info': (DEFAULT_RAM_PER_WORKER, DEFAULT_RAM_PER_WORKER), },
        2: {'mem_info': (DEFAULT_RAM_PER_WORKER//2, DEFAULT_RAM_PER_WORKER//2), },
        3: {'mem_info': (DEFAULT_RAM_PER_WORKER, DEFAULT_RAM_PER_WORKER), },
    }
    spec = cluster_spec(
        cpus=6, cudas=(0, ), has_cupy=True, cuda_info=cuda_info,
        ram_per_cuda_worker=DEFAULT_RAM_PER_WORKER, max_workers_per_cuda=1
    )
    per_resource, per_cpu_id, per_cuda_id = characterize(spec)
    assert per_resource == {
        'compute': 6,
        'CPU': 6,
        'CUDA': 1,
        'ndarray': 6,
        'service': 1,
    }
    assert per_cpu_id == {
        # 0 is CUDA
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
    }
    assert per_cuda_id == {0: 1}


def test_cluster_spec_2():
    cuda_info = {
        0: {'mem_info': (2*DEFAULT_RAM_PER_WORKER, 2*DEFAULT_RAM_PER_WORKER), },
        1: {'mem_info': (DEFAULT_RAM_PER_WORKER, DEFAULT_RAM_PER_WORKER), },
        2: {'mem_info': (DEFAULT_RAM_PER_WORKER//2, DEFAULT_RAM_PER_WORKER//2), },
        3: {'mem_info': (DEFAULT_RAM_PER_WORKER*10, DEFAULT_RAM_PER_WORKER*10), },
    }
    spec = cluster_spec(
        cpus=6, cudas=(0, 1, 2, 3, 2), has_cupy=True, cuda_info=cuda_info,
        ram_per_cuda_worker=DEFAULT_RAM_PER_WORKER, max_workers_per_cuda=7
    )
    per_resource, per_cpu_id, per_cuda_id = characterize(spec)
    assert per_resource == {
        'compute': 12,
        'CPU': 6,
        'CUDA': 12,
        'ndarray': 12,
        'service': 1,
    }
    assert per_cuda_id == {
        0: 2,  # RAM
        1: 1,  # RAM
        2: 2,  # We forced two workers with cudas=...
        3: 7,  # max_workers
    }
    assert per_cpu_id == {}  # All are hybrid workers assigned to CUDA devices


def test_cluster_spec_3():
    cuda_info = {
        0: {'mem_info': (2*DEFAULT_RAM_PER_WORKER, 2*DEFAULT_RAM_PER_WORKER), },
        1: {'mem_info': (DEFAULT_RAM_PER_WORKER, DEFAULT_RAM_PER_WORKER), },
        2: {'mem_info': (DEFAULT_RAM_PER_WORKER//2, DEFAULT_RAM_PER_WORKER//2), },
        3: {'mem_info': (DEFAULT_RAM_PER_WORKER*10, DEFAULT_RAM_PER_WORKER*10), },
    }
    spec = cluster_spec(
        cpus=(1, 3, 5, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 14, 14),
        cudas=(0, 1, 2, 3, 2),
        has_cupy=False, cuda_info=cuda_info,
        ram_per_cuda_worker=DEFAULT_RAM_PER_WORKER, max_workers_per_cuda=7,
        num_service=3
    )
    per_resource, per_cpu_id, per_cuda_id = characterize(spec)
    # No hybrid workers since no CuPy
    assert per_resource == {
        'compute': 27,
        'CPU': 15,
        'CUDA': 12,
        'ndarray': 15,
        'service': 3,
    }
    assert per_cuda_id == {
        0: 2,  # RAM
        1: 1,  # RAM
        2: 2,  # We forced two workers with cudas=...
        3: 7,  # max_workers
    }
    assert per_cpu_id == {
        1: 1,
        3: 1,
        5: 1,
        7: 1,
        8: 1,
        9: 1,
        10: 1,
        11: 1,
        12: 1,
        13: 1,
        14: 5,
    }


@pytest.mark.slow
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
def test_start_local_cudaonly(hdf5_ds_1):
    cudas = detect()['cudas']
    # Make sure we have enough partitions
    hdf5_ds_1.set_num_cores(len(cudas))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]

    spec = cluster_spec(cpus=(), cudas=cudas, has_cupy=False)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        udf_res = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda', )), dataset=hdf5_ds_1)
        # No CPU compute resources
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('numpy',)), dataset=hdf5_ds_1)
        # No ndarray (CuPy) resources
        with pytest.raises(RuntimeError):
            _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy',)), dataset=hdf5_ds_1)

    found = {}

    for val in udf_res['device_id'].data[0].values():
        print(val)
        # no CPU
        assert val["cpu"] is None
        # Register which GPUs got work
        found[val["cuda"]] = True

    for val in udf_res['backend'].data[0].values():
        print(val)
        # CUDA, but no CuPy, i.e. use NumPy
        assert 'numpy' in val

    # Test if each GPU got work. We have to see if this
    # actually works always since this depends on the scheduler behavior
    assert set(found.keys()) == set(cudas)

    assert np.all(udf_res['device_class'].data == 'cuda')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))


@pytest.mark.slow
def test_preload(hdf5_ds_1):
    # We don't use all since that might be too many
    cpus = (0, 1)
    hdf5_ds_1.set_num_cores(len(cpus))

    class CheckEnvUDF(NoOpUDF):
        def process_tile(self, tile):
            assert os.environ['LT_TEST_1'] == 'hello'
            assert os.environ['LT_TEST_2'] == 'world'

    preloads = (
        "import os; os.environ['LT_TEST_1'] = 'hello'",
        "import os; os.environ['LT_TEST_2'] = 'world'",
    )

    spec = cluster_spec(cpus=cpus, cudas=[], has_cupy=False, preload=preloads)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        ctx.run_udf(udf=CheckEnvUDF(), dataset=hdf5_ds_1)


@pytest.mark.slow
def test_use_plain_dask(hdf5_ds_1):
    # We deactivate the resource scheduling and run on a plain dask cluster
    hdf5_ds_1.set_num_cores(2)
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)
    with dd.LocalCluster(n_workers=2, threads_per_worker=1) as cluster:
        client = dd.Client(cluster, set_as_default=False)
        try:
            executor = DaskJobExecutor(client=client)
            ctx = api.Context(executor=executor)
            analysis = ctx.create_mask_analysis(
                dataset=hdf5_ds_1, factories=[lambda: mask]
            )
            results = ctx.run(analysis)
            udf_res = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
            # Requesting CuPy, which is not available
            with pytest.raises(RuntimeError):
                _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy',)), dataset=hdf5_ds_1)
        finally:
            # to fix "distributed.client - ERROR - Failed to reconnect to scheduler after 10.00 seconds, closing client"  # NOQA
            client.close()

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )

    for val in udf_res['device_id'].data[0].values():
        print(val)
        # no CUDA
        assert val["cuda"] is None
        # Default without worker setup
        assert val["cpu"] == 0

    for val in udf_res['backend'].data[0].values():
        print(val)
        # no CUDA
        assert 'numpy' in val

    assert np.all(udf_res['device_class'].data == 'cpu')
    assert np.allclose(udf_res['on_device'].data, data.sum(axis=(0, 1)))
