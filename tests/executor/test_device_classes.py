import numpy as np
import pytest
import distributed as dd

from libertem import api
from utils import _naive_mask_apply, _mk_random
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.executor.pipelined import PipelinedExecutor
from libertem.utils.devices import detect, has_cupy

from utils import DebugDeviceUDF


@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_with_cupy(
    hdf5_ds_1,
    local_cluster_ctx: api.Context,
    pipelined_ctx: api.Context,
    executor
):
    """
    This test only runs if GPUs are detected and cupy is available, and
    makes sure the UDFs run on workers with correct device classes.
    """
    if executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'local_cluster':
        ctx = local_cluster_ctx
    else:
        raise ValueError(f"invalid executor {executor}")
    mask = _mk_random(size=(16, 16))
    d = detect()
    if not d['cudas'] or not d['has_cupy']:
        pytest.skip('this test only runs with a working cupy installation')

    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    analysis = ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )

    # Based on ApplyMasksUDF, which is CuPy-enabled
    hybrid = ctx.run(analysis)
    _ = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'cuda', 'numpy')), dataset=hdf5_ds_1)
    cuda_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('cuda', 'numpy')),
        dataset=hdf5_ds_1,
        backends=('cuda',)
    )

    cupy_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('cupy', 'numpy')),
        dataset=hdf5_ds_1,
        backends=('cupy',)
    )
    numpy_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('numpy',)),
        dataset=hdf5_ds_1
    )

    assert np.allclose(
        hybrid.mask_0.raw_data,
        expected
    )
    assert np.all(cuda_only['device_class'].data == 'cuda')
    assert np.all(cupy_only['device_class'].data == 'cuda')
    assert np.all(numpy_only['device_class'].data == 'cpu')


@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_cuda_no_cupy(
    hdf5_ds_1,
    local_cluster_ctx: api.Context,
    pipelined_ctx: api.Context,
    executor
):
    """
    This test only runs if GPUs are detected, but cupy is not available, and
    makes sure the UDFs run on workers with correct device classes.
    """
    if executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'local_cluster':
        ctx = local_cluster_ctx
    else:
        raise ValueError(f"invalid executor {executor}")
    mask = _mk_random(size=(16, 16))
    d = detect()
    if not d['cudas'] or d['has_cupy']:
        pytest.skip('this test only runs with GPU available, but without cupy')

    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    analysis = ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )

    # Based on ApplyMasksUDF, which is CuPy-enabled
    hybrid = ctx.run(analysis)
    _ = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'cuda', 'numpy')), dataset=hdf5_ds_1)
    cuda_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('cuda', 'numpy')),
        dataset=hdf5_ds_1,
        backends=('cuda',)
    )
    with pytest.raises(RuntimeError):
        ctx.run_udf(
            udf=DebugDeviceUDF(backends=('cupy', 'numpy')),
            dataset=hdf5_ds_1,
            backends=('cupy',)
        )
    numpy_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('numpy',)),
        dataset=hdf5_ds_1
    )

    assert np.allclose(
        hybrid.mask_0.raw_data,
        expected
    )
    assert np.all(cuda_only['device_class'].data == 'cuda')
    assert np.all(numpy_only['device_class'].data == 'cpu')


@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_no_gpu(
    hdf5_ds_1,
    local_cluster_ctx: api.Context,
    pipelined_ctx: api.Context,
    executor
):
    """
    This test only runs if no GPUs are detected, and makes sure the UDFs run on workers
    with correct device classes
    """
    if executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'local_cluster':
        ctx = local_cluster_ctx
    else:
        raise ValueError(f"invalid executor {executor}")
    mask = _mk_random(size=(16, 16))
    d = detect()
    if d['cudas']:
        pytest.skip('this test only runs without GPUs')

    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    analysis = ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask]
    )

    # Based on ApplyMasksUDF, which is CuPy-enabled
    hybrid = ctx.run(analysis)
    _ = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cuda', 'numpy')), dataset=hdf5_ds_1)
    _ = ctx.run_udf(udf=DebugDeviceUDF(backends=('cupy', 'cuda', 'numpy')), dataset=hdf5_ds_1)

    numpy_only = ctx.run_udf(
        udf=DebugDeviceUDF(backends=('numpy',)),
        dataset=hdf5_ds_1
    )
    assert np.allclose(
        hybrid.mask_0.raw_data,
        expected
    )
    assert np.all(numpy_only['device_class'].data == 'cpu')


@pytest.mark.slow
@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_limit_to_cpus(hdf5_ds_1, executor):
    """
    This test explicitly starts executors with only CPU workers, and checks that
    UDFs run on workers with the correct device classes.
    """
    # We don't use all since that might be too many
    # create a Context with only cuda workers:
    cpus = (0, 1)
    if executor == 'pipelined':
        ctx = api.Context.make_with('pipelined', gpus=0, cpus=cpus)
    elif executor == 'local_cluster':
        ctx = api.Context.make_with('dask', gpus=0, cpus=cpus)
    else:
        raise ValueError(f"invalid executor {executor}")
    hdf5_ds_1.set_num_cores(len(cpus))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with ctx:
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
@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_only_cupy(hdf5_ds_1, executor):
    cudas = detect()['cudas']
    # create a Context with only cuda workers:
    if executor == 'pipelined':
        ctx = api.Context.make_with('pipelined', gpus=cudas, cpus=0)
    elif executor == 'local_cluster':
        ctx = api.Context.make_with('dask', gpus=cudas, cpus=0)
    else:
        raise ValueError(f"invalid executor {executor}")
    # Make sure we have enough partitions
    hdf5_ds_1.set_num_cores(len(cudas))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with ctx:
        # Uses ApplyMasksUDF, which supports CuPy
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)
        udf_res = ctx.run_udf(udf=DebugDeviceUDF(), dataset=hdf5_ds_1)
        # No CPU compute resources
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


@pytest.mark.slow
@pytest.mark.skipif(not detect()['cudas'], reason="No CUDA devices")
@pytest.mark.parametrize('executor', ['pipelined', 'local_cluster'])
def test_device_classes_limit_to_cuda(hdf5_ds_1, executor):
    cudas = detect()['cudas']
    # create a Context with only cuda workers (force has_cupy=False):
    if executor == 'pipelined':
        spec = PipelinedExecutor.make_spec(cpus=[], cudas=cudas, has_cupy=False)
        exc = PipelinedExecutor(spec=spec)
    elif executor == 'local_cluster':
        spec = cluster_spec(cpus=(), cudas=cudas, has_cupy=False)
        exc = DaskJobExecutor.make_local(spec=spec)
    else:
        raise ValueError(f"invalid executor {executor}")
    ctx = api.Context(executor=exc)
    # Make sure we have enough partitions
    hdf5_ds_1.set_num_cores(len(cudas))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]

    with ctx:
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
