import numpy as np
import pytest

from libertem import api
from utils import _naive_mask_apply, _mk_random
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.utils.devices import detect, has_cupy

from utils import DebugDeviceUDF


@pytest.mark.functional
def test_start_local_default(hdf5_ds_1):
    mask = _mk_random(size=(16, 16))
    d = detect()
    cudas = d['cudas']
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with api.Context() as ctx:
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
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
                    backends=('cupy',)
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
    assert np.all(numpy_only['device_class'].data == 'cpu')


@pytest.mark.functional
def test_start_local_cpuonly(hdf5_ds_1):
    # We don't use all since that might be too many
    cpus = (0, 1)
    hdf5_ds_1.set_num_cores(len(cpus))
    mask = _mk_random(size=(16, 16))
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    spec = cluster_spec(cpus=cpus, cudas=(), has_cupy=False)
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


@pytest.mark.functional
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


@pytest.mark.functional
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
