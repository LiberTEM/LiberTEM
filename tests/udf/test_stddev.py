import pytest
import numpy as np
import numba
import sparse

from libertem.udf.stddev import StdDevUDF, run_stddev, process_tile, merge
from libertem.io.dataset.memory import MemoryDataSet
from libertem.common.sparse import as_format, NUMPY
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.utils.devices import detect

from utils import _mk_random


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "use_roi", [True, False]
)
@pytest.mark.parametrize(
    "dtype", [np.float32, np.complex128]
)
@pytest.mark.parametrize(
    "use_numba", [True, False]
)
@pytest.mark.parametrize(
    "sparse_input", [True, False]
)
@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_stddev(lt_ctx, delayed_ctx, use_roi, dtype, use_numba, sparse_input, backend):
    """
    Test variance, standard deviation, sum of frames, and mean computation
    implemented in udf/stddev.py

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    if sparse_input and backend == 'cupy':
        pytest.skip("No sparse input with CuPy yet")
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        if sparse_input:
            data = sparse.random((30, 3, 516))
        else:
            data = _mk_random(size=(30, 3, 516), dtype=dtype)
        dataset = MemoryDataSet(data=data, tileshape=(3, 2, 257),
                                num_partitions=2, sig_dims=2)
        if use_roi:
            roi = np.random.choice([True, False], size=dataset.shape.nav)
            res = run_stddev(lt_ctx, dataset, roi=roi, use_numba=use_numba)
            res_delayed = run_stddev(delayed_ctx, dataset, roi=roi, use_numba=use_numba)
        else:
            roi = np.ones(dataset.shape.nav, dtype=bool)
            res = run_stddev(lt_ctx, dataset, use_numba=use_numba)
            res_delayed = run_stddev(delayed_ctx, dataset, use_numba=use_numba)

        assert 'sum' in res
        assert 'num_frames' in res
        assert 'var' in res
        assert 'mean' in res
        assert 'std' in res

        N = np.count_nonzero(roi)
        assert res['num_frames'] == N  # check the total number of frames
        assert res_delayed['num_frames'] == N

        print('sum')
        refsum = as_format(np.sum(data[roi], axis=0), NUMPY)
        print(res['sum'])
        print(refsum)
        print(res['sum'] - refsum)
        assert np.allclose(res['sum'], refsum)  # check sum of frames
        assert np.allclose(res_delayed['sum'], refsum)

        ref_mean = as_format(np.mean(data[roi], axis=0), NUMPY)
        assert np.allclose(res['mean'], ref_mean)  # check mean
        assert np.allclose(res_delayed['mean'], ref_mean)

        refvar = as_format(np.var(data[roi], axis=0), NUMPY)
        print('var')
        print(res['var'])
        print(refvar)
        print(refvar - res['var'])
        assert np.allclose(refvar, res['var'])  # check variance
        assert np.allclose(refvar, res_delayed['var'])

        refstd = as_format(np.std(data[roi], axis=0), NUMPY)
        assert np.allclose(refstd, res['std'])  # check standard deviation
        assert np.allclose(refstd, res_delayed['std'])
    finally:
        set_use_cpu(0)


@pytest.mark.slow
def test_stddev_fuzz(concurrent_ctx):
    for i in range(100):
        total = np.random.randint(1, 20)
        tile = np.random.randint(1, total + 1)
        n_part = np.random.randint(1, total//tile + 1)
        print(total, tile, n_part)
        data = np.random.randn(total, 512, 512) + 1j*np.random.randn(total, 512, 512)
        ds = concurrent_ctx.load(
            'memory',
            data=data,
            tileshape=(tile, 512, 512),
            num_partitions=n_part,
            sig_dims=2,
        )
        var = np.var(data, axis=0)
        res = concurrent_ctx.run_udf(dataset=ds, udf=StdDevUDF())
        assert np.allclose(res['var'], var)


@numba.njit(boundscheck=True, cache=True)
def _stability_workhorse(data):
    # This function imitates the calculation flow of a single pixel for a very big dataset.
    # The partition and tile structure is given by the shape of data
    # data.shape[0]: partitions
    # data.shape[1]: tiles per partition
    # data.shape[2]: frames per tile
    # data.shape[3]: mock signal shape, can be 1
    # The test uses numba since this allows efficient calculations for small units
    # of data. Running it with JIT disabled is very, very slow!
    s = np.zeros(1, dtype=np.float32)
    varsum = np.zeros(1, dtype=np.float32)
    N = 0
    partitions = data.shape[0]
    tiles = data.shape[1]
    frames = data.shape[2]
    for partition in range(partitions):
        partition_sum = np.zeros(1, dtype=np.float32)
        partition_varsum = np.zeros(1, dtype=np.float32)
        partition_N = 0
        for tile in range(tiles):
            if partition_N == 0:
                partition_sum[:] = np.sum(data[partition, tile])
                partition_varsum[:] = np.var(data[partition, tile]) * frames
                partition_N = frames
            else:
                partition_N = process_tile(
                    tile=data[partition, tile],
                    n_0=partition_N,
                    sum_inout=partition_sum,
                    varsum_inout=partition_varsum
                )
        N = merge(
            dest_n=N,
            dest_sum=s,
            dest_varsum=varsum,
            src_n=partition_N,
            src_sum=partition_sum,
            src_varsum=partition_varsum
        )
    return N, s, varsum


# This shouldn't be @pytest.mark.numba
# since the calculation will take forever
# with JIT disabled
@pytest.mark.slow
def test_stability(lt_ctx):
    data = _mk_random(size=(1024, 1024, 8, 1), dtype="float32")

    N, s, varsum = _stability_workhorse(data)

    assert N == np.prod(data.shape)
    assert np.allclose(data.sum(), s)
    assert np.allclose(data.var(ddof=N-1), varsum)
