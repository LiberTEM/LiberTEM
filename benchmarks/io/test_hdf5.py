import os
import timeit

import numpy as np
import pytest
import h5py

from libertem.api import Context
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.base import UDF
from libertem.executor.inline import InlineJobExecutor

from utils import drop_cache, warmup_cache


@pytest.fixture(scope='module')
def chunked_emd(tmpdir_factory):
    lt_ctx = Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('hdf5_chunked_data')
    filename = os.path.join(datadir, 'chunked.emd')

    chunks = (32, 32, 128, 128)

    with h5py.File(filename, mode="w") as f:
        f.attrs.create('version_major', 0)
        f.attrs.create('version_minor', 2)

        f.create_group('experimental/science_data')
        group = f['experimental/science_data']
        group.attrs.create('emd_group_type', 1)

        data = np.ones((256, 256, 128, 128), dtype=np.float32)

        group.create_dataset(name='data', data=data, chunks=chunks)
        group.create_dataset(name='dim1', data=range(256))
        group['dim1'].attrs.create('name', b'dim1')
        group['dim1'].attrs.create('units', b'units1')
        group.create_dataset(name='dim2', data=range(256))
        group['dim2'].attrs.create('name', b'dim2')
        group['dim2'].attrs.create('units', b'units2')
        group.create_dataset(name='dim3', data=range(128))
        group['dim3'].attrs.create('name', b'dim3')
        group['dim3'].attrs.create('units', b'units3')
        group.create_dataset(name='dim4', data=range(128))
        group['dim4'].attrs.create('name', b'dim4')
        group['dim4'].attrs.create('units', b'units4')
        f.close()

    yield lt_ctx.load("auto", path=filename, ds_path="/experimental/science_data/data")


class TestUseSharedExecutor:
    @pytest.mark.benchmark(
        group="io"
    )
    @pytest.mark.parametrize(
        "drop", ("cold_cache", "warm_cache")
    )
    @pytest.mark.parametrize(
        "context", ("dist", "inline")
    )
    def test_mask(self, benchmark, drop, shared_dist_ctx, lt_ctx, context, chunked_emd):
        if context == 'dist':
            ctx = shared_dist_ctx
        elif context == 'inline':
            ctx = lt_ctx
        else:
            raise ValueError

        ds = chunked_emd

        def mask():
            return np.ones(ds.shape.sig, dtype=bool)

        udf = ApplyMasksUDF(mask_factories=[mask], backends=('numpy', ))

        # warmup executor
        ctx.run_udf(udf=udf, dataset=ds)

        if drop == "cold_cache":
            drop_cache([ds.path])
        elif drop == "warm_cache":
            warmup_cache([ds.path])
        else:
            raise ValueError("bad param")

        benchmark.pedantic(
            ctx.run_udf, kwargs=dict(udf=udf, dataset=ds),
            warmup_rounds=0,
            rounds=1,
            iterations=1
        )


class TimeoutSumUDF(UDF):
    def __init__(self, max_duration, start=None):
        if start is None:
            start = timeit.default_timer()
        super().__init__(max_duration=max_duration, start=start)

    def get_preferred_input_dtype(self):
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        return {
            "sum": self.buffer(kind="single", dtype=np.int64)
        }

    def merge(self, dest, src):
        dest["sum"][:] += src["sum"][:]

    def process_tile(self, tile):
        if timeit.default_timer() > self.params.start + self.params.max_duration:
            raise TimeoutError
        self.results.sum[:] += np.sum(tile)


def random_hdf5_params():
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    factors = np.concatenate((primes, [2, 2])).astype(int)
    n_dims = np.random.randint(2, 8)
    axes = np.ones(n_dims, dtype=int)

    for f in factors:
        axis = np.random.randint(0, n_dims)
        axes[axis] *= f

    sig_dims = np.random.randint(1, n_dims)

    chunking = np.zeros(n_dims, dtype=int)

    while np.prod(chunking) < 4096 or np.prod(chunking) > 128*1024*1024:
        for d in range(n_dims):
            # Distribution with heavy skew towards small chunk sizes
            chunking[d] = int(np.exp(np.random.uniform(0, np.log(axes[d]))))

    return (factors, axes, chunking, sig_dims)


@pytest.mark.slow
@pytest.mark.parametrize(
    "hdf5_params", [random_hdf5_params() for i in range(10)]
)
def test_stress(tmpdir_factory, lt_ctx, hdf5_params):
    (factors, axes, chunking, sig_dims) = hdf5_params
    size = np.prod(factors)
    timer = timeit.default_timer
    print(f"START axes={axes}, chunking={chunking}, sig_dims={sig_dims}, "
          f"size={size}, chunk_size={np.prod(chunking)}")

    rand = np.random.randint(32, size=size, dtype=np.uint16)

    start = timer()
    result = np.sum(rand)
    reference_sum_time = timer() - start

    print(f'Time for in-memory sum: {reference_sum_time}')

    tmpdir = tmpdir_factory.mktemp('hdf5_fuzzed_chunks')
    filename = os.path.join(tmpdir, 'file.hdf5')
    print("Creating HDF5...")
    start = timer()
    with h5py.File(filename, mode="w") as f:
        f.create_dataset(
            "data",
            shape=tuple(axes),
            chunks=tuple(chunking),
            compression="gzip",
            dtype="uint16",
            data=rand.reshape(axes)
        )
    create_hdf5_time = timer() - start
    print(f'Time for creating HDF5: {create_hdf5_time}')

    print("Reading and summing HDF5 directly...")
    start = timer()
    with h5py.File(filename, mode="r") as f:
        hdf5_sum = np.sum(f["data"])
    sum_hdf5_time = timer() - start
    assert hdf5_sum == result
    print(f'Time for reading and summing HDF5 directly: {sum_hdf5_time}')

    print("Reading and summing HDF5 using LiberTEM UDF...")
    # Here we should test if LiberTEM issued a warning about the chunking
    ds = lt_ctx.load("hdf5", path=filename, sig_dims=sig_dims)
    # TODO see above
    warned = False
    timeout = False
    limit = sum_hdf5_time*3
    start = timer()
    try:
        lt_result = lt_ctx.run_udf(udf=TimeoutSumUDF(max_duration=limit), dataset=ds)
    except TimeoutError:
        timeout = True
    sum_lt_time = timer() - start
    if sum_lt_time > limit:
        timeout = True
    print(f'Time for reading and summing HDF5 using LiberTEM inline '
          f'executor: {sum_lt_time}, timeout={timeout}')

    if timeout:
        if warned:
            print(f"WARN axes={axes}, chunking={chunking}, sig_dims={sig_dims}")
        else:
            print(f"FAIL axes={axes}, chunking={chunking}, sig_dims={sig_dims}")
            assert False
    else:
        assert lt_result["sum"].data == result
        if not warned:
            print(f"SUCCESS axes={axes}, chunking={chunking}, sig_dims={sig_dims}")
        else:
            print(f"FALSE_POSITIVE axes={axes}, chunking={chunking}, sig_dims={sig_dims}")
            assert False
