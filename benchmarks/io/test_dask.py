import pathlib
import importlib

import pytest
import numpy as np
import dask.array as da

from libertem.udf.base import NoOpUDF
from libertem.udf.sum import SumUDF

# Load the dask dataset utils from the tests folder
# This is really ugly but we're outside of the package structure!
basedir = pathlib.Path(__file__).parent
location = (basedir / "../../tests/io/datasets/utils_dask.py").resolve()
spec = importlib.util.spec_from_file_location("utils_dask", location)
utils_dask = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils_dask)
_mk_dask_from_delayed = utils_dask._mk_dask_from_delayed


# (100, 100, 1216, 1216) by default
__nblock_params = ((100, 1, 1, 1),
                   (50, 1, 1, 1),
                   (25, 1, 1, 1),
                   (10, 1, 1, 1),
                   (4, 1, 1, 1),
                   (100, 25, 1, 1),
                   (100, 4, 1, 1),
                   (20, 1, 2, 2))


@pytest.mark.benchmark(
    group="dask nav chunking",
)
@pytest.mark.parametrize(
    "nblocks",
    __nblock_params,
    ids=tuple(f'nblocks: {c}' for c in __nblock_params)
)
def test_dask_nav_chunking(shared_dist_ctx_globaldask, large_raw_file, nblocks, benchmark):
    _run_benchmark(shared_dist_ctx_globaldask, large_raw_file, nblocks, benchmark,
                preserve_dim=True, min_size=None)


def _run_benchmark(shared_dist_ctx, large_raw_file, nblocks, benchmark, preserve_dim, min_size):
    filename, shape, dtype = large_raw_file

    chunking = tuple(dim // nb for dim, nb in zip(shape, nblocks))
    print(chunking)

    d_arr = _mk_dask_from_delayed(shape=shape,
                                  dtype=dtype,
                                  chunking=chunking,
                                  filename=filename)

    udf = NoOpUDF()
    ds = shared_dist_ctx.load(
        'dask',
        d_arr,
        sig_dims=2,
        preserve_dimensions=preserve_dim,
        min_size=min_size
    )

    benchmark.pedantic(
        shared_dist_ctx.run_udf,
        kwargs=dict(
            dataset=ds,
            udf=udf,
        ),
        warmup_rounds=0,
        rounds=5,
        iterations=1,
    )


class TestDaskArray:
    @pytest.mark.benchmark(
        group="dask from_array",
    )
    @pytest.mark.parametrize(
        'method', ('from_array', 'native', 'delayed')
    )
    @pytest.mark.parametrize(
        "bench", ('libertem', 'mmap')
    )
    @pytest.mark.parametrize(
        "typ", ('uint16', 'float32')
    )
    def test_inline(self, lt_ctx_fast, medium_raw, medium_raw_float32, method, bench, typ, benchmark):
        ctx = lt_ctx_fast
        if typ == 'uint16':
            medium_raw = medium_raw
        elif typ == 'float32':
            medium_raw = medium_raw_float32
        else:
            raise ValueError()
        ds = _mk_ds(method=method, ctx=ctx, raw_ds=medium_raw)
        if bench == 'libertem':
            _libertem_bench(ctx=ctx, ds=ds, benchmark=benchmark)
        elif bench == 'mmap':
            _mmap_bench(ds=ds, benchmark=benchmark)
        else:
            raise ValueError()

    @pytest.mark.benchmark(
        group="dask from_array",
    )
    @pytest.mark.parametrize(
        'method', ('from_array', 'native', 'delayed')
    )
    @pytest.mark.parametrize(
        "bench", ('libertem', 'dask.array')
    )
    @pytest.mark.parametrize(
        "typ", ('uint16', 'float32')
    )
    def test_concurrent(self, concurrent_ctx, medium_raw, medium_raw_float32, method, bench, typ, benchmark):
        ctx = concurrent_ctx
        if typ == 'uint16':
            medium_raw = medium_raw
        elif typ == 'float32':
            medium_raw = medium_raw_float32
        else:
            raise ValueError()
        ds = _mk_ds(method=method, ctx=ctx, raw_ds=medium_raw)
        if bench == 'libertem':
            _libertem_bench(ctx=ctx, ds=ds, benchmark=benchmark)
        elif bench == 'dask.array':
            _dask_bench(ctx=ctx, ds=ds, benchmark=benchmark)
        else:
            raise ValueError()

    @pytest.mark.benchmark(
        group="dask from_array",
    )
    @pytest.mark.parametrize(
        'method', ('from_array', 'native', 'delayed')
    )
    @pytest.mark.parametrize(
        "bench", ('libertem', 'dask.array')
    )
    @pytest.mark.parametrize(
        "typ", ('uint16', 'float32')
    )
    def test_dist(self, shared_dist_ctx_globaldask, medium_raw, medium_raw_float32, method, bench, typ, benchmark):
        # This one has to run separately since using the shared_dist_ctx_globaldask fixture
        # makes all Dask operations use the distributed scheduler, making it perform poorly
        # with the inline and concurrent executor for LiberTEM
        ctx = shared_dist_ctx_globaldask
        if typ == 'uint16':
            medium_raw = medium_raw
        elif typ == 'float32':
            medium_raw = medium_raw_float32
        else:
            raise ValueError()
        ds = _mk_ds(method=method, ctx=ctx, raw_ds=medium_raw)
        if bench == 'libertem':
            _libertem_bench(ctx=ctx, ds=ds, benchmark=benchmark)
        elif bench == 'dask.array':
            _dask_bench(ctx=ctx, ds=ds, benchmark=benchmark)
        else:
            raise ValueError()


def _mk_ds(method, ctx, raw_ds):
    filename = raw_ds._path
    shape = tuple(raw_ds.shape)
    dtype = raw_ds.dtype
    if method == 'from_array':
        arr = da.from_array(np.memmap(filename, shape=shape, dtype=dtype, mode='r'))
        ds = ctx.load('dask', arr, sig_dims=2)
    elif method == 'native':
        ds = raw_ds
    elif method == 'delayed':
        arr = _mk_dask_from_delayed(
            shape=shape,
            dtype=dtype,
            chunking=(4, -1, 64, -1),
            filename=filename
        )
        ds = ctx.load('dask', arr, sig_dims=2)
    else:
        raise ValueError

    return ds


def _libertem_bench(ctx, ds, benchmark):
    benchmark(
        ctx.run_udf,
        dataset=ds,
        udf=SumUDF()
    )


def _dask_bench(ctx, ds, benchmark):
    if hasattr(ds, '_array') and isinstance(ds._array, da.Array):
        benchmark(ds._array.sum(axis=(0, 1)).compute)
    else:
        pytest.skip("Not a Dask array")


def _mmap_bench(ds, benchmark):
    if not hasattr(ds, '_array') and hasattr(ds, '_dtype'):
        filename = ds._path
        shape = tuple(ds.shape)
        dtype = ds.dtype
        arr = np.memmap(filename, shape=shape, dtype=dtype, mode='r')

        benchmark(arr.sum, axis=(0, 1))
    else:
        pytest.skip("Not a RAW dataset")
