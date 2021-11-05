import pathlib
import importlib
import pytest

from libertem.udf.base import NoOpUDF

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
                   (25, 1, 1, 1),
                   (10, 1, 1, 1),
                   (4, 1, 1, 1),
                   (100, 25, 1, 1))


@pytest.mark.benchmark(
    group="dask nav chunking",
)
@pytest.mark.parametrize(
    "nblocks",
    __nblock_params,
    ids=tuple(f'nblocks: {c}' for c in __nblock_params)
)
def test_dask_nav_chunking(shared_dist_ctx, large_raw_file, nblocks, benchmark):
    _run_benchmark(shared_dist_ctx, large_raw_file, nblocks, benchmark,
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
