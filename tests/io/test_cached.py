import os
import pickle

import numpy as np
import pytest
import dask.distributed as dd

from libertem.io.dataset.cached import CachedDataSet, LRUCacheStrategy
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape


@pytest.fixture
def default_cached_ds(tmpdir_factory, default_raw, lt_ctx):
    datadir = tmpdir_factory.mktemp('cached_ds_directory')
    strategy = LRUCacheStrategy(capacity=1024*1024*1024)
    ds = CachedDataSet(
        source_ds=default_raw,
        cache_path=datadir,
        strategy=strategy,
    )
    ds = ds.initialize(executor=lt_ctx.executor)
    yield ds


@pytest.fixture(scope='module', autouse=True)
def client_in_background():
    # A running Dask client can introduce a timing issue
    # between automatic closing of a numpy.memmap object and renaming
    # the underlying file
    with dd.LocalCluster() as cluster:
        client = dd.Client(cluster, set_as_default=False)
        yield
        # to fix "distributed.client - ERROR - Failed to reconnect to scheduler after 10.00 seconds, closing client"  # NOQA
        client.close()


def test_simple(default_cached_ds: CachedDataSet):
    parts = list(default_cached_ds.get_const_partitions(partition_size=128))
    p = parts[0]

    datadir = default_cached_ds._cache_path
    default_raw = default_cached_ds._source_ds

    tileshape = Shape(
        (16,) + tuple(default_raw.shape.sig),
        sig_dims=default_raw.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    print(datadir, os.listdir(datadir))

    print(p)

    t_cache = next(p.get_tiles(tiling_scheme=tiling_scheme))
    t_orig = next(
        next(
            default_raw.get_const_partitions(partition_size=128)
        ).get_tiles(tiling_scheme=tiling_scheme)
    )

    print(t_cache)
    print(t_orig)
    assert np.allclose(t_cache.data, t_orig.data)

    for p in default_cached_ds.get_const_partitions(partition_size=128):
        for tile in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


@pytest.mark.slow
def test_with_roi(default_cached_ds: CachedDataSet):
    roi = np.random.choice(a=[0, 1], size=tuple(default_cached_ds.shape.nav))

    tileshape = Shape(
        (16,) + tuple(default_cached_ds.shape.sig),
        sig_dims=default_cached_ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_cached_ds.shape,
    )

    for p in default_cached_ds.get_const_partitions(partition_size=128):
        for tile in p.get_tiles(tiling_scheme=tiling_scheme, roi=roi):
            pass


@pytest.mark.asyncio
async def test_with_dask_executor(tmpdir_factory, default_raw, dask_executor):
    """
    integration test with the dask executor
    """
    datadir = tmpdir_factory.mktemp('cached_ds_directory')
    strategy = LRUCacheStrategy(capacity=1024*1024*1024)
    ds = CachedDataSet(
        source_ds=default_raw,
        cache_path=datadir,
        strategy=strategy,
    )
    ds = ds.initialize(executor=dask_executor)
    list(ds.get_const_partitions(partition_size=128))  # trigger data locality queries


def test_partition_pickles(default_cached_ds):
    """
    assert that we don't do anything unpickleable in the Partition code!
    """
    p = next(default_cached_ds.get_const_partitions(partition_size=128))
    pickle.loads(pickle.dumps(p))
