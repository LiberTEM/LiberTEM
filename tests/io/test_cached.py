import os
import pickle

import numpy as np
import pytest

from libertem.io.dataset.cached import CachedDataSet, LRUCacheStrategy


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


def test_simple(default_cached_ds):
    parts = list(default_cached_ds.get_partitions())
    p = parts[0]

    datadir = default_cached_ds._cache_path
    default_raw = default_cached_ds._source_ds

    print(datadir, os.listdir(datadir))

    print(p)

    t_cache = next(p.get_tiles())
    t_orig = next(next(default_raw.get_partitions()).get_tiles())

    print(t_cache)
    print(t_orig)
    assert np.allclose(t_cache.data, t_orig.data)

    for p in default_cached_ds.get_partitions():
        for tile in p.get_tiles():
            pass


def test_with_roi(default_cached_ds):
    roi = np.random.choice(a=[0, 1], size=tuple(default_cached_ds.shape.nav))

    for p in default_cached_ds.get_partitions():
        for tile in p.get_tiles(roi=roi):
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
    list(ds.get_partitions())  # trigger data locality queries


def test_partition_pickles(default_cached_ds):
    """
    assert that we don't do anything unpickleable in the Partition code!
    """
    p = next(default_cached_ds.get_partitions())
    pickle.loads(pickle.dumps(p))
