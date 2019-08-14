import os
import json
import hashlib

from .base import (
    DataSet, Partition, PartitionStructure
)
from libertem.io.dataset.cluster import ClusterDataSet


class CachedDataSet(DataSet):
    """
    Cached DataSet.

    Assumes the source DataSet is significantly slower than the cache location
    (otherwise, it may result in memory pressure, as we don't use direct I/O
    to write to the cache.)

    Parameters
    ----------
    source_ds : DataSet
        DataSet on slower file system

    cache_path : str
        Where should the cache be written to? Should be a directory on a fast
        local device (i.e. NVMe SSD if possible, local hard drive if it is faster than network)
    """
    def __init__(self, source_ds, cache_path, enable_direct=False):
        self._source_ds = source_ds
        self._cache_path = cache_path
        self._cache_key = self._make_cache_key(source_ds.get_cache_key())
        self._path = os.path.normpath(os.path.join(cache_path, self._cache_key))
        self._enable_direct = enable_direct
        self._cluster_ds = None
        self._executor = None

    def _make_cache_key(self, inp):
        inp_as_str = json.dumps(inp)
        return hashlib.sha256(inp_as_str.encode("utf-8")).hexdigest()

    def initialize(self, executor):
        source_structure = PartitionStructure.from_ds(self._source_ds)
        executor.run_each_host(self._ensure_cache_structure)
        cluster_ds = ClusterDataSet(
            path=self._path,
            structure=source_structure,
            enable_direct=self._enable_direct,
        )
        cluster_ds.check_valid()
        self._cluster_ds = cluster_ds.initialize(executor=executor)
        self._executor = executor
        return self

    def _ensure_cache_structure(self):
        os.makedirs(self._path, exist_ok=True)

    @property
    def dtype(self):
        return self._cluster_ds.dtype

    @property
    def shape(self):
        return self._cluster_ds.shape

    @classmethod
    def get_msg_converter(cls):
        raise NotImplementedError(
            "not yet usable from web API"
        )

    def check_valid(self):
        # TODO: validate self._cache_path, what else?
        # - ask cache backend if things look valid (i.e. sidecar cache info is OK)
        return True

    def get_partitions(self):
        for source_part, cluster_part in zip(self._source_ds.get_partitions(),
                                            self._cluster_ds.get_partitions()):
            yield CachedPartition(
                source_part=source_part,
                cluster_part=cluster_part,
                meta=cluster_part.meta,
                partition_slice=cluster_part.slice,
            )

    def evict(self, executor):
        for _ in executor.run_each_partition(self.get_partitions(),
                                             lambda p: p.evict(), all_nodes=True):
            pass

    def __repr__(self):
        return "<CachedDataSet dtype=%s shape=%s source_ds=%s cache_path=%s path=%s>" % (
            self.dtype, self.shape, self._source_ds, self._cache_path, self._path
        )


class CachedPartition(Partition):
    def __init__(self, source_part, cluster_part, meta, partition_slice):
        super().__init__(meta=meta, partition_slice=partition_slice)
        self._source_part = source_part
        self._cluster_part = cluster_part

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False,
                  dest_dtype="float32", roi=None, target_size=None):
        cached_tiles = self._cluster_part.get_tiles(crop_to=crop_to, full_frames=full_frames,
                                                   mmap=mmap, dest_dtype=dest_dtype, roi=roi,
                                                   target_size=target_size)
        if self._cluster_part._have_data():
            yield from cached_tiles
        else:
            source_tiles = self._source_part.get_tiles(crop_to=crop_to, full_frames=full_frames,
                                                       mmap=mmap, dest_dtype=dest_dtype, roi=None,
                                                       target_size=target_size)
            wh = self._cluster_part.get_write_handle()
            if roi is None:
                with wh:
                    yield from wh.write_tiles(source_tiles)
            else:
                with wh:
                    # get source tiles without roi, read and cache whole partition, then
                    # read from _cluster_part with roi
                    for tile in wh.write_tiles(source_tiles):
                        pass
                yield from cached_tiles

    def get_locations(self):
        """
        returns locations where this partition is cached
        """
        return self._cluster_part.get_locations()

    def evict(self):
        self._cluster_part.delete()
