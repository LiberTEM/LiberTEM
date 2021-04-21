import os
import json
import hashlib
import sqlite3
import time
import glob
from typing import Union

import numpy as np

from .base import (
    DataSet, Partition, PartitionStructure
)
from libertem.io.dataset.cluster import ClusterDataSet


class VerboseRow(sqlite3.Row):
    """sqlite3.Row with a __repr__"""
    def __repr__(self):
        return "<VerboseRow %r>" % (
            {
                k: self[k]
                for k in self.keys()
            },
        )


class CacheItem:
    """
    A CacheItem describes a single unit of data that is cached, in this case
    a partition of the CachedDataSet.
    """
    def __init__(self, dataset: str, partition: int, size: int, path: str):
        self.dataset = dataset  # dataset id string, for example the cache key
        self.partition = partition  # partition index as integer
        self.size = size  # partition size in bytes
        self.path = path  # full absolute path to the file for the partition
        self.is_orphan = False  # quack

    def __eq__(self, other):
        # dataset and partition are composite pk
        return self.dataset == other.dataset and self.partition == other.partition

    def __repr__(self):
        return "<CacheItem: %s/%d>" % (self.dataset, self.partition)

    @classmethod
    def from_row(cls, row):
        return cls(
            dataset=row["dataset"],
            partition=row["partition"],
            size=row["size"],
            path=row["path"]
        )


class OrphanItem:
    """
    An orphan, a file in the cache structure, which we don't know much about
    (only path and size)
    """
    def __init__(self, path, size):
        self.path = path
        self.size = size
        self.is_orphan = True

    def __eq__(self, other):
        return self.path == other.path

    def __repr__(self):
        return "<OrphanItem: %s>" % (self.path,)

    @classmethod
    def from_row(cls, row):
        return cls(
            size=row["size"],
            path=row["path"]
        )


class CacheStats:
    def __init__(self, db_path):
        self._db_path = db_path
        self._conn = None
        self._connect()

    def _connect(self):
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = VerboseRow
        self._conn = conn

    def close(self):
        self._conn.close()
        self._conn = None

    def initialize_schema(self):
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS stats (
            dataset VARCHAR NOT NULL,           -- dataset id, for example the cache key
            partition INTEGER NOT NULL,         -- partition index as integer
            hits INTEGER NOT NULL,              -- access counter
            size INTEGER NOT NULL,              -- in bytes
            last_access REAL NOT NULL,          -- float timestamp, like time.time()
            path VARCHAR NOT NULL,              -- full path to the file for this partition
            PRIMARY KEY (dataset, partition)
        );""")
        self._conn.execute("""
        CREATE TABLE IF NOT EXISTS orphans (
            path VARCHAR NOT NULL,              -- full path to the orphaned file
            size INTEGER NOT NULL,              -- in bytes
            PRIMARY KEY (path)
        );""")
        self._conn.execute("PRAGMA user_version = 1;")

    def _have_item(self, cache_item: CacheItem):
        rows = self._conn.execute("""
        SELECT hits FROM stats
        WHERE dataset = ? AND partition = ?
        """, [cache_item.dataset, cache_item.partition]).fetchall()
        return len(rows) > 0

    def record_hit(self, cache_item: CacheItem):
        now = time.time()
        self._conn.execute("BEGIN")
        if not self._have_item(cache_item):
            self._conn.execute("""
            INSERT INTO stats (partition, dataset, hits, size, last_access, path)
            VALUES (?, ?, 1, ?, ?, ?)
            """, [cache_item.partition, cache_item.dataset,
                  cache_item.size, now, cache_item.path])
        else:
            self._conn.execute("""
            UPDATE stats
            SET hits = MAX(hits + 1, 1), last_access = ?
            WHERE dataset = ? AND partition = ?
            """, [now, cache_item.dataset, cache_item.partition])
        self._conn.execute("DELETE FROM orphans WHERE path = ?", [cache_item.path])
        self._conn.commit()

    def record_miss(self, cache_item: CacheItem):
        now = time.time()

        self._conn.execute("BEGIN")
        if not self._have_item(cache_item):
            self._conn.execute("""
            INSERT INTO stats (partition, dataset, hits, size, last_access, path)
            VALUES (?, ?, 0, ?, ?, ?)
            """, [cache_item.partition, cache_item.dataset, cache_item.size,
                  now, cache_item.path])
        else:
            self._conn.execute("""
            UPDATE stats
            SET hits = 0, last_access = ?
            WHERE dataset = ? AND partition = ?
            """, [now, cache_item.dataset, cache_item.partition])
        self._conn.execute("DELETE FROM orphans WHERE path = ?", [cache_item.path])
        self._conn.commit()

    def record_eviction(self, cache_item: Union[CacheItem, OrphanItem]):
        if cache_item.is_orphan:
            self.remove_orphan(cache_item)
        else:
            self._conn.execute("""
            DELETE FROM stats
            WHERE partition = ? AND dataset = ?
            """, [cache_item.partition, cache_item.dataset])
            self._conn.commit()

    def maybe_orphan(self, orphan: OrphanItem):
        """
        Create an entry for a file we don't have any statistics about, after checking
        the stats table for the given path.
        Getting a conflict here means concurrently running maybe_orphan processes,
        so we can safely ignore it.
        """
        exists = len(self._conn.execute("""
        SELECT 1 FROM stats WHERE path = ?
        """, [orphan.path]).fetchall()) > 0
        if not exists:
            self._conn.execute("""
            INSERT OR IGNORE INTO orphans (path, size)
            VALUES (?, ?)
            """, [orphan.path, orphan.size])
            self._conn.commit()
            return orphan

    def get_orphans(self):
        cursor = self._conn.execute("SELECT path, size FROM orphans ORDER BY size DESC")
        return [
            OrphanItem.from_row(row)
            for row in cursor
        ]

    def remove_orphan(self, path: str):
        self._conn.execute("""
        DELETE FROM orphans WHERE path = ?
        """, [path])
        self._conn.commit()

    def get_stats_for_dataset(self, cache_key):
        """
        Return dataset cache stats as dict mapping partition ids to dicts of their
        properties (keys: size, last_access, hits)
        """
        cursor = self._conn.execute("""
        SELECT partition, path, size, hits, last_access
        FROM stats
        WHERE dataset = ?
        """, [cache_key])
        return {
            row["partition"]: self._format_row(row)
            for row in cursor.fetchall()
        }

    def query(self, sql, args=None):
        """
        Custom sqlite query, returns a sqlite3 Cursor object
        """
        if args is None:
            args = []
        return self._conn.execute(sql, args)

    def _format_row(self, row):
        return {
            "path": row["path"],
            "size": row["size"],
            "last_access": row["last_access"],
            "hits": row["hits"],
        }

    def get_used_capacity(self):
        size = self._conn.execute("""
        SELECT SUM(size) AS "total_size" FROM stats;
        """).fetchone()["total_size"]
        size_orphans = self._conn.execute("""
        SELECT SUM(size) AS "total_size" FROM orphans;
        """).fetchone()["total_size"]
        return (size or 0) + (size_orphans or 0)


class CacheStrategy:
    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that should be deleted to make a new item
        with size in bytes `size`.
        """
        raise NotImplementedError()


class LRUCacheStrategy(CacheStrategy):
    def __init__(self, capacity: int):
        self._capacity = capacity
        super().__init__()

    def get_victim_list(self, cache_key: str, size: int, stats: CacheStats):
        """
        Return a list of `CacheItem`s that should be deleted to make
        place for `partition`.
        """
        # LRU with the following modifications:
        # 1) Don't evict from the same dataset, as our accesses
        #    are highly correlated in a single dataset
        # 2) Include orphaned files as preferred victims
        # 3) TODO: work in an estimated miss cost (challenge: estimate I/O cost
        #    independently from whatever calculation the user decides to run!)
        if self.sufficient_space_for(size, stats):
            return []
        victims = []
        space_to_free = size - self.get_available(stats)

        orphans = stats.get_orphans()

        candidates = stats.query("""
        SELECT dataset, partition, size, path
        FROM stats
        WHERE dataset != ?
        ORDER BY last_access ASC
        """, [cache_key])

        to_check = orphans + [
            CacheItem.from_row(row)
            for row in candidates
        ]

        for item in to_check:
            if space_to_free <= 0:
                break
            victims.append(item)
            space_to_free -= item.size
        if space_to_free > 0:
            raise RuntimeError(
                "not enough cache capacity for the requested operation"
            )  # FIXME: exception class
        return victims

    def sufficient_space_for(self, size: int, stats: CacheStats):
        return size <= self.get_available(stats)

    def get_available(self, stats: CacheStats):
        """
        available cache capacity in bytes
        """
        return self._capacity - self.get_used(stats)

    def get_used(self, stats: CacheStats):
        """
        used cache capacity in bytes
        """
        return stats.get_used_capacity()


class Cache:
    """
    Cache object, to be used on a worker node. The interface used by `Partition`\\ s
    to manage the cache. May directly remove files, directories, etc.
    """
    def __init__(self, stats: CacheStats, strategy: CacheStrategy):
        self._stats = stats
        self.strategy = strategy

    def record_hit(self, cache_item: CacheItem):
        self._stats.record_hit(cache_item)

    def record_miss(self, cache_item: CacheItem):
        self._stats.record_miss(cache_item)

    def evict(self, cache_key: str, size: int):
        """
        Make place for `size` bytes which will be used
        by the dataset identified by the `cache_key`.
        """
        victims = self.strategy.get_victim_list(cache_key, size, self._stats)
        for cache_item in victims:
            # if it has been deleted by the user, we don't care and just remove
            # the record from the database:
            if os.path.exists(cache_item.path):
                os.unlink(cache_item.path)
            self._stats.record_eviction(cache_item)

    def collect_orphans(self, base_path: str):
        """
        Check the filesystem structure and record all partitions
        that are missing in the db as orphans, to be deleted on demand.
        """
        # the structure here is: {base_path}/{dataset_cache_key}/parts/*
        orphans = []
        for path in glob.glob(os.path.join(base_path, "*", "parts", "*")):
            size = os.stat(path).st_size
            res = self._stats.maybe_orphan(OrphanItem(path=path, size=size))
            if res is not None:
                orphans.append(res)
        return orphans


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
        A subdirectory will be created for each dataset.

    strategy : CacheStrategy
        A class implementing a cache eviction strategy, for example LRUCacheStrategy
    """
    def __init__(self, source_ds, cache_path, strategy, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._source_ds = source_ds
        self._cache_path = cache_path
        self._cache_key = self._make_cache_key(source_ds.get_cache_key())
        self._path = os.path.join(cache_path, self._cache_key)
        self._cluster_ds = None
        self._executor = None
        self._cache_strategy = strategy

    def _make_cache_key(self, inp):
        inp_as_str = json.dumps(inp)
        return hashlib.sha256(inp_as_str.encode("utf-8")).hexdigest()

    def initialize(self, executor):
        source_structure = PartitionStructure.from_ds(self._source_ds)
        executor.run_each_host(self._ensure_cache_structure)
        cluster_ds = ClusterDataSet(
            path=self._path,
            structure=source_structure,
        )
        cluster_ds.check_valid()
        self._cluster_ds = cluster_ds.initialize(executor=executor)
        self._executor = executor
        return self

    def _get_db_path(self):
        return os.path.join(self._cache_path, "cache.db")

    def _ensure_cache_structure(self):
        os.makedirs(self._path, exist_ok=True)

        cache_stats = CacheStats(self._get_db_path())
        cache_stats.initialize_schema()
        cache = Cache(stats=cache_stats, strategy=self._cache_strategy)
        cache.collect_orphans(self._cache_path)

    @property
    def dtype(self):
        return self._cluster_ds.dtype

    @property
    def shape(self):
        return self._cluster_ds.shape

    @classmethod
    def get_msg_converter(cls):
        raise NotImplementedError(
            "not directly usable from web API"
        )

    def check_valid(self):
        # TODO: validate self._cache_path, what else?
        # - ask cache backend if things look valid (i.e. sidecar cache info is OK)
        return True

    def get_partitions(self):
        for idx, (source_part, cluster_part) in enumerate(zip(self._source_ds.get_partitions(),
                                                              self._cluster_ds.get_partitions())):
            yield CachedPartition(
                source_part=source_part,
                cluster_part=cluster_part,
                meta=cluster_part.meta,
                partition_slice=cluster_part.slice,
                cache_key=self._cache_key,
                cache_strategy=self._cache_strategy,
                db_path=self._get_db_path(),
                idx=idx,
                io_backend=self.get_io_backend(),
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
    def __init__(self, source_part, cluster_part, meta, partition_slice,
                 cache_key, cache_strategy, db_path, idx, io_backend):
        super().__init__(meta=meta, partition_slice=partition_slice, io_backend=io_backend)
        self._source_part = source_part
        self._cluster_part = cluster_part
        self._cache_key = cache_key
        self._cache_strategy = cache_strategy
        self._db_path = db_path
        self._idx = idx

    def _get_cache(self):
        cache_stats = CacheStats(self._db_path)
        return Cache(stats=cache_stats, strategy=self._cache_strategy)

    def _sizeof(self):
        return self.slice.shape.size * np.dtype(self.dtype).itemsize

    def _write_tiles_noroi(self, wh, source_tiles, dest_dtype):
        """
        Write tiles from source_tiles to the cache. After each tile is written, yield
        it for further processing, potentially doing dtype conversion on the fly.
        """
        with wh:
            miss_tiles = wh.write_tiles(source_tiles)
            if np.dtype(dest_dtype) != np.dtype(self._cluster_part.dtype):
                for tile in miss_tiles:
                    yield tile.astype(dest_dtype)
            else:
                yield from miss_tiles

    def _write_tiles_roi(self, wh, source_tiles, cached_tiles):
        """
        Get source tiles without roi, read and cache whole partition, then
        read all tiles selected via roi from the cache (_cluster_part aka cached_tiles).
        """
        with wh:
            for tile in wh.write_tiles(source_tiles):
                pass
        yield from cached_tiles

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        cache = self._get_cache()
        cached_tiles = self._cluster_part.get_tiles(
            tiling_scheme=tiling_scheme,
            dest_dtype=dest_dtype,
            roi=roi,
        )

        cache_item = CacheItem(
            dataset=self._cache_key,
            partition=self._idx,
            path=self._cluster_part.get_canonical_path(),
            size=self._sizeof(),
        )
        if self._cluster_part._have_data():
            yield from cached_tiles
            cache.record_hit(cache_item)
        else:
            cache.evict(cache_key=self._cache_key, size=self._sizeof())
            # NOTE: source_tiles are in native dtype!
            source_tiles = self._source_part.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=self._cluster_part.dtype,
                roi=None,  # NOTE: want to always cache the whole dataset, thus roi=None
            )
            wh = self._cluster_part.get_write_handle()
            if roi is None:
                yield from self._write_tiles_noroi(wh, source_tiles, dest_dtype)
            else:
                yield from self._write_tiles_roi(wh, source_tiles, cached_tiles)
            cache.record_miss(cache_item)

    def get_locations(self):
        """
        returns locations where this partition is cached
        """
        return self._cluster_part.get_locations()

    def evict(self):
        self._cluster_part.delete()
