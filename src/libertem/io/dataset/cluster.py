import os
import json
from collections import defaultdict

import numpy as np

from .base import (
    DataSet, BasePartition, DataSetMeta, DataSetException,
    WritablePartition, WritableDataSet
)
from .raw import RawFile, RawFileSet
from libertem.executor.scheduler import WorkerSet
from libertem.io.writers.base import WriteHandle
from libertem.common import Shape, Slice


class ClusterDataSet(WritableDataSet, DataSet):
    """
    ClusterDataSet: a distributed RAW data set

     * to be used for the cache, for live acquisition, and for simulation integration
     * each node has a directory for a ClusterDataSet
     * the directory contains partitions, each its own raw file
     * information about the structure is saved as a json sidecar file
     * A ClusterDataSet dataset can be incomplete, that is, it can miss complete partitions
       (but partitions themselves are guaranteed to be complete once they have their final filename)
     * use cases for incomplete datasets:
        * each node only caches the partitions it is responsible for
        * partial acquisitions support
     * missing partitions can later be written
     * file names and structure/partitioning are deterministic
     * assumption: all workers on a single host share the dataset

    Parameters
    ----------

    path : str
        Absolute filesystem base path, pointing to an existing directory.
        Assumes a uniform setup (same absolute path used on all nodes)

    structure : PartitionStructure
        Partitioning structure instance. Must be specified when creating a new dataset.
    """
    def __init__(self, path, structure=None, io_backend=None):
        super().__init__(io_backend=io_backend)
        self._path = path
        self._dtype = structure.dtype
        self._structure = structure
        self._meta = DataSetMeta(
            shape=structure.shape,
            raw_dtype=np.dtype(structure.dtype),
            sync_offset=0,
            image_count=int(np.prod(structure.shape.nav)),
        )
        self._executor = None

    def initialize(self, executor):
        """
        Initialize is running on the master node, but we have
        access to the executor.
        """
        # 1) create sidecar files on hosts that don't have them yet
        executor.run_each_host(self._ensure_sidecar)
        # 2) check that all sidecar files that exist are equal
        sidecars = executor.run_each_host(self._read_sidecar)
        given_structure = self._structure.serialize()
        if not all(s == given_structure for s in sidecars.values()):
            print(sidecars.values())
            print(given_structure)
            raise DataSetException(
                "inconsistent sidecars, please inspect %s on each node" % (
                    self._sidecar_path(),
                )
            )
        self._executor = executor
        return self

    def _get_fileset(self):
        return RawFileSet([
            RawFile(
                path=self._get_path_for_idx(idx),
                start_idx=start_idx,
                end_idx=end_idx,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
            for (idx, (start_idx, end_idx)) in enumerate(self._structure.slices)
        ])

    def _get_path_for_idx(self, idx):
        # FIXME: possibly different paths per node
        return os.path.join(self._path, "parts", "partition-%08d" % idx)

    @property
    def dtype(self):
        return self._dtype

    @property
    def raw_dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._meta.shape

    def _ensure_sidecar(self):
        """
        run on each node on initialization
        """
        os.makedirs(os.path.join(self._path, "parts"), exist_ok=True)
        if not os.path.exists(self._sidecar_path()):
            self._write_sidecar()

    def _sidecar_path(self):
        return os.path.join(self._path, "structure.json")

    def _read_sidecar(self):
        """
        run on each node on initialization
        """
        with open(self._sidecar_path(), "r") as fh:
            return json.load(fh)

    def _write_sidecar(self):
        """
        run on each node on initialization, if sidecar doesn't exist
        """
        with open(self._sidecar_path(), "w") as fh:
            json.dump(self._structure.serialize(), fh)

    def check_valid(self):
        if not os.path.exists(self._path) or not os.path.isdir(self._path):
            raise DataSetException("path %s does not exist or is not a directory" % self._path)

    @classmethod
    def detect_params(cls, path, executor):
        # TODO: read sidecar file etc.?
        return False

    @classmethod
    def get_msg_converter(cls):
        raise NotImplementedError()

    def get_diagnostics(self):
        return []

    def get_partitions(self):
        idx_to_workers = self._get_all_workers()
        fileset = self._get_fileset()
        for (idx, (start_idx, end_idx)) in enumerate(self._structure.slices):
            part_slice = Slice(
                origin=(start_idx,) + tuple([0] * self.shape.sig.dims),
                shape=Shape(((end_idx - start_idx),) + tuple(self.shape.sig),
                            sig_dims=self.shape.sig.dims)
            )
            yield ClusterDSPartition(
                path=self._get_path_for_idx(idx),
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start_idx,
                num_frames=end_idx - start_idx,
                workers=idx_to_workers[idx] or None,
                io_backend=self.get_io_backend(),
            )

    def _get_all_workers(self):
        """
        returns a mapping idx -> workers
        """
        if self._executor is None:
            raise RuntimeError(
                "invalid state: _get_all_workers needs access to the executor"
            )

        paths = [
            (idx, self._get_path_for_idx(idx))
            for (idx, slice_) in enumerate(self._structure.slices)
        ]

        def _check_paths():
            return [
                idx
                for idx, path in paths
                if os.path.exists(path)
            ]

        host_to_idxs = self._executor.run_each_host(_check_paths)
        workers = self._executor.get_available_workers()

        # invert: idx -> workers
        result = defaultdict(lambda: WorkerSet([]))
        for host, idxs in host_to_idxs.items():
            for idx in idxs:
                result[idx] = result[idx].extend(workers.get_by_host(host))
        return result


class ClusterDSPartition(WritablePartition, BasePartition):
    def __init__(self, path, workers, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._path = path
        self._workers = workers

    def get_write_handle(self):
        # FIXME: make example executable? involves temporary directory, source dataset, ...
        # TODO: Direct I/O writing support
        """
        Get a handle to write to this partition. Current rules:

        1) You can only write a complete partition at once, which is then immutable afterwards.
           If you want to change something, you need to write the whole partition again.
        2) Once the `with`-block is exited successfully, the data is written down to disk. If
           there is an error while writing the data, the partition will not be moved into its final
           place and it will be missing from the data set. There cannot be a "parially written"
           partition.

        Example
        -------
        >>> with dest_part.get_write_handle() as wh:  #  doctest: +SKIP
        ...     for tile in wh.write_tiles(source_part.get_tiles()):
        ...         pass  # do something with `tile`
        """
        tmp_path = os.path.dirname(self._path)
        return WriteHandle(
            path=self._path,
            tmp_base_path=tmp_path,
            dtype=self.dtype,
            part_slice=self.slice,
        )

    def delete(self):
        """
        delete this partition. needs to run on all nodes that have this partition
        """
        if os.path.exists(self._path):
            os.unlink(self._path)

    def _have_data(self):
        return os.path.exists(self._path)

    def get_locations(self):
        return self._workers

    def get_canonical_path(self):
        return os.path.realpath(self._path)
