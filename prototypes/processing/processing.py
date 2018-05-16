# -*- encoding: utf-8 -*-
import os
import json
import socket
import operator
import functools
import hdfs3
import numpy as np
from dask import distributed as dd


# threading in openblas interferes with our own
# and does some stupid sched_yield'ing
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

"""
TODO:
 + split into modules
 + implement BinaryLocalFSDataSet and BinaryLocalFSPartition
 + implement HDF5LocalFSDataSet and *Partition
 + move out of prototypes folder into src/libertem/...
"""


class Slice(object):
    def __init__(self, origin, shape):
        """
        Parameters
        ----------
        origin : (int, int) or (int, int, int, int)
            global top-left coordinates of this slice, will be "broadcast" to 4D
        shape : (int, int, int, int)
            the size of this slice
        """
        if len(origin) == 2:
            origin = (origin[0], origin[1], 0, 0)
        self.origin = tuple(origin)
        self.shape = tuple(shape)
        # TODO: allow to use Slice objects directly for... slices!
        # arr[slice]
        # or use a Slicer object, a little bit like hyperspy .isig, .inav?
        # Slicer(arr)[slice]
        # can we implement some kind of slicer interface? __slice__?

    def __repr__(self):
        return "<Slice origin=%r shape=%r>" % (self.origin, self.shape)

    def shift(self, other):
        """
        make a new ``Slice`` with origin relative to ``other.origin``
        and the same shape as this ``Slice``
        """
        assert len(other.origin) == len(self.origin)
        return Slice(origin=tuple(their_coord - our_coord
                                  for (our_coord, their_coord) in zip(self.origin, other.origin)),
                     shape=self.shape)

    def get(self, arr=None):
        o, s = self.origin, self.shape
        if arr:
            return arr[
                o[0]:(o[0] + s[0]),
                o[1]:(o[1] + s[1]),
                o[2]:(o[2] + s[2]),
                o[3]:(o[3] + s[3]),
            ]
        else:
            return (
                slice(o[0], (o[0] + s[0])),
                slice(o[1], (o[1] + s[1])),
                slice(o[2], (o[2] + s[2])),
                slice(o[3], (o[3] + s[3])),
            )

    def subslices(self, shape):
        """
        Parameters
        ----------
        shape : (int, int, int, int)
            the shape of each sub-slice

        Yields
        ------
        Slice
            all subslices, in fast-access order
        """
        for i in range(len(self.shape)):
            assert self.shape[i] % shape[i] == 0
        ny = self.shape[0] // shape[0]
        nx = self.shape[1] // shape[1]
        nv = self.shape[2] // shape[2]
        nu = self.shape[3] // shape[3]

        return (
            Slice(
                origin=(
                    self.origin[0] + y * shape[0],
                    self.origin[1] + x * shape[1],
                    self.origin[2] + v * shape[2],
                    self.origin[3] + u * shape[3],
                ),
                shape=shape,
            )
            for y in range(ny)
            for x in range(nx)
            for v in range(nv)
            for u in range(nu)
        )


class DataSet(object):
    def get_partitions(self):
        raise NotImplementedError()


class BinaryHDFSDataSet(DataSet):
    def __init__(self, index_path, host, port, stackheight=8):
        self.index_path = index_path
        self.dirname = os.path.dirname(index_path)
        self.host = host
        self.port = port
        self._fs = self.get_fs()
        self._load()
        # TODO: stackheight and framesize are only passed to the partition
        # do they belong in DataSet?
        self.stackheight = stackheight
        self.framesize = functools.reduce(operator.mul, tuple(self._index['shape'][-2:]))

    def get_fs(self):
        return hdfs3.HDFileSystem(
            host=self.host, port=self.port, pars={
                'input.localread.default.buffersize': '1',
                'input.read.default.verify': '0',
            }
        )

    def _load(self):
        with self._fs.open(self.index_path) as f:
            self._index = json.load(f)
        assert self._index['mode'] == 'rect', 'unsupported mode: %s' % self._index['mode']

    def get_partitions(self):
        for partition in self._index['partitions']:
            yield BinaryHDFSPartition(
                path=os.path.join(self.dirname, partition['filename']),
                tileshape=(self.stackheight, self.framesize),
                dataset=self,
                dtype=self._index['dtype'],
                partition_slice=Slice(origin=partition['origin'], shape=partition['shape']),
            )

    def __repr__(self):
        return "<BinaryHDFSDataSet %s>" % self.index_path


class DataTile(object):
    def __init__(self, data, tile_slice):
        """
        Parameters
        ----------
        tile_slice : Slice
            the global coordinates for this data tile

        data : numpy.ndarray
        """
        self.data = data
        self.tile_slice = tile_slice


class Partition(object):
    def __init__(self, dataset, dtype, partition_slice):
        self.dataset = dataset
        self.dtype = dtype
        self.slice = partition_slice

    def get_tiles(self):
        raise NotImplementedError()

    def get_locations(self):
        raise NotImplementedError()


class BinaryHDFSPartition(Partition):
    """
    Store your DataSet as a bunch of binary files (see ingest prototype for format)
    """

    def __init__(self, path, tileshape, *args, **kwargs):
        self.path = path
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self):
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        assert (self.slice.shape[0] * self.slice.shape[1]) % self.tileshape[0] == 0,\
            "please chose a tileshape that evenly divides the partition"
        num_stacks = (self.slice.shape[0] * self.slice.shape[1]) // self.tileshape[0]
        subslices = list(self.slice.subslices(shape=(1, self.tileshape[0],
                                                     self.slice.shape[2],
                                                     self.slice.shape[3])))
        assert num_stacks == len(subslices)
        # NOTE: computation is done on (stackheight, framesize) tiles, but logically, they
        # are equivalent to tiles of shape (1, stackheight, frameheight, framewidth)
        with self.dataset.get_fs().open(self.path, 'rb') as f:
            for tile_slice in subslices:
                f.read(length=data.nbytes, out_buffer=data)
                yield DataTile(data=data, tile_slice=tile_slice)

    def get_locations(self):
        """
        Returns
        -------
        list of str
            IP addresses of hosts on which this partition is available
        """
        locs = self.dataset.get_fs().get_block_locations(self.path)
        assert len(locs) == 1, "splitting partitions into multiple hdfs chunks is not supported"
        return [socket.gethostbyname(l.decode('utf-8'))
                for l in locs[0]['hosts']]

    def __repr__(self):
        return "<BinaryHDFSPartition of %r [%r] with tileshape=%s>" % (
            self.dataset, self.slice, self.tileshape
        )


class Job(object):
    """
    A computation on a DataSet. Inherit from this class and implement ``get_tasks``
    to yield tasks for your specific computation.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def get_tasks(self):
        """
        Yields
        ------
        Task
            ...
        """
        raise NotImplementedError()


class ApplyMasksJob(Job):
    def __init__(self, masks, *args, **kwargs):
        self.masks = self._merge_masks(masks)
        self.orig_masks = masks
        super().__init__(*args, **kwargs)

    def _merge_masks(self, masks):
        """
        flatten and merge masks into one array

        Parameters
        ----------
        masks : [ndarray]
            list of 2D arrays that represent masks
        """
        masks = [m.flatten() for m in masks]
        return np.stack(masks, axis=1)

    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield ApplyMasksTask(partition=partition, masks=self.masks)

    @property
    def maskcount(self):
        return len(self.orig_masks)


class Task(object):
    """
    A computation on a partition. Inherit from this class and implement ``__call__``
    for your specific computation.
    """

    def __init__(self, partition):
        self.partition = partition

    def get_locations(self):
        return self.partition.get_locations()

    def __call__(self):
        raise NotImplementedError()


class ApplyMasksTask(Task):
    def __init__(self, masks, *args, **kwargs):
        self.masks = masks
        super().__init__(*args, **kwargs)

    def __call__(self):
        parts = []
        for data_tile in self.partition.get_tiles():
            result = data_tile.data.dot(self.masks)
            parts.append(
                ResultTile(
                    data=result,
                    tile_slice=data_tile.tile_slice,
                )
            )
        return parts


class ResultTile(object):
    def __init__(self, data, tile_slice):
        self.data = data
        self.tile_slice = tile_slice

    def __repr__(self):
        return "<ResultTile for slice=%r>" % self.tile_slice

    def copy_to_result(self, result):
        # FIXME: assumes tile size is less than or equal one row of frames. is this true?
        # let's assert it for now:
        assert self.tile_slice.shape[0] == 1

        # (frames, masks) -> (masks, _, frames)
        shape = self.data.shape
        reshaped_data = self.data.reshape(shape[0], 1, shape[1]).transpose()
        result[(Ellipsis,) + self.tile_slice.get()[0:2]] = reshaped_data
        return result


class JobExecutor(object):
    def run_job(self, job):
        raise NotImplementedError()


class DaskJobExecutor(JobExecutor):
    def __init__(self, scheduler_uri):
        self.scheduler_uri = scheduler_uri
        self.client = dd.Client(self.scheduler_uri, processes=False)

    def run_job(self, job):
        futures = [
            self.client.submit(task, workers=task.get_locations())
            for task in job.get_tasks()
        ]
        for future, result in dd.as_completed(futures, with_results=True):
            yield result
