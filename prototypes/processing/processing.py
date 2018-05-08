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
 + fix partitioning scheme to only create rectangular partitions
 + implement BinaryLocalFSDataSet and BinaryLocalFSPartition
 + implement HDF5LocalFSDataSet and *Partition
 + move out of prototypes folder into src/libertem/...
"""


class DataSet(object):
    def get_partitions(self):
        pass

    def load_from_uri(self, uri):
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
        self.framesize = functools.reduce(operator.mul, tuple(self._index['orig_shape'][-2:]))

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
        assert self._index['mode'] == 'linear', 'unsupported mode: %s' % self._index['mode']

    def get_partitions(self):
        for partition in self._index['partitions']:
            yield BinaryHDFSPartition(
                path=os.path.join(self.dirname, partition['filename']),
                start=partition['start'],
                end=partition['end'],
                tileshape=(self.stackheight, self.framesize),
                dataset=self,
                dtype=self._index['dtype'],
                hyperslab=None,  # TODO
            )

    def __repr__(self):
        return "<BinaryHDFSDataSet %s>" % self.index_path


class DataTile(object):
    def __init__(self, data):
        self.data = data


class Partition(object):
    def __init__(self, dataset, dtype, hyperslab):
        self.dataset = dataset
        self.dtype = dtype
        self.hyperslab = hyperslab

    def get_tiles(self):
        raise NotImplementedError()

    def get_locations(self):
        raise NotImplementedError()


class BinaryHDFSPartition(Partition):
    """
    Store your DataSet as a bunch of binary files (see ingest prototype for format)
    """

    def __init__(self, path, start, end, tileshape, *args, **kwargs):
        self.path = path
        self.start = start
        self.end = end
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self):
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        with self.dataset.get_fs().open(self.path, 'rb') as f:
            # FIXME: assumes that we don't have any short reads, i.e. that data.nbytes divides
            # the partition evenly
            num_stacks = (self.end - self.start) // self.tileshape[0]
            for stack in range(num_stacks):
                f.read(length=data.nbytes, out_buffer=data)
                yield DataTile(data=data)  # TODO: the tile needs to know its 'coordinates', right?

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
        return "<BinaryHDFSPartition of %r [%d, %d] with tileshape=%s>" % (
            self.dataset, self.start, self.end, self.tileshape
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
        self.masks = masks
        super().__init__(*args, **kwargs)

    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield ApplyMasksTask(partition=partition, masks=self.masks)


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
        return [
            ResultTile(
                result=data_tile.data.dot(self.masks),
                hyperslab=None,  # TODO
            )
            for data_tile in self.partition.get_tiles()
        ]


class ResultTile(object):
    def __init__(self, result, hyperslab):
        self.result = result
        self.hyperslab = hyperslab

    def __repr__(self):
        return "<ResultTile for hyperslab=%r>" % self.hyperslab


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
