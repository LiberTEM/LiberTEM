import os
import json
import socket
import operator
import functools

import numpy as np
import hdfs3

from .base import DataSet, Partition
from ..slice import Slice
from ..tiling import DataTile


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

    @property
    def dtype(self):
        return self._index['dtype']

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
        # num_stacks is only computed for comparison to subslices
        num_stacks = (self.slice.shape[0] * self.slice.shape[1]) // self.tileshape[0]
        # NOTE: computation is done on (stackheight, framesize) tiles, but logically, they
        # are equivalent to tiles of shape (1, stackheight, frameheight, framewidth)
        subslices = list(self.slice.subslices(shape=(1, self.tileshape[0],
                                                     self.slice.shape[2],
                                                     self.slice.shape[3])))
        assert num_stacks == len(subslices)
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
