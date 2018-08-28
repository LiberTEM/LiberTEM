import os
import json
import socket
import operator
import functools

import numpy as np
import hdfs3

from libertem.common.slice import Slice
from .base import DataSet, Partition, DataTile, DataSetException


class BinaryHDFSDataSet(DataSet):
    def __init__(self, index_path, host, port, tileshape):
        self.index_path = index_path
        self.dirname = os.path.dirname(index_path)
        self.host = host
        self.port = port
        self._fs = self.get_fs()
        self.check_valid()
        self._load()
        self.tileshape = tileshape
        self.framesize = functools.reduce(operator.mul, tuple(self._index['shape'][-2:]))

    def get_fs(self):
        # TODO: maybe this needs to be a context manager, too, so we can do:
        # with ds.get_fs() as fs:
        #   with fs.open("...") as f:
        #       f.read()
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

    @property
    def shape(self):
        return self._index['shape']

    def check_valid(self):
        # TODO: maybe later relax the validity requirements to reduce load
        try:
            self._load()
            for partition in self._index['partitions']:
                path = os.path.join(self.dirname, partition['filename'])
                with self.get_fs().open(path, "rb"):
                    pass
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        for partition in self._index['partitions']:
            yield BinaryHDFSPartition(
                path=os.path.join(self.dirname, partition['filename']),
                tileshape=self.tileshape,
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

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape[2:] != self.dataset.shape[2:]:
                raise DataSetException("BinaryHDFSDataSet only supports whole-frame crops for now")
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        subslices = list(self.slice.subslices(shape=self.tileshape))
        with self.dataset.get_fs().open(self.path, 'rb') as f:
            for tile_slice in subslices:
                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
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
