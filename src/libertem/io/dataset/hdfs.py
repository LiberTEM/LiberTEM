import os
import json
import socket

import numpy as np
import hdfs3

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException


class BinaryHDFSDataSet(DataSet):
    def __init__(self, index_path, host, port, tileshape):
        self.index_path = index_path
        self.dirname = os.path.dirname(index_path)
        self.host = host
        self.port = port
        self._fs = self.get_fs()
        self.tileshape = tileshape
        self._sig_dims = 2  # FIXME: need to put this into the json metadata!

    def initialize(self):
        with self._fs.open(self.index_path) as f:
            self._index = json.load(f)
        assert self._index['mode'] == 'rect', 'unsupported mode: %s' % self._index['mode']
        return self

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

    @property
    def dtype(self):
        return self._index['dtype']

    @property
    def shape(self):
        return Shape(self._index['shape'], sig_dims=self._sig_dims)

    @property
    def raw_shape(self):
        # FIXME: need to distinguish shape/raw_shape in json metadata
        return Shape(self._index['shape'], sig_dims=self._sig_dims)

    def check_valid(self):
        # TODO: maybe later relax the validity requirements to reduce load
        try:
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
            if crop_to.shape.sig != self.dataset.shape.sig:
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
