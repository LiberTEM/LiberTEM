import os
import json
import logging

import numpy as np
import hdfs3

from libertem.common import Slice, Shape
from libertem.web.messages import MessageConverter
from .base import DataSet, Partition, DataTile, DataSetException, DataSetMeta


log = logging.getLogger(__name__)


class HDFSDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/HDFSDatasetParams.schema.json",
        "title": "HDFSDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "hdfs"},
            "path": {"type": "string"},
            "tileshape": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["type", "path", "tileshape"],
    }

    def convert_to_python(self, raw_data):
        data = {
            "index_path": raw_data["path"],
            "tileshape": raw_data["tileshape"],
        }
        return data


class HDFSReader(object):
    def __init__(self, host, port):
        self._host = host
        self._port = port

    def get_fs(self):
        # TODO: maybe this needs to be a context manager, too, so we can do:
        # with reader.get_fs() as fs:
        #   with fs.open("...") as f:
        #       f.read()
        return hdfs3.HDFileSystem(
            host=self._host, port=self._port, pars={
                'input.localread.default.buffersize': '1',
                'input.read.default.verify': '0',
                'dfs.domain.socket.path': '/run/user/1000/hdfs-short-circuit.socket',
            }
        )


class BinaryHDFSDataSet(DataSet):
    def __init__(self, index_path, host, port, tileshape, worker_map=None):
        super().__init__()
        self.index_path = index_path
        self.dirname = os.path.dirname(index_path)
        self.host = host
        self.port = port
        self.tileshape = tileshape
        self._worker_map = worker_map
        self._sig_dims = 2  # FIXME: need to put this into the json metadata!

    def initialize(self):
        with self.get_reader().get_fs().open(self.index_path) as f:
            self._index = json.load(f)
        assert self._index['mode'] == 'rect', 'unsupported mode: %s' % self._index['mode']
        self._meta = DataSetMeta(
            shape=self.shape,
            dtype=self.dtype,
        )
        return self

    def get_reader(self):
        return HDFSReader(host=self.host, port=self.port)

    @classmethod
    def get_msg_converter(cls):
        return HDFSDatasetParams

    @property
    def dtype(self):
        return self._index['dtype']

    @property
    def shape(self):
        return Shape(self._index['shape'], sig_dims=self._sig_dims)

    def check_valid(self):
        # TODO: maybe later relax the validity requirements to reduce load
        try:
            for partition in self._index['partitions']:
                path = os.path.join(self.dirname, partition['filename'])
                with self.get_reader().get_fs().open(path, "rb"):
                    pass
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_partitions(self):
        for partition in self._index['partitions']:
            reader = self.get_reader()
            yield BinaryHDFSPartition(
                path=os.path.join(self.dirname, partition['filename']),
                tileshape=self.tileshape,
                meta=self._meta,
                reader=reader,
                worker_map=self._worker_map,
                partition_slice=Slice(
                    origin=tuple(partition['origin']) + (0, 0),
                    shape=Shape(partition['shape'], sig_dims=2),
                ),
            )

    def __repr__(self):
        return "<BinaryHDFSDataSet %s>" % self.index_path


class BinaryHDFSPartition(Partition):
    """
    Store your DataSet as a bunch of binary files (see ingest prototype for format)
    """

    def __init__(self, path, tileshape, worker_map, reader, *args, **kwargs):
        self.path = path
        self.tileshape = tileshape
        self._worker_map = worker_map
        self._reader = reader
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape.sig != self.meta.shape.sig:
                raise DataSetException("BinaryHDFSDataSet only supports whole-frame crops for now")
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        subslices = list(self.slice.subslices(shape=self.tileshape))
        with self._reader.get_fs().open(self.path, 'rb') as f:
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
        worker_map = self._worker_map or {}
        locs = self._reader.get_fs().get_block_locations(self.path)
        assert len(locs) == 1, "splitting partitions into multiple hdfs chunks is not supported"
        locations = [l.decode('utf-8')
                     for l in locs[0]['hosts']]
        log.debug("locations=%r", locations)
        return [
            worker_map.get(l, l)
            for l in locations
        ]

    def __repr__(self):
        return "<BinaryHDFSPartition [%r] with tileshape=%s>" % (
            self.slice, self.tileshape
        )
