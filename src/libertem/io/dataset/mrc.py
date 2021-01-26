import os
import logging
import numpy as np

from ncempy.io.mrc import fileMRC

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import DataSet, FileSet, BasePartition, DataSetException, DataSetMeta, LocalFile

log = logging.getLogger(__name__)


class MRCDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/MRCDatasetParams.schema.json",
        "title": "MRCDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "MRC"},
            "path": {"type": "string"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
        },
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class MRCFile(LocalFile):
    def open(self):
        self._file = fileMRC(self._path)
        self._mmap = self._file.getMemmap()
        self._raw_mmap = self._mmap

    def close(self):
        self._file = None
        self._mmap = None
        self._raw_mmap = None

    def fileno(self):
        return None


class MRCFileSet(FileSet):
    pass


class MRCDataSet(DataSet):
    """
    Read MRC files.

    Examples
    --------

    >>> ds = ctx.load("mrc", path="/path/to/file.mrc")  # doctest: +SKIP

    Parameters
    ----------
    path: str
        Path to the .mrc file

    nav_shape: tuple of int, optional
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int, optional
        Signal/detector size (height, width)

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    """
    def __init__(self, path, nav_shape=None, sig_shape=None, sync_offset=0, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("MRCDataSet currently doesn't support alternative I/O backends")
        self._path = path
        self._meta = None
        self._filesize = None
        self._image_count = None
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset

    def _do_initialize(self):
        self._filesize = os.stat(self._path).st_size
        f = fileMRC(self._path)
        data = f.getMemmap()
        native_shape = data.shape
        dtype = data.dtype

        self._image_count = native_shape[0]

        if self._nav_shape is None:
            self._nav_shape = tuple((int(native_shape[0]),))

        native_sig_shape = tuple(
            int(i)
            for i in f.gridSize
            if i != 1
        )
        if self._sig_shape is None:
            self._sig_shape = native_sig_shape
        elif int(np.prod(self._sig_shape)) != int(np.prod(native_sig_shape)):
            raise DataSetException(
                "sig_shape must be of size: %s" % int(np.prod(native_sig_shape))
            )

        self._sig_dims = len(self._sig_shape)
        self._shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._nav_shape_product = self._shape.nav.size
        self._sync_offset_info = self.get_sync_offset_info()

        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        return MRCDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return set(["mrc"])

    @classmethod
    def detect_params(cls, path, executor):
        if path.lower().endswith(".mrc"):
            f = fileMRC(path)
            data = f.getMemmap()
            shape = data.shape
            sig_shape = tuple(
                int(i)
                for i in f.gridSize
                if i != 1
            )
            nav_shape = shape[0]
            return {
                "parameters": {
                    "path": path,
                    "nav_shape": tuple((int(nav_shape),)),
                    "sig_shape": sig_shape,
                },
                "info": {
                    "image_count": int(nav_shape),
                    "native_sig_shape": sig_shape,
                }
            }
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        return True  # anything to check?

    def get_cache_key(self):
        return {
            "path": self._path,
            "shape": tuple(self.shape),
            "sync_offset": self._sync_offset,
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

    def _get_fileset(self):
        assert self._image_count is not None
        return MRCFileSet([
            MRCFile(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in MRCPartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions(),
                sync_offset=self._sync_offset):
            yield MRCPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<MRCDataSet for %s>" % (self._path,)


class MRCPartition(BasePartition):
    pass
