import os
import logging
import warnings
import contextlib

import numpy as np
from ncempy.io.ser import fileSER

from libertem.common import Shape, Slice
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta,
    _roi_to_indices, DataTile,
)

log = logging.getLogger(__name__)


class SERDatasetParams(MessageConverter):
    SCHEMA = {
      "$schema": "http://json-schema.org/draft-07/schema#",
      "$id": "http://libertem.org/SERDatasetParams.schema.json",
      "title": "SERDatasetParams",
      "type": "object",
      "properties": {
          "type": {"const": "SER"},
          "path": {"type": "string"},
      },
      "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        return data


class SERFile:
    def __init__(self, path, num_frames):
        self._path = path
        self._num_frames = num_frames

    def _get_handle(self):
        return fileSER(self._path)

    @contextlib.contextmanager
    def get_handle(self):
        with self._get_handle() as f:
            yield f

    @property
    def num_frames(self):
        return self._num_frames

    @property
    def start_idx(self):
        return 0

    @property
    def end_idx(self):
        return self.num_frames


class SERFileSet(FileSet):
    pass


class SERDataSet(DataSet):
    """
    Read TIA SER files.

    Parameters
    ----------
    path: str
        Path to the .ser file
    """
    def __init__(self, path, emipath=None):
        super().__init__()
        self._path = path
        self._meta = None
        self._filesize = None
        self._num_frames = None
        if emipath is not None:
            warnings.warn(
                "emipath is not used anymore, as it was removed from ncempy", DeprecationWarning
            )

    def _do_initialize(self):
        self._filesize = os.stat(self._path).st_size
        reader = SERFile(path=self._path, num_frames=None)

        with reader.get_handle() as f1:
            self._num_frames = f1.head['ValidNumberElements']
            if f1.head['ValidNumberElements'] == 0:
                raise DataSetException("no data found in file")

            data, meta_data = f1.getDataset(0)
            dtype = f1._dictDataType[meta_data['DataType']]
            nav_dims = tuple(
                reversed([
                    int(dim['DimensionSize'])
                    for dim in f1.head['Dimensions']
                ])
            )
            shape = nav_dims + tuple(data.shape)
            sig_dims = len(data.shape)
            self._meta = DataSetMeta(
                shape=Shape(shape, sig_dims=sig_dims),
                raw_dtype=dtype,
            )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        return SERDatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return set(["ser"])

    @classmethod
    def detect_params(cls, path, executor):
        if path.lower().endswith(".ser"):
            return {
                "parameters": {
                    "path": path
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
        try:
            with fileSER(self._path) as f1:
                if f1.head['ValidNumberElements'] == 0:
                    raise DataSetException("no data found in file")
                if f1.head['DataTypeID'] not in (0x4120, 0x4122):
                    raise DataSetException("unknown datatype id: %s" % f1.head['DataTypeID'])
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e) from e

    def get_cache_key(self):
        return {
            "path": self._path,
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (64*1024*1024))
        return res

    def _get_fileset(self):
        assert self._num_frames is not None
        return SERFileSet([
            SERFile(
                path=self._path,
                num_frames=self._num_frames,
            )
        ])

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in SERPartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield SERPartition(
                path=self._path,
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<SERDataSet for %s>" % (self._path,)


class SERPartition(BasePartition):
    def __init__(self, path, *args, **kwargs):
        self._path = path
        super().__init__(*args, **kwargs)

    def validate_tiling_scheme(self, tiling_scheme):
        supported = (1,) + tuple(self.shape.sig)
        if tuple(tiling_scheme.shape) != supported:
            raise ValueError(
                "invalid tiling scheme: only supports %r, not %r" % (supported, tiling_scheme.shape)
            )

    def adjust_tileshape(self, tileshape):
        # force single-frame tiles
        return (1,) + tileshape[1:]

    def get_base_shape(self):
        return (1,) + tuple(self.shape.sig)

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        shape = Shape((1,) + tuple(self.shape.sig), sig_dims=self.shape.sig.dims)

        self.validate_tiling_scheme(tiling_scheme)

        start = self._start_frame
        stop = start + self._num_frames
        if roi is None:
            indices = np.arange(start, stop)
            offset = start
        else:
            indices = _roi_to_indices(roi, start, stop)
            offset = np.count_nonzero(roi.reshape((-1,))[:start])

        with fileSER(self._path) as f:
            for num, idx in enumerate(indices):
                tile_slice = Slice(origin=(num + offset, 0, 0), shape=shape)
                data, metadata = f.getDataset(int(idx))
                if data.dtype != np.dtype(dest_dtype):
                    data = data.astype(dest_dtype)
                self._preprocess(data, tile_slice)
                yield DataTile(
                    data.reshape(shape),
                    tile_slice=tile_slice,
                    scheme_idx=0,
                )
