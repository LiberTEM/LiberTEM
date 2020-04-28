import os
import warnings

import numpy as np

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    BasePartition, LocalFile, FileSet,
)


class RAWDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/RAWDatasetParams.schema.json",
        "title": "RAWDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "RAW"},
            "path": {"type": "string"},
            "dtype": {"type": "string"},
            "scan_size": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "detector_size": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "enable_direct": {
                "type": "boolean"
            }
        },
        "required": ["type", "path", "dtype", "scan_size", "detector_size"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "dtype", "scan_size", "detector_size", "enable_direct"]
        }
        return data


class RawFile(LocalFile):
    pass


class RawFileSet(FileSet):
    pass


class RawFileDataSet(DataSet):
    """
    Read raw data from a single file of raw binary data. This reader assumes the following
    format:

     * only raw data (no file header)
     * frames are stored in C-order without additional frame headers
     * dtype supported by numpy

    Examples
    --------

    >>> ds = ctx.load("raw", path=path_to_raw, scan_size=(16, 16),
    ...               dtype="float32", detector_size=(128, 128))

    Parameters
    ----------

    path: str
        Path to the file

    scan_size: tuple of int, optional
        A n-tuple that specifies the size of the scanned region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    dtype: numpy dtype
        The dtype of the data as it is on disk. Can contain endian indicator, for
        example >u2 for big-endian 16bit data.

    detector_size: tuple of int
        Common case: (height, width); but can be any dimensionality

    enable_direct: bool
        Enable direct I/O. This bypasses the filesystem cache and is useful for
        systems with very fast I/O and for data sets that are much larger than the
        main memory.
    """
    def __init__(self, path, scan_size, dtype, detector_size=None, enable_direct=False,
                 detector_size_raw=None, crop_detector_to=None, tileshape=None):
        super().__init__()
        # handle backwards-compatability:
        if tileshape is not None:
            warnings.warn("tileshape argument is deprecated, ignored", DeprecationWarning)
        # FIXME execute deprecation after 0.6.0
        if crop_detector_to is not None:
            warnings.warn("crop_detector_to and detector_size_raw are deprecated, "
                          "and will be removed after version 0.6.0. "
                          "please specify detector_size instead or use EMPAD DataSet",
                          DeprecationWarning)
            if detector_size is not None:
                raise ValueError("cannot specify both detector_size and crop_detector_to")
            if detector_size_raw != crop_detector_to:
                raise ValueError("RawFileDataSet can't crop detector anymore, "
                                 "please use EMPAD DataSet")
            detector_size = crop_detector_to

        if detector_size is None:
            raise TypeError("missing 1 required argument: 'detector_size'")

        self._path = path
        self._scan_size = tuple(scan_size)
        self._detector_size = tuple(detector_size)

        self._sig_dims = len(self._detector_size)
        shape = Shape(self._scan_size + self._detector_size, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(dtype),
        )
        self._filesize = None
        self._enable_direct = enable_direct

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)
        return self

    def _get_filesize(self):
        return os.stat(self._path).st_size

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    @classmethod
    def get_msg_converter(cls):
        return RAWDatasetParams

    def _get_fileset(self):
        num_frames = self._meta.shape.flatten_nav()[0]
        return RawFileSet([
            RawFile(
                path=self._path,
                start_idx=0,
                end_idx=num_frames,
                sig_shape=self.shape.sig,
                native_dtype=self._meta.raw_dtype,
            )
        ])

    def check_valid(self):
        if self._enable_direct and not hasattr(os, 'O_DIRECT'):
            raise DataSetException("LiberTEM currently only supports Direct I/O on Linux")
        try:
            fileset = self._get_fileset()
            with fileset:
                return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_cache_key(self):
        return {
            "path": self._path,
            # scan_size + detector_size; included because changing scan_size will change
            # the partition structure and cause errors
            "shape": tuple(self.shape),
            "dtype": str(self.dtype),
        }

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB (converted float data) per partition
        partition_size_px = 1024 * 1024 * 1024 // 4
        total_size_px = np.prod(self.shape)
        res = max(self._cores, total_size_px // partition_size_px)
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in BasePartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield RawPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<RawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)


class RawPartition(BasePartition):
    pass
