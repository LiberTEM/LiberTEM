import os
import warnings
import mmap
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
            "enable_direct": {
                "type": "boolean"
            },
        },
        "required": ["type", "path", "nav_shape", "sig_shape", "dtype"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "dtype", "nav_shape", "sig_shape", "enable_direct"]
        }
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class RawFile(LocalFile):
    def open(self):
        f = open(self._path, "rb")
        self._file = f
        self._raw_mmap = mmap.mmap(
            fileno=f.fileno(),
            length=0,
            offset=0,
            access=mmap.ACCESS_READ,
        )
        # TODO: self._raw_mmap.madvise(mmap.MADV_HUGEPAGE) - benchmark this!
        itemsize = np.dtype(self._native_dtype).itemsize
        assert self._frame_header % itemsize == 0
        assert self._frame_footer % itemsize == 0
        start = self._frame_header // itemsize
        stop = start + int(np.prod(self._sig_shape))
        if self._raw_mmap.size() % int(np.prod(self._sig_shape)) != 0:
            new_mmap_size = self.num_frames * (
                itemsize * np.prod(self.sig_shape, dtype=np.int64)
            )
            skip_partial_frame = self._raw_mmap.size() - new_mmap_size
            self._raw_mmap = memoryview(self._raw_mmap)[:-skip_partial_frame]
        self._mmap = self._mmap_to_array(self._raw_mmap, start, stop)


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

    >>> ds = ctx.load("raw", path=path_to_raw, nav_shape=(16, 16), sig_shape=(128, 128),
    ...               sync_offset=0, dtype="float32",)

    Parameters
    ----------

    path: str
        Path to the file

    nav_shape: tuple of int
        A n-tuple that specifies the size of the navigation region ((y, x), but
        can also be of length 1 for example for a line scan, or length 3 for
        a data cube, for example)

    sig_shape: tuple of int
        Common case: (height, width); but can be any dimensionality

    sync_offset: int, optional
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    dtype: numpy dtype
        The dtype of the data as it is on disk. Can contain endian indicator, for
        example >u2 for big-endian 16bit data.
    """
    def __init__(self, path, dtype, scan_size=None, detector_size=None, enable_direct=False,
                 detector_size_raw=None, crop_detector_to=None, tileshape=None,
                 nav_shape=None, sig_shape=None, sync_offset=0, io_backend=None):
        super().__init__(io_backend=io_backend)
        # handle backwards-compatability:
        if tileshape is not None:
            warnings.warn(
                "tileshape argument is ignored and will be removed after 0.6.0",
                FutureWarning
            )
        # FIXME execute deprecation after 0.6.0
        if crop_detector_to is not None:
            warnings.warn("crop_detector_to and detector_size_raw are deprecated, "
                          "and will be removed after version 0.6.0. "
                          "please specify sig_shape instead or use a more "
                          "specific DataSet like EMPAD",
                          FutureWarning)
            if detector_size is not None:
                raise ValueError("cannot specify both detector_size and crop_detector_to")
            if detector_size_raw != crop_detector_to:
                raise ValueError("RawFileDataSet can't crop detector anymore, "
                                 "please use EMPAD DataSet")
            detector_size = crop_detector_to
        self._nav_shape = tuple(nav_shape) if nav_shape else nav_shape
        self._sig_shape = tuple(sig_shape) if sig_shape else sig_shape
        self._sync_offset = sync_offset
        # handle backwards-compatability:
        if scan_size is not None:
            warnings.warn(
                "scan_size argument is deprecated. please specify nav_shape instead",
                FutureWarning
            )
            if nav_shape is not None:
                raise ValueError("cannot specify both scan_size and nav_shape")
            self._nav_shape = scan_size
        if detector_size is not None:
            warnings.warn(
                "detector_size argument is deprecated. please specify sig_shape instead",
                FutureWarning
            )
            if sig_shape is not None:
                raise ValueError("cannot specify both detector_size and sig_shape")
            self._sig_shape = detector_size
        if self._nav_shape is None:
            raise TypeError("missing 1 required argument: 'nav_shape'")
        if self._sig_shape is None:
            raise TypeError("missing 1 required argument: 'sig_shape'")
        self._path = path
        self._sig_dims = len(self._sig_shape)
        self._dtype = dtype
        if enable_direct:
            warnings.warn(
                "enable_direct is ignored for now, this may be re-added as a separate backend",
                FutureWarning
            )
        self._enable_direct = enable_direct
        self._filesize = None

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)
        if int(np.prod(self._sig_shape)) > int(self._filesize / np.dtype(self._dtype).itemsize):
            raise DataSetException(
                "sig_shape must be less than size: %s" % (
                    int(self._filesize / np.dtype(self._dtype).itemsize)
                )
            )
        self._image_count = int(
            self._filesize / (
                np.dtype(self._dtype).itemsize * np.prod(self._sig_shape, dtype=np.int64)
            )
        )
        self._nav_shape_product = int(np.prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(self._dtype),
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
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
        return RawFileSet([
            RawFile(
                path=self._path,
                start_idx=0,
                end_idx=self._image_count,
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
            # nav_shape + sig_shape; included because changing nav_shape will change
            # the partition structure and cause errors
            "shape": tuple(self.shape),
            "dtype": str(self.dtype),
            "sync_offset": self._sync_offset,
        }

    def get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB (converted float data) per partition
        partition_size_px = 1024 * 1024 * 1024 // 4
        total_size_px = np.prod(self.shape, dtype=np.int64)
        res = max(self._cores, total_size_px // partition_size_px)
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in self.get_slices():
            yield RawPartition(
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<RawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)


class RawPartition(BasePartition):
    pass
