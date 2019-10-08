import os
import mmap
import warnings

import numpy as np

from libertem.common import Shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D,
)


class RAWDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/RAWDatasetParams.schema.json",
        "title": "RAWDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "raw"},
            "path": {"type": "string"},
            "dtype": {"type": "string"},
            "scan_size": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 2,
                "maxItems": 2
            },
            "detector_size": {
                "type": "array",
                "items": {"type": "number"},
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
            for k in ["path", "dtype", "scan_size", "detector_size"]
        }
        return data


class RawFile(File3D):
    def __init__(self, path, num_frames, frame_shape, enable_direct, enable_mmap, dtype,
                 start_idx=0):
        """
        Parameters
        ----------
        path : str
            path to the file

        num_frames : int
            how many frames should be read from the file?

        frame_shape : tuple of int
            shape of frames, will be used for mmap shape

        enable_direct : bool
            enable direct I/O?

        enable_mmap : bool
            enable memory-mapped reading?

        dtype : str or numpy.dtype
            dtype on disk

        start_idx : int
            at which frame index should we start reading?
        """
        super().__init__()

        self._num_frames = num_frames
        self._dtype = np.dtype(dtype)
        self._frame_size = np.product(frame_shape) * self._dtype.itemsize
        self._frame_shape = frame_shape
        self._start_idx = start_idx

        self._path = path
        self._file = None
        self._mmap = None
        self._enable_direct = enable_direct
        self._enable_mmap = enable_mmap

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def num_frames(self):
        return self._num_frames

    def open(self):
        if self._enable_direct:
            fh = os.open(self._path, os.O_RDONLY | os.O_DIRECT)
            f = open(fh, "rb", buffering=0)
        else:
            f = open(self._path, "rb")
        self._file = f
        if self._enable_mmap:
            raw_data = mmap.mmap(
                fileno=f.fileno(),
                length=self.num_frames * self._frame_size,
                offset=self.start_idx * self.num_frames,
                access=mmap.ACCESS_READ,
            )
            self._mmap = np.frombuffer(raw_data, dtype=self._dtype).reshape(
                (self.num_frames,) + self._frame_shape
            )

    def mmap(self):
        return self._mmap

    def close(self):
        self._file.close()
        self._file = None
        self._mmap = None

    def readinto(self, start, stop, out, crop_to=None):
        offset = start * self._frame_size
        try:
            self._file.seek(offset)
        except OSError as e:
            raise DataSetException("could not seek to offset {}: {}".format(offset, e)) from e
        readsize = (stop - start) * self._frame_size
        if out.dtype != self._dtype:
            rawbuf = self.get_buffer("raw_read_buffer", readsize)
            buf = np.frombuffer(rawbuf, dtype=self._dtype).reshape(out.shape)
            bytes_read = self._file.readinto(buf)
            out[:] = buf
        else:
            bytes_read = self._file.readinto(out)
        assert bytes_read == readsize


class RawFileSet(FileSet3D):
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

        if crop_detector_to is not None:
            warnings.warn("crop_detector_to and detector_size_raw are deprecated, "
                          "please specify detecros_size instead or use EMPAD DataSet",
                          DeprecationWarning)
            if detector_size is not None:
                raise ValueError("cannot specify both detector_size and crop_detector_to")
            if detector_size_raw != crop_detector_to:
                raise ValueError("RawFileDataSet can't crop detector anymore, "
                                 "please use EMPAD DataSet")
            detector_size = crop_detector_to

        self._path = path
        self._scan_size = tuple(scan_size)
        self._detector_size = tuple(detector_size)

        # FIXME: should we allow tuning this in some way? I don't like tuning it per-dataset;
        # maybe we can come up with something better. for now, set automatically.
        self._stackheight = None

        self._sig_dims = len(self._detector_size)
        shape = Shape(self._scan_size + self._detector_size, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            raw_dtype=np.dtype(dtype),
            iocaps={"DIRECT", "MMAP", "FULL_FRAMES"},
        )
        if enable_direct:
            self._meta.iocaps.remove("MMAP")
        self._filesize = None
        self._enable_direct = enable_direct

    def initialize(self):
        self._filesize = os.stat(self._path).st_size
        return self

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
        frame_shape = tuple(self._meta.shape.sig)
        num_frames = self._meta.shape.flatten_nav()[0]
        return RawFileSet([
            RawFile(
                path=self._path,
                num_frames=num_frames,
                frame_shape=frame_shape,
                enable_direct=self._enable_direct,
                enable_mmap=not self._enable_direct,
                dtype=self._meta.raw_dtype,
                start_idx=0,
            )
        ])

    def check_valid(self):
        try:
            fileset = self._get_fileset()
            with fileset:
                return True
        except (IOError, OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 1024MB per partition
        res = max(self._cores, self._filesize // (1024*1024*1024))
        return res

    def get_partitions(self):
        fileset = self._get_fileset()
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                stackheight=self._stackheight,
                meta=self._meta,
                fileset=fileset,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<RawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)
