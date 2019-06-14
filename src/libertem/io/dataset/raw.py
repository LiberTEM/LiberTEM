import os
import mmap
import warnings

import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D,
)


class RawFile(File3D):
    def __init__(self, meta, path, enable_direct, enable_mmap):
        super().__init__()
        self._path = path
        self._meta = meta
        self._file = None
        self._mmap = None
        self._enable_direct = enable_direct
        self._enable_mmap = enable_mmap
        self._frame_size = self._meta.shape.sig.size * self._meta.raw_dtype.itemsize

    @property
    def num_frames(self):
        return self._meta.shape.flatten_nav()[0]

    @property
    def start_idx(self):
        return 0

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
            self._mmap = np.frombuffer(raw_data, dtype=self._meta.raw_dtype).reshape(
                (self.num_frames,) + tuple(self._meta.shape.sig)
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
        bytes_read = self._file.readinto(out)
        assert bytes_read == readsize


class RawFileSet(FileSet3D):
    pass


class RawFileDataSet(DataSet):
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

    def _get_fileset(self):
        return RawFileSet([
            RawFile(
                meta=self._meta,
                path=self._path,
                enable_direct=self._enable_direct,
                enable_mmap=not self._enable_direct,
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
