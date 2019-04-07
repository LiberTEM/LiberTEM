import os

import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, DataSetException, DataSetMeta,
    Partition3D, File3D, FileSet3D,
)


class DirectRawFile(File3D):
    def __init__(self, meta, path, enable_direct):
        self._path = path
        self._meta = meta
        self._file = None
        self._enable_direct = enable_direct
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

    def close(self):
        self._file.close()

    def readinto(self, start, stop, out, crop_to=None):
        offset = start * self._frame_size
        try:
            self._file.seek(offset)
        except OSError as e:
            raise DataSetException("could not seek to offset {}: {}".format(offset, e)) from e
        readsize = (stop - start) * self._frame_size
        bytes_read = self._file.readinto(out)
        assert bytes_read == readsize


class DirectRawFileDataSet(DataSet):
    def __init__(self, path, scan_size, dtype, detector_size, stackheight, enable_direct=True):
        self._path = path
        self._scan_size = tuple(scan_size)
        self._detector_size = detector_size
        self._stackheight = stackheight
        self._sig_dims = len(self._detector_size)
        shape = Shape(self._scan_size + self._detector_size, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
            shape=shape,
            dtype=np.dtype(dtype)
        )
        self._filesize = None
        self._enable_direct = enable_direct

    def initialize(self):
        self._filesize = os.stat(self._path).st_size
        return self

    @property
    def dtype(self):
        return self._meta.dtype

    @property
    def shape(self):
        return self._meta.shape

    def _get_fileset(self):
        return FileSet3D([
            DirectRawFile(
                meta=self._meta,
                path=self._path,
                enable_direct=self._enable_direct,
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
        res = max(1, self._filesize // (1024*1024*1024))
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
        return "<DirectRawFileDataSet of %s shape=%s>" % (self.dtype, self.shape)
