import os
import glob
import logging
import contextlib

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, File3D, FileSet3D, Partition3D, DataSetException, DataSetMeta,
)

log = logging.getLogger(__name__)


class DMFile(File3D):
    def __init__(self, path, start_idx):
        super().__init__()
        self._path = path
        self._start_idx = start_idx

    def _get_handle(self):
        return fileDM(self._path, on_memory=True)

    @contextlib.contextmanager
    def get_handle(self):
        with self._get_handle() as f:
            yield f

    @property
    def num_frames(self):
        with self._get_handle() as f1:
            return max(f1.numObjects - 1, 1)

    @property
    def start_idx(self):
        return self._start_idx

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read [`start`, `stop`) images from this file into `out`
        """
        with self._get_handle() as f1:
            num_images = max(f1.numObjects - 1, 1)
            assert start < num_images
            assert stop <= num_images
            assert stop >= start
            for ii in range(start, stop):
                data = f1.getDataset(ii)['data']
                if crop_to is not None:
                    # TODO: maybe limit I/O to the cropped region, too?
                    data = data[crop_to.get(sig_only=True)]
                out[ii - start, ...] = data


class StackedDMFile(File3D):
    """
    a single file from a stack of dm files; this class reads directly
    using offset and size, bypassing the reading of tag structures etc.
    """
    def __init__(self, path, start_idx, offset, shape, dtype):
        """
        Parameters
        ----------

        path : string

        start_idx : int

        offset : int

        shape : (int, int)

        dtype : numpy dtype
        """
        super().__init__()
        self._path = path
        self._start_idx = start_idx
        self._offset = offset
        self._shape = shape
        self._size = int(np.product(shape))
        self._dtype = np.dtype(dtype)
        self._fh = None

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    def open(self):
        self._fh = open(self._path, "rb")

    def close(self):
        self._fh.close()
        self._fh = None

    @property
    def num_frames(self):
        """
        the stacked format contains one frame per dm file
        """
        return 1

    @property
    def start_idx(self):
        return self._start_idx

    def _read_frame(self):
        readsize = self._size * self._dtype.itemsize
        # FIXME: can we get rid of this buffer? maybe only if dtype matches?
        buf = self.get_buffer("readbuf", readsize)
        self._fh.readinto(buf)
        return np.frombuffer(buf, dtype=self._dtype)

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read [`start`, `stop`) images from this file into `out`
        """
        assert stop - start == 1
        data = self._read_frame()

        data = data.reshape((1,) + self.shape)

        if crop_to is not None:
            # TODO: maybe limit I/O to the cropped region, too?
            data = data[crop_to.get(sig_only=True)]
        out[...] = data


class DMFileSet(FileSet3D):
    # NOTE: we disable opening all files in the fileset, as it can cause file handle exhaustion
    # on data sets with many files
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


class DMDataSet(DataSet):
    def __init__(self, path, scan_size=None, stack=False):
        super().__init__()
        self._path = path
        self._meta = None
        self._scan_size = scan_size
        self._filesize = None
        self._stack = stack

    def _get_fileset(self):
        start_idx = 0
        files = []
        for fn in self._get_files():
            f = DMFile(path=fn, start_idx=start_idx)
            files.append(f)
            start_idx += f.num_frames
        return DMFileSet(files)

    def _get_stacked_fileset(self):
        first_fn = self._get_files()[0]
        first_file = fileDM(first_fn, on_memory=True)
        if first_file.numObjects == 1:
            idx = 0
        else:
            idx = 1
        raw_dtype = first_file._DM2NPDataType(first_file.dataType[idx])
        offset = first_file.dataOffset[idx]
        shape = (first_file.ySize[idx], first_file.xSize[idx])

        start_idx = 0
        files = []
        for fn in self._get_files():
            f = StackedDMFile(
                path=fn, start_idx=start_idx,
                offset=offset,
                shape=shape,
                dtype=raw_dtype,
            )
            files.append(f)
            start_idx += f.num_frames
        return DMFileSet(files)

    def _get_files(self):
        if self._stack:
            # FIXME: sort numerically
            # (try to match a pattern of .*[0-9]+\.dm[34] and extract the number as
            # integer and sort)
            return glob.glob(self._path)
        else:
            return [self.path]

    def _get_scan_size(self):
        # TODO: in some cases, the scan size needs to be read
        # from the dm file (see k2is reader for details, needs testing!)
        if self._stack:
            return (len(self._get_files()),)
        return self._scan_size

    def initialize(self):
        self._filesize = sum(
            os.stat(p).st_size
            for p in self._get_files()
        )
        if self._stack:
            fileset = self._get_stacked_fileset()
        else:
            fileset = self._get_fileset()
        first_file = next(fileset.files_from(0))
        nav_dims = self._get_scan_size()
        shape = nav_dims + tuple(first_file.shape)
        sig_dims = len(first_file.shape)
        self._meta = DataSetMeta(
            shape=Shape(shape, sig_dims=sig_dims),
            raw_dtype=first_file.dtype,
            iocaps={"FULL_FRAMES"},
        )
        return self

    @classmethod
    def detect_params(cls, path):
        pl = path.lower()
        if pl.endswith(".dm3") or pl.endswith(".dm4"):
            return {"path": path}
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        try:
            with fileDM(self._path, on_memory=True) as f1:
                if f1.head['ValidNumberElements'] == 0:
                    raise DataSetException("no data found in file")
                if f1.head['DataTypeID'] not in (0x4120, 0x4122):
                    raise DataSetException("unknown datatype id: %s" % f1.head['DataTypeID'])
            return True
        except (IOError, OSError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_num_partitions(self):
        """
        returns the number of partitions the dataset should be split into
        """
        # let's try to aim for 512MB per partition
        res = max(self._cores, self._filesize // (512*1024*1024))
        return res

    def get_partitions(self):
        if self._stack:
            fileset = self._get_stacked_fileset()
        else:
            fileset = self._get_fileset()
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=fileset,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<DMDataSet for %s>" % (self._path,)
