import os
import glob
import logging
import contextlib

from ncempy.io.dm import fileDM

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
        return fileDM(self._path)

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


class DMFileSet(FileSet3D):
    pass


class DMDataSet(DataSet):
    def __init__(self, path, scan_size=None):
        super().__init__()
        self._path = path
        self._meta = None
        self._scan_size = scan_size
        self._filesize = None
        self._pattern = path  # FIXME: set in initialize?

    def _get_fileset(self):
        start_idx = 0
        files = []
        for fn in self._get_files():
            f = DMFile(path=fn, start_idx=start_idx)
            files.append(f)
            start_idx += f.num_frames
        return DMFileSet(files)

    def _get_files(self):
        return glob.glob(self._pattern)

    def _get_scan_size(self):
        # TODO: in some cases, the scan size needs to be read
        # from the dm file (see k2is reader for details, needs testing!)
        return self._scan_size

    def initialize(self):
        self._filesize = sum(
            os.stat(p).st_size
            for p in self._get_files()
        )
        fileset = self._get_fileset()
        first_file = next(fileset.files_from(0))
        with first_file.get_handle() as f1:
            dmds = f1.getDataset(0)
            data = dmds['data']
            dtype = data.dtype

            nav_dims = self._get_scan_size()

            shape = nav_dims + tuple(data.shape)
            sig_dims = len(data.shape)
            self._meta = DataSetMeta(
                shape=Shape(shape, sig_dims=sig_dims),
                raw_dtype=dtype,
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
            with fileDM(self._path) as f1:
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
