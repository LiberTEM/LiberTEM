import os
import logging

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, File3D, FileSet3D, Partition3D, DataSetException, DataSetMeta,
)

log = logging.getLogger(__name__)


def _get_offset(path):
    fh = fileDM(path, on_memory=True)
    if fh.numObjects == 1:
        idx = 0
    else:
        idx = 1
    offset = fh.dataOffset[idx]
    return offset


class StackedDMFile(File3D):
    """
    A single file from a stack of dm files; this class reads directly
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
        # FIXME: reusing the buffer only works if there are multiple frames per file.
        # readsize = self._size * self._dtype.itemsize
        # buf = self.get_buffer("readbuf", readsize)
        # FIXME: can we get rid of this buffer? maybe only if dtype matches?
        buf = np.zeros(self._size, dtype=self._dtype)
        self._fh.seek(self._offset)
        size_read = self._fh.readinto(buf)
        assert size_read == self._size * self._dtype.itemsize
        return buf

    def readinto(self, start, stop, out, crop_to=None):
        """
        Read [`start`, `stop`) images from this file into `out`
        """
        assert stop - start == 1
        data = self._read_frame()

        data = data.reshape(self.shape)

        if crop_to is not None:
            # TODO: maybe limit I/O to the cropped region, too?
            data = data[crop_to.get(sig_only=True)]
        out[...] = data


class DMFileSet(FileSet3D):
    pass


class DMDataSet(DataSet):
    """
    Reader for stacks of DM3/DM4 files. Each file should contain a single frame.

    Note
    ----
    This DataSet is not supported in the GUI yet, as the file dialog needs to be
    updated to `properly handle opening series
    <https://github.com/LiberTEM/LiberTEM/issues/498>`_.

    Note
    ----
    Single-file 4D DM files are not yet supported. The use-case would be
    to read DM4 files from the conversion of K2 data, but those data sets
    are actually transposed (nav/sig are swapped).

    That means the data would have to be transposed back into the usual shape,
    which is slow, or algorithms would have to be adapted to work directly on
    transposed data. As an example, applying a mask in the conventional layout
    corresponds to calculating a weighted sum frame along the navigation
    dimension in the transposed layout.

    Since the transposed layout corresponds to a TEM tilt series, support for
    transposed 4D STEM data could have more general applications beyond
    supporting 4D DM4 files. Please contact us if you have a use-case for
    single-file 4D DM files or other applications that process stacks of TEM
    files, and we may add support!

    Note
    ----
    You can use the PyPi package `natsort <https://pypi.org/project/natsort/>`_
    to sort the filenames by their numerical components, this is especially useful
    for filenames without leading zeros.

    Parameters
    ----------

    files : List[str]
        List of paths to the files that should be loaded. The order is important,
        as it determines the order in the navigation axis.

    scan_size : Tuple[int] or None
        By default, the files are loaded as a 3D stack. You can change this
        by specifying the scan size, which reshapes the navigation dimensions.
        Raises a `DataSetException` if the shape is incompatible with the data
        that is loaded.

    same_offset : bool
        When reading a stack of dm3/dm4 files, it can be expensive to read in all
        the metadata from all files, which we currently only use for getting the
        offsets to the main data in each file. If you absolutely know that the offsets
        are the same for all files, you can set this parameter and we will skip reading
        all offsets but the one from the first file.
    """
    def __init__(self, files=None, scan_size=None, same_offset=False):
        super().__init__()
        self._meta = None
        self._same_offset = same_offset
        self._scan_size = tuple(scan_size) if scan_size else scan_size
        self._filesize = None
        self._files = files
        if len(files) == 0:
            raise DataSetException("need at least one file as input!")
        self._fileset = None
        self._offsets = {}

    def _get_fileset(self):
        first_fn = self._get_files()[0]
        first_file = fileDM(first_fn, on_memory=True)
        if first_file.numObjects == 1:
            idx = 0
        else:
            idx = 1
        try:
            raw_dtype = first_file._DM2NPDataType(first_file.dataType[idx])
            shape = (first_file.ySize[idx], first_file.xSize[idx])
        except IndexError as e:
            raise DataSetException("could not determine dtype or signal shape") from e
        start_idx = 0
        files = []
        for fn in self._get_files():
            f = StackedDMFile(
                path=fn, start_idx=start_idx,
                offset=self._offsets[fn],
                shape=shape,
                dtype=raw_dtype,
            )
            files.append(f)
            start_idx += f.num_frames
        return DMFileSet(files)

    def _get_files(self):
        return self._files

    def _get_scan_size(self):
        if self._scan_size:
            return self._scan_size
        return (len(self._get_files()),)

    def _get_filesize(self):
        return sum(
            os.stat(p).st_size
            for p in self._get_files()
        )

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)

        if self._same_offset:
            offset = executor.run_function(_get_offset, self._get_files()[0])
            self._offsets = {
                fn: offset
                for fn in self._get_files()
            }
        else:
            self._offsets = {
                fn: offset
                for offset, fn in zip(
                    executor.map(_get_offset, self._get_files()),
                    self._get_files()
                )
            }
        self._fileset = executor.run_function(self._get_fileset)
        first_file = next(self._fileset.files_from(0))
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
    def get_supported_extensions(cls):
        return set(["dm3", "dm4"])

    @classmethod
    def detect_params(cls, path, executor):
        # FIXME: this doesn't really make sense for file series
        pl = path.lower()
        if pl.endswith(".dm3") or pl.endswith(".dm4"):
            return {
                "parameters": {
                    "files": [path]
                },
            }
        return False

    @property
    def dtype(self):
        return self._meta.raw_dtype

    @property
    def shape(self):
        return self._meta.shape

    def check_valid(self):
        first_fn = self._get_files()[0]
        try:
            with fileDM(first_fn, on_memory=True):
                pass
            if (self._scan_size is not None
                    and np.product(self._scan_size) != len(self._get_files())):
                raise DataSetException("incompatible scan_size")
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
        for part_slice, start, stop in Partition3D.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield Partition3D(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=self._fileset,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<DMDataSet for a stack of %d files>" % (len(self._get_files()),)
