import os
import logging

from ncempy.io.dm import fileDM
import numpy as np

from libertem.common import Shape
from .base import (
    DataSet, FileSet, BasePartition, DataSetException, DataSetMeta,
    LocalFile,
)

log = logging.getLogger(__name__)


def _get_metadata(path):
    fh = fileDM(path, on_memory=True)
    if fh.numObjects == 1:
        idx = 0
    else:
        idx = 1
    return {
        'offset': fh.dataOffset[idx],
        'zsize': fh.zSize[idx],
    }


class StackedDMFile(LocalFile):
    def _mmap_to_array(self, raw_mmap, start, stop):
        res = np.frombuffer(raw_mmap, dtype="uint8")
        itemsize = np.dtype(self._native_dtype).itemsize
        sigsize = int(np.prod(self._sig_shape, dtype=np.int64))
        cutoff = 0
        cutoff += (
            self.num_frames * itemsize * sigsize
        )
        res = res[:cutoff]
        return res.view(dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, start:stop]


class DMFileSet(FileSet):
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
        When reading a stack of dm3/dm4 files, it can be expensive to read in
        all the metadata from all files, which we currently only use for
        getting the offsets and sizes of the main data in each file. If you
        absolutely know that the offsets and sizes are the same for all files,
        you can set this parameter and we will skip reading all metadata but
        the one from the first file.
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
        # per-file cached attributes:
        self._z_sizes = {}
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
            z_size = self._z_sizes[fn]
            f = StackedDMFile(
                path=fn,
                start_idx=start_idx,
                end_idx=start_idx + z_size,
                sig_shape=shape,
                native_dtype=raw_dtype,
                file_header=self._offsets[fn],
            )
            files.append(f)
            start_idx += 1  # FIXME: .nav.size?
        return DMFileSet(files)

    def _get_files(self):
        return self._files

    def _get_scan_size(self):
        if self._scan_size:
            return self._scan_size
        return (sum(self._z_sizes.values()),)

    def _get_filesize(self):
        return sum(
            os.stat(p).st_size
            for p in self._get_files()
        )

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_filesize)

        if self._same_offset:
            metadata = executor.run_function(_get_metadata, self._get_files()[0])
            self._offsets = {
                fn: metadata['offset']
                for fn in self._get_files()
            }
            self._z_sizes = {
                fn: metadata['zsize']
                for fn in self._get_files()
            }
        else:
            metadata = dict(zip(
                self._get_files(),
                executor.map(_get_metadata, self._get_files()),
            ))
            self._offsets = {
                fn: metadata[fn]['offset']
                for fn in self._get_files()
            }
            self._z_sizes = {
                fn: metadata[fn]['zsize']
                for fn in self._get_files()
            }
        self._fileset = executor.run_function(self._get_fileset)
        first_file = next(self._fileset.files_from(0))
        nav_dims = self._get_scan_size()
        shape = nav_dims + tuple(first_file.sig_shape)
        sig_dims = len(first_file.sig_shape)
        self._meta = DataSetMeta(
            shape=Shape(shape, sig_dims=sig_dims),
            raw_dtype=first_file.native_dtype,
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
        for part_slice, start, stop in BasePartition.make_slices(
                shape=self.shape,
                num_partitions=self._get_num_partitions()):
            yield BasePartition(
                meta=self._meta,
                partition_slice=part_slice,
                fileset=self._fileset,
                start_frame=start,
                num_frames=stop - start,
            )

    def __repr__(self):
        return "<DMDataSet for a stack of %d files>" % (len(self._get_files()),)
