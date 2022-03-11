import os
import pathlib
import logging
import itertools
import numpy as np
from typing import Union, TYPE_CHECKING, Tuple, List, Optional

from libertem.common.math import prod
from libertem.common import Shape
from .base import DataSetMeta, DataSetException
from libertem.io.dataset.raw import RawFileDataSet, RawFileSet, RawFile
from libertem.io.dataset.base import MMapBackend
from libertem.io.dataset.base.partition import BasePartition

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    import numpy.typing as nt


class RawFileGroupSet(RawFileSet):
    def __getitem__(self, idx):
        """
        Slices into fileset return a sub-fileset with same parameters
        Indexing into fileset returns the file
        """
        _files = super().__getitem__(idx)
        if isinstance(_files, list):
            _files = RawFileGroupSet(_files,
                                     frame_header_bytes=self._frame_header_bytes,
                                     frame_footer_bytes=self._frame_footer_bytes)
        return _files


class RawFileGroupDataSet(RawFileDataSet):
    # Maximum number of files to open at one time
    # or to allow in a single partition
    MAX_OPEN_FILES = 256

    def __init__(self,
                 paths: List[Union[str, pathlib.Path]],
                 *,
                 dtype: "nt.DTypeLike",
                 nav_shape: Tuple[int, ...] = None,
                 sig_shape: Tuple[int, ...] = None,
                 file_header: int = 0,
                 frame_header: int = 0,
                 frame_footer: int = 0,
                 **kwargs):
        """
        Wrapper around RawFileDataSet providing capability to read sets of raw
        files as a single dataset, and adds support for file/frame headers and frame footers.

        No decoding of headers or footers is performed.

        Files in the set can each contain one-or-more frames (and indeed a variable)
        number of frames per file, as long as file_header, frame_header and frame_footer
        are all constant throughout.

        It is the user's responsibility to ensure that the supplied paths are
        sorted according to a flat navigation dimension in C-ordering.

        See :class:`~libertem.io.dataset.raw.RawFileDataSet` for the majority of
        the signature, only crucial arguments and those specific to this class
        are specified below.

        Parameters
        ----------
        paths : list[Union[str, pathlib.Path]]
            List of file paths to interpret as the dataset.
        dtype : nt.DTypeLike
            The dtype of the data as stored on disk
        nav_shape : tuple[int, ...]
            The shape of the navigation dimensions
        sig_shape : tuple[int, ...]
            The shape of the signal dimensions
        file_header : int, optional
            Bytes to skip at beginning of each file, by default 0
        frame_header : int, optional
            Bytes to skip at beginning of each frame, by default 0
        frame_footer : int, optional
            Bytes to skip at end of each frame, by default 0
        """
        if isinstance(paths, (str, pathlib.Path)):
            paths = [paths]
        super().__init__(paths[0], dtype=dtype,
                         nav_shape=nav_shape, sig_shape=sig_shape,
                         **kwargs)
        self._path = None
        self._paths: List[Union[str, pathlib.Path]] = paths

        self._file_header = file_header
        self._frame_header = frame_header
        self._frame_footer = frame_footer

    def initialize(self, executor) -> 'RawFileGroupDataSet':
        if len(set(self._paths)) != len(self._paths):
            log.warning(f'Paths passed to {self.__class__.__name__} contains duplicates')
        # Stat files in groups up to size chunk_size
        # If we are running in a threaded/distributed executor each chunk of stat
        # calls will be hopefully be parallelised, should not be different from
        # a straight loop when running inline
        chunk_size = self.MAX_OPEN_FILES
        chunked_paths = [self._paths[start:start + chunk_size]
                         for start in range(0, len(self._paths), chunk_size)]
        _filesizes_lists = executor.map(self._get_filesizes, chunked_paths)
        _filesizes = tuple(itertools.chain(*_filesizes_lists))
        self._filesize = sum(v[1] for v in _filesizes)
        self._image_counts = tuple(self._frames_per_file(path, filesize=filesize)
                                   for path, filesize in _filesizes)
        self._image_count = sum(self._image_counts)
        self._nav_shape_product = int(prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=self._sig_dims)
        self._meta = DataSetMeta(
                                 shape=shape,
                                 raw_dtype=np.dtype(self._dtype),
                                 sync_offset=self._sync_offset,
                                 image_count=self._image_count,
        )

        if ((self._frame_header % self.dtype.itemsize or self._frame_footer % self.dtype.itemsize)
                and isinstance(self.get_io_backend(), MMapBackend)):
            raise DataSetException('Cannot have frame header/footer which are '
                                   'not multiples of bytesize of raw_dtype when '
                                   'using MMapBackend. Specifiy another IOBackend '
                                   'or use file_header if single frame-per-file.')

        return self

    def _get_fileset(self) -> RawFileGroupSet:
        """
        Return RawFile descriptors for each file in self._paths
        """
        end_idxs = tuple(itertools.accumulate(self._image_counts))
        start_idxs = (0,) + end_idxs[:-1]
        return RawFileGroupSet([RawFile(path=p,
                                        start_idx=s,
                                        end_idx=e,
                                        sig_shape=self.shape.sig,
                                        native_dtype=self._meta.raw_dtype,
                                        frame_footer=self._frame_footer,
                                        frame_header=self._frame_header,
                                        file_header=self._file_header)
                           for p, s, e in zip(self._paths, start_idxs, end_idxs)],
                        frame_header_bytes=self._frame_header,
                        frame_footer_bytes=self._frame_footer)

    def _get_filesize(self, path: Union[str, pathlib.Path]) -> int:
        """
        Get the file size of a single file
        """
        return os.stat(path).st_size

    def _get_filesizes(self, paths: Union[str, pathlib.Path]) -> List[Tuple[Union[str,
                                                                                  pathlib.Path],
                                                                            int]]:
        """
        Compute the filesizes for a list of paths
        """
        return [(p, self._get_filesize(p)) for p in paths]

    def _frames_per_file(self,
                         path: Union[str, pathlib.Path],
                         filesize: Optional[int] = None) -> int:
        """
        Calculate the number of frames in each file based on its filesize

        Raises if the file size and header/footer parameters are incompatible
        """
        frame_size = (self._frame_header
                      + np.dtype(self._dtype).itemsize * prod(self._sig_shape)
                      + self._frame_footer)
        if filesize is None:
            filesize = self._get_filesize(path)
        nframes = (filesize - self._file_header) / frame_size
        if nframes % 1 != 0:
            raise DataSetException(f"File {path} has size inconsistent with supplied parameters")
        return int(nframes)

    def check_valid(self) -> bool:
        """
        Check the fileset for validity in groups of MAX_OPEN_FILES

        Necessary to avoid a 'too many open files' error when
        opening very large datasets

        :meta private:
        """
        try:
            backend = self.get_io_backend().get_impl()
            fileset = self._get_fileset()
            for start in range(0, len(fileset), self.MAX_OPEN_FILES):
                with backend.open_files(fileset[start:start + self.MAX_OPEN_FILES]):
                    pass
            return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_num_partitions(self) -> int:
        """
        Get the number of partitions for the dataset in a way
        which respects MAX_OPEN_FILES to avoid an OSError

        Will likely still fail if using a threaded executor
        as I think the OS-level max open files is for the whole process

        This can probably be cached or pre-computed once the dataset
        is initialized.

        Returns
        -------
        int
            Number of partitions to split the dataset into
        """
        num_part = super().get_num_partitions()
        frames_cumulative = np.cumsum(self._image_counts)
        files_per_part = self._nfiles_for_partitions(num_part, frames_cumulative)
        while any(nf > self.MAX_OPEN_FILES for nf in files_per_part):
            num_part += 1
            files_per_part = self._nfiles_for_partitions(num_part, frames_cumulative)
        return num_part

    def _nfiles_for_partitions(self, num_part: int, frames_cumulative: np.ndarray) -> List[int]:
        """
        Get the number of files corresponding to each partition

        Parameters
        ----------
        num_part : int
            The number of partitions to split the files into
        frames_cumulative : np.ndarray
            The cumulative number of frames given by the ordered list of paths

        Returns
        -------
        List[int]
            The number of files that map to each partition
        """
        nfiles = []
        for _, start, stop in BasePartition.make_slices(self.meta.shape, num_part):
            nfiles.append(self._files_in_partition(start, stop, frames_cumulative))
        return nfiles

    def _files_in_partition(self, start: int, stop: int, frames_cumulative: np.ndarray) -> int:
        """
        Get the number of files corresponding to a single partition

        Parameters
        ----------
        start : int
            Start flat navigation index for the partition
        stop : int
            End flat navigation index for the partition
        frames_cumulative : np.ndarray
            The cumulative number of frames given by the ordered list of paths

        Returns
        -------
        int
            The number of files in the partition
        """
        start_idx, end_idx = np.searchsorted(frames_cumulative, [start, stop])
        # possible off-by-one but it's not critical
        end_idx = max(end_idx, start_idx + 1)
        return end_idx - start_idx
