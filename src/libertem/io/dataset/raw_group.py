import os
import pathlib
import itertools
import numpy as np
from typing import Union, TYPE_CHECKING, Tuple, List, Dict, Optional

from libertem.common.math import prod
from libertem.common import Shape
from .base import DataSetMeta, DataSetException
from libertem.io.dataset.raw import RawFileDataSet, RawFileSet, RawFile
from libertem.io.dataset.base import MMapBackend

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
        _filesizes_list: List[int] = executor.map(self._get_filesize, self._paths)
        _filesizes = {p: f for p, f in zip(self._paths, _filesizes_list)}
        self._filesize = sum(_filesizes.values())
        self._image_counts = tuple(self._frames_per_file(path, filesize=filesize)
                                   for path, filesize in _filesizes.items())
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

    def _get_filesizes(self) -> Dict[Union[str, pathlib.Path], int]:
        """
        Compute the filesize of each file in self._paths

        This should be the only time os.stat is called on each path
        during initialization

        In a network filesystem context this can be quite slow, might be
        with running this method in a threaded or async way
        """
        return {p: self._get_filesize(p) for p in self._paths}

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
        Check the fileset for validity in groups of MAX_OPEN files

        Necessary to avoid a 'too many open files' error when
        opening very large datasets

        :meta private:
        """
        MAX_OPEN = 1000
        try:
            backend = self.get_io_backend().get_impl()
            fileset = self._get_fileset()
            for start in range(0, len(fileset), MAX_OPEN):
                with backend.open_files(fileset[start:start + MAX_OPEN]):
                    pass
            return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)
