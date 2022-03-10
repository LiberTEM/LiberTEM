import os
import itertools
import numpy as np

from libertem.common.math import prod
from libertem.common import Shape
from .base import DataSetMeta, DataSetException
from libertem.io.dataset.raw import RawFileDataSet, RawFileSet, RawFile
from libertem.io.dataset.base import MMapBackend


class RawFileGroupDataSet(RawFileDataSet):
    def __init__(self, paths, *args, file_header=0, frame_header=0, frame_footer=0, **kwargs):
        super().__init__(paths[0], *args, **kwargs)
        self._path = None
        self._paths = paths

        self._file_header = file_header
        self._frame_header = frame_header
        self._frame_footer = frame_footer

    def initialize(self, executor):
        self._filesize = executor.run_function(self._get_total_filesize)
        self._image_counts = executor.run_function(self._get_image_counts)
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

    def _get_fileset(self):
        end_idxs = tuple(itertools.accumulate(self._image_counts))
        start_idxs = (0,) + end_idxs[:-1]
        return RawFileSet([RawFile(path=p,
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

    def _get_filesize(self, path):
        return os.stat(path).st_size

    def _get_total_filesize(self):
        return sum(self._get_filesize(p) for p in self._paths)

    def _frames_per_file(self, path):
        frame_size = (self._frame_header
                      + np.dtype(self._dtype).itemsize * prod(self._sig_shape)
                      + self._frame_footer)
        nframes = (self._get_filesize(path) - self._file_header) / frame_size
        if nframes % 1 != 0:
            raise DataSetException(f"File {path} has size inconsistent with supplied parameters")
        return int(nframes)

    def _get_image_counts(self):
        return tuple(self._frames_per_file(p) for p in self._paths)

    def check_valid(self):
        try:
            fileset = self._get_fileset()
            backend = self.get_io_backend().get_impl()
            with backend.open_files(fileset):
                return True
        except (OSError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)
