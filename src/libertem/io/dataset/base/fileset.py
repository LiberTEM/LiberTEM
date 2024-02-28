import typing

import numpy as np

from libertem.common.math import flat_nonzero
from .file import File
from .utils import FileTree
from .tiling import default_get_read_ranges
from .tiling_scheme import TilingScheme


class FileSet:
    """
    Parameters
    ----------
    files
        files that are part of a partition or dataset
    """
    def __init__(
        self, files: list[File],
        frame_header_bytes: int = 0, frame_footer_bytes: int = 0,
    ):
        self._files = files
        assert len(files) > 0
        self._tree = FileTree.make(files)
        if self._tree is None:
            raise ValueError(str(files))
        # FIXME: maybe should be moved into the array representation
        # if the fileset, taken from individual files
        self._frame_header_bytes = frame_header_bytes
        self._frame_footer_bytes = frame_footer_bytes

    def get_for_range(self, start, stop):
        """
        return new FileSet filtered for files having frames in the [start, stop) range
        """
        files = self._get_files_for_range(start, stop)
        return self._clone(
            files=files,
            frame_header_bytes=self._frame_header_bytes,
            frame_footer_bytes=self._frame_footer_bytes,
        )

    def _clone(self, *args, **kwargs):
        return self.__class__(*args, **kwargs)

    def _get_files_for_range(self, start, stop):
        """
        return new list of files filtered for files having frames in the [start, stop) range
        """
        files = []
        for f in self.files_from(start):
            if f.start_idx > stop:
                break
            files.append(f)
        assert len(files) > 0
        return files

    def files_from(self, start):
        lower_bound, f = self._tree.search_start(start)
        for idx in range(lower_bound, len(self._files)):
            yield self._files[idx]

    def __iter__(self):
        return iter(self._files)

    def __len__(self):
        return len(self._files)

    def __getitem__(self, idx):
        return self._files[idx]

    def __repr__(self):
        return f"<{self.__class__.__name__} {self._files!r}>"

    def get_as_arr(self):
        fileset_arr = np.zeros((len(self), 4), dtype=np.int64)
        for idx, f in enumerate(self._files):
            fileset_arr[idx] = (f.start_idx, f.end_idx, idx, f.file_header_bytes)
        return fileset_arr

    def get_read_ranges(
        self, start_at_frame: int, stop_before_frame: int,
        dtype, tiling_scheme: TilingScheme, sync_offset: int = 0,
        roi: typing.Union[np.ndarray, None] = None,
    ):
        fileset_arr = self.get_as_arr()
        roi_nonzero = None
        if roi is not None:
            roi_nonzero = flat_nonzero(roi).astype(np.int64)
        return default_get_read_ranges(
            start_at_frame=start_at_frame,
            stop_before_frame=stop_before_frame,
            roi_nonzero=roi_nonzero,
            depth=tiling_scheme.depth,
            slices_arr=tiling_scheme.slices_array,
            fileset_arr=fileset_arr,
            sig_shape=tuple(tiling_scheme.dataset_shape.sig),
            sync_offset=sync_offset,
            bpp=np.dtype(dtype).itemsize * 8,
            frame_header_bytes=self._frame_header_bytes,
            frame_footer_bytes=self._frame_footer_bytes,
        )
