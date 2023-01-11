import os
import pathlib
import numpy as np

from .shape import Shape
from .math import prod

import typing
import numpy.typing as nt

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.tiling import DataTile


class FileWriter:
    implements = None

    def __init__(self, path: os.PathLike, shape: Shape, dtype: nt.DTypeLike):
        self.path = pathlib.Path(path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self._file_obj = None

    def __enter__(self):
        raise NotImplementedError()

    def __exit__(self, type, value, traceback):
        raise NotImplementedError()

    def write(self):
        raise NotImplementedError()


class NumpyWriter(FileWriter):
    implements = 'npy'

    def __enter__(self):
        self._file_obj = np.lib.format.open_memmap(
            self.path,
            shape=tuple(self.shape),
            mode='w+',
            dtype=self.dtype
        )
        return self

    def __exit__(self, type, value, traceback):
        self._file_obj.flush()
        self._file_obj = None

    def write(self, part_data: 'DataTile'):
        if self._file_obj is None:
            raise RuntimeError('Cannot write to file outside of context manager')

        flat_nav_shape = self.shape.flatten_nav().to_tuple()
        start_coord = tuple([a] for a in part_data.tile_slice.origin)
        start_idx = np.ravel_multi_index(start_coord, flat_nav_shape).item()
        end_idx = start_idx + part_data.size

        self._file_obj: np.ndarray
        self._file_obj.flat[start_idx: end_idx] = part_data.ravel()


class RawWriter(FileWriter):
    implements = 'raw'

    def __enter__(self):
        self._file_obj = open(self.path, 'wb')
        self._max_byte = 0
        return self

    def __exit__(self, type, value, traceback):
        # If we have a positive sync offset need to pad the end of file
        target_length = prod(self.shape) * np.dtype(self.dtype).itemsize
        if self._max_byte < target_length:
            self._file_obj.write(b'\x00' * (target_length - self._max_byte))
        self._file_obj.flush()
        self._file_obj.close()
        self._file_obj = None
        self._max_byte = 0

    def write(self, part_data: 'DataTile'):
        if self._file_obj is None:
            raise RuntimeError('Cannot write to file outside of context manager')

        flat_nav_shape = self.shape.flatten_nav().to_tuple()
        start_coord = tuple([a] for a in part_data.tile_slice.origin)
        start_idx = np.ravel_multi_index(start_coord, flat_nav_shape)
        start_byte = start_idx.item() * np.dtype(self.dtype).itemsize

        # Assuming here that seeking beyond end of file will zero-pad
        self._file_obj.seek(start_byte)
        self._file_obj.write(part_data.ravel().tobytes())
        # Manually track max byte written in case writes are out-of-order
        self._max_byte = max(self._max_byte, self._file_obj.tell())


file_writers = {
    c.implements: c
    for c in (
        NumpyWriter,
        RawWriter,
    )
}
