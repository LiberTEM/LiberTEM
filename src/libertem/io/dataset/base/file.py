from typing import NamedTuple
import numpy as np

from libertem.common.math import prod


class OffsetsSizes(NamedTuple):
    """
    General slicing for slicing the data out of a file
    (removing file header and per-frame headers)
    """

    # whole-file slicing, in bytes
    file_offset: int  # number of bytes to skip at the beginning of the file
    skip_end: int  # how many bytes to skip at the end

    # per-frame slicing, in number of items in native dtype
    frame_offset: int  # number of items to skip at the beginning of each frame
    frame_size: int  # number of items for each frame


class File:
    """
    A description of a file that is part of a dataset. Contains
    information about the internal structure, like sizes
    of headers, frames, frame headers, frame footers, ...

    Parameters
    ----------
    path : str
        The path of the file. Interpretation may be backend-specific

    start_idx : int
        Start index of signal elements in this file (inclusive),
        in the flattened navigation axis

    end_idx : int
        End index of signal elements in this file (exclusive),
        in the flattened navigation axis

    native_dtype : np.dtype
        The dtype that is used for reading the data. This may
        match the "real" dtype of data, or in some cases, when
        no direct match is possible (decoding is necessary),
        it falls back to bytes (np.uint8)

    sig_shape : Shape | Tuple[int, ...]
        The shape of each signal element

    file_header: int
        Number of bytes to ignore at the beginning of the file

    frame_header: int
        Number of bytes to ignore before each frame

    frame_footer: int
        Number of bytes to ignore after each frame
    """
    def __init__(self, path, start_idx, end_idx,
                native_dtype, sig_shape,
                frame_footer=0, frame_header=0, file_header=0):
        self._start_idx = int(start_idx)
        self._end_idx = int(end_idx)
        self._native_dtype = native_dtype
        self._path = path
        self._file_header = file_header
        self._frame_header = frame_header
        self._frame_footer = frame_footer
        self._sig_shape = tuple(sig_shape)

    @property
    def file_header_bytes(self) -> int:
        return self._file_header

    @property
    def start_idx(self) -> int:
        return self._start_idx

    @property
    def end_idx(self) -> int:
        return self._end_idx

    @property
    def num_frames(self) -> int:
        return self._end_idx - self._start_idx

    @property
    def sig_shape(self) -> tuple[int, ...]:
        return self._sig_shape

    @property
    def native_dtype(self) -> np.dtype:
        return self._native_dtype

    @property
    def path(self) -> str:
        return self._path

    def get_offsets_sizes(self, size: int) -> OffsetsSizes:
        """
        Get file and frame offsets/sizes

        Parameters
        ----------
        size : int
            len(memoryview) for the whole file

        Returns
        -------
        slicing
            The file/frame slicing
        """
        itemsize = np.dtype(self._native_dtype).itemsize
        assert self._frame_header % itemsize == 0
        assert self._frame_footer % itemsize == 0
        frame_size = int(prod(self._sig_shape))
        frame_offset = self._frame_header // itemsize
        file_offset = self._file_header
        skip_end = 0

        # cut off any extra data at the end of the file:
        if size % int(prod(self._sig_shape)):
            new_mmap_size = self.num_frames * (
                (itemsize * frame_size) + self._frame_header + self._frame_footer
            )
            skip_end = (size - file_offset) - new_mmap_size
        assert skip_end >= 0

        return OffsetsSizes(
            file_offset=file_offset,
            skip_end=skip_end,
            frame_offset=frame_offset,
            frame_size=frame_size,
        )

    def get_array_from_memview(self, mem: memoryview, slicing: OffsetsSizes) -> np.ndarray:
        """
        Convert a memoryview of the file's data into an ndarray, cutting away
        frame headers and footers as defined by `start` and `stop` parameters.

        Parameters
        ----------
        mem
            The input memoryview

        start
            Cut off frame headers of this size; usually start = frame_header_bytes // itemsize

        stop
            End index; usually stop = start + prod(sig_shape)

        Returns
        -------
        np.ndarray
            The output array. Should have shape (num_frames, prod(sig_shape)) and native dtype
        """
        if slicing.skip_end > 0:
            mem = mem[slicing.file_offset:-slicing.skip_end]
        else:
            mem = mem[slicing.file_offset:]
        itemsize = np.dtype(self._native_dtype).itemsize
        assert len(mem) % itemsize == 0, \
            "len(mem) must fit the dtype"
        assert len(mem) // itemsize % self.num_frames == 0, \
            "len(mem) must fit the number of frames"
        assert len(mem) // itemsize // self.num_frames % (
            slicing.frame_size + (self._frame_header + self._frame_footer) // itemsize
        ) == 0, "len(mem) must fit the sig shape"
        arr_uncut = np.frombuffer(mem, dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )
        arr = arr_uncut[:, slicing.frame_offset:slicing.frame_offset + slicing.frame_size]
        assert arr.shape[1] == slicing.frame_size, \
            "array shape must fit the signal shape"
        assert arr.size > 0
        return arr

    def __repr__(self):
        return "<%s %d:%d>" % (self.__class__.__name__, self._start_idx, self._end_idx)
