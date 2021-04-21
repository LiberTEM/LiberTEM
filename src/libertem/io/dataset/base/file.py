
class File:
    """
    Parameters
    ----------
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
        self._start_idx = start_idx
        self._end_idx = end_idx
        self._mmap = None
        self._raw_mmap = None
        self._file = None
        self._native_dtype = native_dtype
        self._path = path
        self._file_header = file_header
        self._frame_header = frame_header
        self._frame_footer = frame_footer
        self._sig_shape = tuple(sig_shape)
        super().__init__()

    @property
    def file_header_bytes(self):
        return self._file_header

    @property
    def start_idx(self):
        return self._start_idx

    @property
    def end_idx(self):
        return self._end_idx

    @property
    def num_frames(self):
        return self._end_idx - self._start_idx

    @property
    def sig_shape(self):
        return self._sig_shape

    @property
    def native_dtype(self):
        return self._native_dtype

    def open(self):
        raise NotImplementedError()

    def close(self):
        raise NotImplementedError()

    def fileno(self):
        return self._file.fileno()

    def __repr__(self):
        return "<%s %d:%d>" % (self.__class__.__name__, self._start_idx, self._end_idx)
