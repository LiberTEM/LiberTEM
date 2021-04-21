import os
import platform
import tempfile

import numpy as np


class WriteHandle:
    """
    Parameters:
    -----------
    path : str
        Full path to the file to be written (should not exist, but the directory should)

    tmp_base_path : str
        Path to a directory where temporary files should be written
        (must be on the same file system as `path`, so maybe)

    part_slice : Slice
        Slice of the object (i.e. Partition) we are writing, to convert
        from global tile coordinates to local slices.

    dtype : numpy dtype
        For which dtype should the file be opened
    """
    def __init__(self, path, tmp_base_path, part_slice, dtype):
        # TODO: support for direct I/O
        # (useful if we have a very high write rate, otherwise not so much; very high being
        # multiple GiB/s)
        self._path = path
        self._tmp_base_path = tmp_base_path
        self._slice = part_slice
        self._dtype = dtype
        self._dest = None
        self._tmp_file = None
        self._aborted = False

    def write_tile(self, tile):
        """
        Write a single `DataTile`
        """
        assert self._tmp_file is not None
        dest_slice = tile.tile_slice.shift(self._slice)
        self._dest[dest_slice.get()] = tile.data

    def write_tiles(self, tiles):
        """
        Write all `tiles`, while yielding each tile for further processing
        """
        for tile in tiles:
            self.write_tile(tile)
            yield tile

    def abort(self):
        self._cleanup()
        self._aborted = True

    def __enter__(self):
        self._open_tmp()
        shape = tuple(self._slice.shape)
        self._dest = np.memmap(self._tmp_file.name, dtype=self._dtype, mode='write', shape=shape)
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None or self._aborted:
            # we have an exception, or abort() was called
            self._cleanup()
            return
        self._finalize()

    def _open_tmp(self):
        """
        Open a temporary file for writing. We pass delete=False, because, on success, we move the
        file to its new name, and don't want it to be deleted. Only in case of errors should
        it be removed (done in _cleanup).

        Filename can be accessed as `self._tmp_file.name`
        """
        assert self._tmp_file is None
        prefix = os.path.basename(".tmp-%s" % self._path)
        self._tmp_file = tempfile.NamedTemporaryFile(
            prefix=prefix,
            dir=self._tmp_base_path,
            delete=False
        )

    def _cleanup(self):
        """
        In case of errors, remove the temporary file.
        """
        self._dest = None
        if self._tmp_file is not None:
            self._tmp_file.close()
            os.unlink(self._tmp_file.name)
            self._tmp_file = None

    def _finalize(self):
        """
        Called in case of success.

        What we need to do here:
         + call msync(2) w/ MS_SYNC to flush changes to filesystem
           (done by numpy when caling `flush` on the memmap object)
         + atomically move partition to final filename
         + call fsync on the destination directory
        """
        self._dest.flush()  # msync

        # to make Windows™ happy:
        self._tmp_file.close()

        # FIXME temporary workaround, see if fixed upstream:
        # https://github.com/numpy/numpy/issues/13510
        mm = self._dest._mmap
        if mm is not None:
            mm.close()

        os.rename(self._tmp_file.name, self._path)
        dest_dir = os.path.dirname(self._path)
        self.sync_dir(dest_dir)
        self._tmp_file = None
        self._dest = None

    def sync_dir(self, path):
        # noop on windows, as:
        # "On Windows… err, there is no clear answer. You can not call FlushFileBuffers()
        # on a directory handle as far as I can see."
        # (from http://blog.httrack.com/blog/2013/11/15/everything-you-always-wanted-to-know-about-fsync/)  # NOQA
        if platform.system() == "Windows":
            return
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
