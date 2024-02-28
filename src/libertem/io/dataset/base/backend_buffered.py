import os
import io

import numpy as np
import numba
from numba.typed import Dict
import contextlib
from sparseconverter import CUDA, NUMPY, ArrayBackend

from libertem.common.math import prod
from libertem.io.dataset.base.backend import IOBackend, IOBackendImpl
from libertem.io.dataset.base.fileset import FileSet
from libertem.common import Shape, Slice
from libertem.common.buffers import BufferPool, ManagedBuffer
from libertem.common.numba import cached_njit
from .tiling import DataTile
from .decode import DtypeConversionDecoder

_r_n_d_cache = {}


def _make_buffered_reader_and_decoder(decode):
    """
    decode: from buffers, in bytes, possibly interpreted as native_dtype, to out_decoded.dtype
    """
    @cached_njit(boundscheck=False, nogil=True)
    def _buffered_tilereader(outer_idx, buffers, sig_dims, tile_read_ranges,
                           out_decoded, native_dtype, do_zero,
                           origin, shape, ds_shape, offsets):
        if do_zero:
            out_decoded[:] = 0
        for rr_idx in range(tile_read_ranges.shape[0]):
            rr = tile_read_ranges[rr_idx]
            if rr[1] == rr[2] == 0:
                break
            buf = buffers[rr[0]]
            offset = offsets[rr[0]]  # per-file offset of start of read buffer
            decode(
                inp=buf[rr[1] - offset:rr[2] - offset],
                out=out_decoded,
                idx=rr_idx,
                native_dtype=native_dtype,
                rr=rr,
                origin=origin,
                shape=shape,
                ds_shape=ds_shape,
            )
        return out_decoded
    return _buffered_tilereader


@numba.njit(cache=True, nogil=True)
def block_get_min_fill_factor(rrs):
    """
    Try to find out how sparse the given read ranges are, per file.

    Returns the smallest fill factor and maximum required buffer size.
    """
    # FIXME: replace with fixed arrays? If this fn stands out on a profile...
    min_per_file = {}
    max_per_file = {}
    payload_bytes = {}
    for tile_idx in range(rrs.shape[0]):
        for part_idx in range(rrs.shape[1]):
            part_rr = rrs[tile_idx, part_idx]
            fileno = part_rr[0]

            if part_rr[2] == part_rr[1] == 0:
                continue

            if fileno not in min_per_file:
                min_per_file[fileno] = part_rr[1]
            else:
                min_per_file[fileno] = min(part_rr[1], min_per_file[fileno])

            if fileno not in max_per_file:
                max_per_file[fileno] = part_rr[2]
            else:
                max_per_file[fileno] = max(part_rr[2], max_per_file[fileno])

            if fileno not in payload_bytes:
                payload_bytes[fileno] = 0
            payload_bytes[fileno] += part_rr[2] - part_rr[1]

    min_fill_factor = -1
    max_buffer_size = 0

    for k in min_per_file:
        read_size = max_per_file[k] - min_per_file[k]
        fill_factor = payload_bytes[k] / (read_size)
        # print(k, fill_factor, max_per_file[k], min_per_file[k], payload_bytes[k])
        if fill_factor < min_fill_factor or min_fill_factor == -1:
            min_fill_factor = fill_factor
        if read_size > max_buffer_size:
            max_buffer_size = read_size
    return min_fill_factor, max_buffer_size, min_per_file, max_per_file


class BufferedFile:
    def __init__(self, path, desc):
        self.path = path
        self.desc = desc
        self._handle = None
        self._arr = None

    def get_blocksize(self):
        """
        Return the block size to which reads should be aligned to. In case of normal
        buffered I/O, this can be 1
        """
        return 1

    def open(self):
        # disable internal I/O buffering, as we want to read directly
        # into our own buffer here:
        self._handle = open(self.path, "rb", buffering=0)
        return self

    def close(self):
        self._handle.close()
        self._handle = None

    @property
    def handle(self):
        if self._handle is None:
            raise RuntimeError("trying to access file handle of closed file")
        return self._handle

    def seek(self, offset):
        self._handle.seek(offset)

    def readinto(self, buf):
        BLOCKSIZE = self.get_blocksize()
        buf_orig = buf
        buf = memoryview(buf)
        to_read = len(buf)
        offset = 0
        last_to_read = to_read
        # `readinto` may return early, so we may need to re-run it:
        while to_read > 0:
            # Make sure reads are aligned to the blocksize,
            # after an early return of `readinto` to allow Direct I/O on Windows.
            # On Linux, a misaligned read which turns out to be at the end of the
            # file doesn't cause an error
            blockcount, remainder = divmod(offset, BLOCKSIZE)
            to_read += remainder
            offset = blockcount * BLOCKSIZE
            if remainder > 0:
                self._handle.seek(-remainder, io.SEEK_CUR)
            bytes_read = self._handle.readinto(buf[offset:])
            if bytes_read == 0:
                break
            to_read -= bytes_read
            offset += bytes_read
            # The block aligning code would otherwise
            # try to read a "tail" in an infinite loop if the file size
            # is not a multiple of BLOCKSIZE
            if to_read == last_to_read:
                break
            last_to_read = to_read
        return buf_orig[:offset]


class DirectBufferedFile(BufferedFile):
    def get_blocksize(self):
        """
        Return the block size to which reads should be aligned to. On Linux,
        this should be a multiple of the logical block size of the file system.
        """
        # NOTE: if required, this can be replaced with the appropriate syscall
        # to determine the block size. On Linux, this would be BLKBSZGET.
        return 4096

    def open(self):
        # disable internal I/O buffering, as we want to read directly
        # into our own buffer here:
        if hasattr(os, 'O_DIRECT'):
            self._fh = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            self._handle = open(self._fh, "rb", buffering=0)
        elif os.name == 'nt':
            import win32file
            import msvcrt
            # Make sure to keep a reference, otherwise
            # it will be closed
            self._fh = win32file.CreateFile(
                self.path,  # fileName
                win32file.GENERIC_READ,  # desiredAccess
                win32file.FILE_SHARE_READ
                | win32file.FILE_SHARE_WRITE
                | win32file.FILE_SHARE_DELETE,  # shareMode
                None,  # attributes
                win32file.OPEN_EXISTING,  # CreationDisposition
                # O_DIRECT equivalent (?)
                win32file.FILE_FLAG_NO_BUFFERING,  # flagsAndAttributes
                0,  # hTemplateFile
            )
            fd = msvcrt.open_osfhandle(int(self._fh), os.O_RDONLY)
            self._handle = os.fdopen(fd, "rb", buffering=0)
        else:
            raise RuntimeError("Direct I/O not supported on this platform.")
        return self

    def close(self):
        super().close()
        self._fh = None


class BufferedBackend(IOBackend, id_="buffered"):
    """
    I/O backend using a buffered reading strategy. Useful for slower media
    like HDDs, where seeks cause performance drops. Used by default
    on Windows.

    This does not perform optimally on SSDs under all circumstances, for
    better best-case performance, try using
    :class:`~libertem.io.dataset.base.MMapBackend` instead.

    Parameters
    ----------
    max_buffer_size : int
        Maximum buffer size, in bytes. This is passed to the tileshape
        negotiation to select the right depth.
    """
    def __init__(self, max_buffer_size=16*1024*1024):
        self._max_buffer_size = max_buffer_size

    @classmethod
    def from_json(cls, msg):
        """
        Construct an instance from the already-decoded `msg`.
        """
        raise NotImplementedError("TODO! implement me!")

    def get_impl(self):
        return BufferedBackendImpl(
            max_buffer_size=self._max_buffer_size,
        )


class BufferedBackendImpl(IOBackendImpl):
    def __init__(self, max_buffer_size, direct_io=False):
        super().__init__()
        self._max_buffer_size = max_buffer_size
        self._direct_io = direct_io
        self._buffer_pool = BufferPool()

    @contextlib.contextmanager
    def open_files(self, fileset: FileSet):
        cls: type[BufferedFile]
        if self._direct_io:
            cls = DirectBufferedFile
        else:
            cls = BufferedFile
        files = [
            cls(path=f.path, desc=f).open()
            for f in fileset
        ]
        yield files
        for f in files:
            f.close()

    def need_copy(
        self, decoder, roi, native_dtype, read_dtype, tiling_scheme=None, fileset=None,
        sync_offset=0, corrections=None,
    ):
        return True  # we always copy in this backend

    def get_read_and_decode(self, decode):
        key = (decode, "read")
        if key in _r_n_d_cache:
            return _r_n_d_cache[key]
        r_n_d = _make_buffered_reader_and_decoder(decode=decode)
        _r_n_d_cache[key] = r_n_d
        return r_n_d

    def get_max_io_size(self):
        return self._max_buffer_size

    def _get_tiles_by_block(
        self, tiling_scheme, open_files, read_ranges, read_dtype, native_dtype, decoder=None,
        corrections=None, sync_offset=0,
    ):
        if decoder is None:
            decoder = DtypeConversionDecoder()
        decode = decoder.get_decode(
            native_dtype=np.dtype(native_dtype),
            read_dtype=np.dtype(read_dtype),
        )
        r_n_d = self._r_n_d = self.get_read_and_decode(decode)

        native_dtype = decoder.get_native_dtype(native_dtype, read_dtype)

        sig_dims = tiling_scheme.shape.sig.dims
        ds_shape = np.array(tiling_scheme.dataset_shape)

        largest_slice = sorted((
            (prod(s_.shape), s_)
            for _, s_ in tiling_scheme.slices
        ), key=lambda x: x[0], reverse=True)[0][1]

        buf_shape = (tiling_scheme.depth,) + tuple(largest_slice.shape)

        need_clear = decoder.do_clear()

        slices = read_ranges[0]
        # Use NumPy prod for multidimensional array and axis parameter
        shape_prods = np.prod(slices[..., 1, :], axis=1, dtype=np.int64)
        ranges = read_ranges[1]
        scheme_indices = read_ranges[2]
        tile_block_size = len(tiling_scheme)

        with self._buffer_pool.empty(buf_shape, dtype=read_dtype) as out_decoded:
            out_decoded = out_decoded.reshape((-1,))
            for block_idx in range(0, slices.shape[0], tile_block_size):
                block_ranges = ranges[block_idx:block_idx + tile_block_size]

                fill_factor, req_buf_size, min_per_file, max_per_file = block_get_min_fill_factor(
                    block_ranges
                )
                # TODO: if it makes sense, implement sparse variant
                # if req_buf_size > self._max_buffer_size or fill_factor < self._sparse_threshold:
                yield from self._read_block_dense(
                    block_idx, tile_block_size, min_per_file, max_per_file, open_files,
                    slices, ranges, scheme_indices, shape_prods, out_decoded, r_n_d,
                    sig_dims, ds_shape, need_clear, native_dtype, corrections,
                )

    def _read_block_dense(
        self, block_idx, tile_block_size, min_per_file, max_per_file, open_files,
        slices, ranges, scheme_indices, shape_prods, out_decoded, r_n_d,
        sig_dims, ds_shape, need_clear, native_dtype, corrections,
    ):
        """
        Reads a block of tiles, starting at `block_idx`, having a size of
        `tile_block_size` read range entries.
        """
        # phase 1: read
        buffers = Dict()

        # this list manages the lifetime of the ManagedBuffer instances;
        # after `buf_ref` goes out of scope, the buffers are returned to
        # the buffer pool, so make sure that this matches with the usage
        # of the buffers!
        buf_ref = []
        for fileno in min_per_file.keys():
            fh = open_files[fileno]
            # add align_to to allow for alignment cut:
            align_to = fh.get_blocksize()
            read_size = max_per_file[fileno] - min_per_file[fileno] + align_to
            # ManagedBuffer gives us memory in 4k blocks, so the size is 4k aligned
            mb = ManagedBuffer(self._buffer_pool, read_size, alignment=fh.get_blocksize())
            arr = np.frombuffer(mb.buf, dtype=np.uint8)
            buf_ref.append(mb)
            seek_pos = min_per_file[fileno]
            alignment = 0
            # seek_pos needs to be aligned to 4k block size, too:
            if seek_pos % align_to != 0:
                alignment = seek_pos % align_to
                seek_pos = align_to * (seek_pos // align_to)
            fh.seek(seek_pos)
            read_result = fh.readinto(arr)
            # read may be truncated, if the buffer is larger than the file; we
            # truncate the buffer, too, to make sure we don't use any
            # uninitialized values. Also cut off `alignment` bytes at the beginning,
            # which were read to make O_DIRECT happy:
            buffers[fileno] = read_result[alignment:]

        # phase 2: decode tiles from the data that was read
        for idx in range(block_idx, block_idx + tile_block_size):
            origin = slices[idx, 0]
            shape = slices[idx, 1]
            tile_ranges = ranges[idx]
            scheme_idx = scheme_indices[idx]
            out_cut = out_decoded[:shape_prods[idx]].reshape((shape[0], -1))

            data = r_n_d(
                idx,
                buffers, sig_dims, tile_ranges,
                out_cut, native_dtype, do_zero=need_clear,
                origin=origin, shape=shape, ds_shape=ds_shape,
                offsets=min_per_file,
            )
            tile_slice = Slice(
                origin=origin,
                shape=Shape(shape, sig_dims=sig_dims)
            )
            data = data.reshape(shape)
            self.preprocess(data, tile_slice, corrections)
            yield DataTile(
                data,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )

    def get_tiles(
        self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype, decoder,
        sync_offset, corrections, array_backend: ArrayBackend,
    ):
        assert array_backend in (NUMPY, CUDA)
        with self.open_files(fileset) as open_files:
            yield from self._get_tiles_by_block(
                tiling_scheme=tiling_scheme,
                open_files=open_files,
                read_ranges=read_ranges,
                read_dtype=read_dtype,
                native_dtype=native_dtype,
                decoder=decoder,
                corrections=corrections,
                sync_offset=sync_offset,
            )
