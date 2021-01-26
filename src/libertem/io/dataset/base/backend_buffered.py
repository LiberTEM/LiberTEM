import numpy as np
import numba
from numba.typed import Dict

from libertem.io.dataset.base.backend import IOBackend, IOBackendImpl
from libertem.common import Shape, Slice
from libertem.common.buffers import BufferPool
from libertem.common.numba import cached_njit
from .tiling import DataTile
from .decode import DtypeConversionDecoder

_r_n_d_cache = {}


def _make_buffered_reader_and_decoder(decode):
    """
    decode: from buffers, in bytes, possibly interpreted as native_dtype, to out_decoded.dtype
    """
    @cached_njit(boundscheck=False)
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


@numba.njit(cache=True)
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
    def __init__(self, max_buffer_size):
        super().__init__()
        self._max_buffer_size = max_buffer_size
        self._buffer_pool = BufferPool()

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
        self, tiling_scheme, fileset, read_ranges, read_dtype, native_dtype, decoder=None,
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

        largest_slice = sorted([
            (np.prod(s_.shape), s_)
            for _, s_ in tiling_scheme.slices
        ], key=lambda x: x[0], reverse=True)[0][1]

        buf_shape = (tiling_scheme.depth,) + tuple(largest_slice.shape)

        need_clear = decoder.do_clear()

        slices = read_ranges[0]
        shape_prods = np.prod(slices[..., 1, :], axis=1)
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
                    block_idx, tile_block_size, min_per_file, max_per_file, fileset,
                    slices, ranges, scheme_indices, shape_prods, out_decoded, r_n_d,
                    sig_dims, ds_shape, need_clear, native_dtype, corrections,
                )

    def _read_block_dense(
        self, block_idx, tile_block_size, min_per_file, max_per_file, fileset,
        slices, ranges, scheme_indices, shape_prods, out_decoded, r_n_d,
        sig_dims, ds_shape, need_clear, native_dtype, corrections,
    ):
        """
        Reads a block of tiles, starting at `block_idx`, having a size of
        `tile_block_size` read range entries.
        """
        # phase 1: read
        buffers = Dict()
        for fileno in min_per_file.keys():
            fh = fileset[fileno]
            read_size = max_per_file[fileno] - min_per_file[fileno]
            # FIXME: re-use buffers
            buffers[fileno] = np.zeros(read_size, dtype=np.uint8)
            # FIXME: file header offset handling is a bit weird
            # FIXME: maybe file header offset should be folded into the read ranges instead?
            fh.seek(min_per_file[fileno] + fh._file_header)
            fh.readinto(buffers[fileno])

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
        sync_offset, corrections,
    ):
        with fileset:
            yield from self._get_tiles_by_block(
                tiling_scheme=tiling_scheme,
                fileset=fileset,
                read_ranges=read_ranges,
                read_dtype=read_dtype,
                native_dtype=native_dtype,
                decoder=decoder,
                corrections=corrections,
                sync_offset=sync_offset,
            )
