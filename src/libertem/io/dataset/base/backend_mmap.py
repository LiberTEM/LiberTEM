import os

import numpy as np
from numba.typed import List
import numba

from libertem.io.dataset.base.backend import IOBackend, IOBackendImpl
from libertem.common import Shape, Slice
from libertem.common.buffers import BufferPool
from libertem.common.numba import cached_njit
from .tiling import DataTile
from .decode import DtypeConversionDecoder


_r_n_d_cache = {}


def _make_mmap_reader_and_decoder(decode):
    """
    decode: from inp, in bytes, possibly interpreted as native_dtype, to out_decoded.dtype
    """
    @cached_njit(boundscheck=False, cache=True)
    def _mmap_tilereader_w_copy(outer_idx, mmaps, sig_dims, tile_read_ranges,
                           out_decoded, native_dtype, do_zero,
                           origin, shape, ds_shape):
        """
        Read and decode a single tile
        """
        if do_zero:
            out_decoded[:] = 0
        for rr_idx in range(tile_read_ranges.shape[0]):
            rr = tile_read_ranges[rr_idx]
            if rr[1] == rr[2] == 0:
                break
            memmap = mmaps[rr[0]]
            decode(
                inp=memmap[rr[1]:rr[2]],
                out=out_decoded,
                idx=rr_idx,
                native_dtype=native_dtype,
                rr=rr,
                origin=origin,
                shape=shape,
                ds_shape=ds_shape,
            )
        return out_decoded
    return _mmap_tilereader_w_copy


@numba.njit
def _get_prefetch_ranges(num_files, tile_ranges):
    res = np.zeros((num_files, 3), dtype=np.int64)
    for rr_idx in range(tile_ranges.shape[0]):
        rr = tile_ranges[rr_idx]
        if res[rr[0], 0] == 0:
            res[rr[0], 0] = rr[1]  # if minimum is 0, pick first value
        res[rr[0], 0] = min(res[rr[0], 0], rr[1])
        res[rr[0], 1] = max(res[rr[0], 1], rr[2])
        res[rr[0], 2] = rr[0]
    return res


class MMapBackend(IOBackend, id_="mmap"):
    """
    I/O backend using memory mapped files. Used by default on non-Windows
    systems.

    Parameters
    ----------
    enable_readahead_hints : bool
        Linux only. Try to influence readahead behavior (experimental).
    """
    def __init__(self, enable_readahead_hints=False):
        self._enable_readahead = enable_readahead_hints

    def get_impl(self):
        return MMapBackendImpl(self._enable_readahead)

    @classmethod
    def from_json(cls, msg):
        """
        Construct an instance from the already-decoded `msg`.
        """
        raise NotImplementedError("TODO! implement me!")


class MMapBackendImpl(IOBackendImpl):
    def __init__(self, enable_readahead_hints=False):
        super().__init__()
        self._enable_readahead = enable_readahead_hints
        self._buffer_pool = BufferPool()

    def _get_tiles_straight(self, tiling_scheme, fileset, read_ranges, sync_offset=0):
        """
        Read straight from the file system cache, via memory mapping, without
        any decoding step.

        This method makes a few assumptions:
         - no corrections are needed
         - tiles don't span multiple files
         - the `LocalFile` has already cut away headers and footers
           (both per-file and per-frame) from the mmap

        Parameters
        ----------

        fileset : FileSet
            The fileset must correspond to the indices used in the `read_ranges`.
            Usually, that means it is limited to the files that are part of the
            current partition.

        read_ranges : Tuple[np.ndarray, np.ndarray, np.ndarray]
            As returned by `get_read_ranges`
        """

        ds_sig_shape = tuple(tiling_scheme.dataset_shape.sig)
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges
        for idx in range(slices.shape[0]):
            origin, shape = slices[idx]
            tile_ranges = ranges[idx]
            scheme_idx = scheme_indices[idx]

            # NOTE: for straight mmap, read_ranges must not contain tiles
            # that span multiple files. This is ensured in IOBackend.need_copy
            # (that is, we force copying if files are smaller than tiles)
            file_idx = tile_ranges[0][0]
            fh = fileset[file_idx]
            memmap = fh.mmap().reshape((fh.num_frames,) + ds_sig_shape)
            tile_slice = Slice(
                origin=origin,
                shape=Shape(shape, sig_dims=sig_dims)
            )
            # sync_offset is either zero or positive
            # in case of negative sync_offset, _get_tiles_w_copy is used
            data_slice = (
                slice(
                    origin[0] - fh.start_idx + sync_offset,
                    origin[0] - fh.start_idx + shape[0] + sync_offset
                ),
            ) + tuple([
                slice(o, (o + s))
                for (o, s) in zip(origin[1:], shape[1:])
            ])
            data = memmap[data_slice]
            yield DataTile(
                data,
                tile_slice=tile_slice,
                scheme_idx=scheme_idx,
            )

    def get_read_and_decode(self, decode):
        key = (decode, "mmap")
        if key in _r_n_d_cache:
            return _r_n_d_cache[key]
        r_n_d = _make_mmap_reader_and_decoder(decode=decode)
        _r_n_d_cache[key] = r_n_d
        return r_n_d

    def _get_tiles_w_copy(
        self, tiling_scheme, fileset, read_ranges, read_dtype, native_dtype,
        decoder=None, corrections=None,
    ):
        if decoder is None:
            decoder = DtypeConversionDecoder()
        decode = decoder.get_decode(
            native_dtype=np.dtype(native_dtype),
            read_dtype=np.dtype(read_dtype),
        )
        r_n_d = self._r_n_d = self.get_read_and_decode(decode)

        native_dtype = decoder.get_native_dtype(native_dtype, read_dtype)
        mmaps = List()
        for fh in fileset:
            mmaps.append(np.frombuffer(fh.raw_mmap(), dtype=np.uint8))

        sig_dims = tiling_scheme.shape.sig.dims
        ds_shape = np.array(tiling_scheme.dataset_shape)

        largest_slice = sorted([
            (np.prod(s_.shape), s_)
            for _, s_ in tiling_scheme.slices
        ], key=lambda x: x[0], reverse=True)[0][1]

        buf_shape = (tiling_scheme.depth,) + tuple(largest_slice.shape)
        need_clear = decoder.do_clear()

        with self._buffer_pool.empty(buf_shape, dtype=read_dtype) as out_decoded:
            out_decoded = out_decoded.reshape((-1,))
            slices = read_ranges[0]
            shape_prods = np.prod(slices[..., 1, :], axis=1)
            ranges = read_ranges[1]
            scheme_indices = read_ranges[2]
            for idx in range(slices.shape[0]):
                origin = slices[idx, 0]
                shape = slices[idx, 1]
                tile_slice = Slice(
                    origin=origin,
                    shape=Shape(shape, sig_dims=sig_dims)
                )
                tile_ranges = ranges[idx]
                scheme_idx = scheme_indices[idx]
                # if idx < slices.shape[0] - 1:
                #     self._prefetch_for_tile(fileset, ranges[idx + 1])
                #     pass
                out_cut = out_decoded[:shape_prods[idx]].reshape((shape[0], -1))
                data = r_n_d(
                    idx,
                    mmaps, sig_dims, tile_ranges,
                    out_cut, native_dtype, do_zero=need_clear,
                    origin=origin, shape=shape, ds_shape=ds_shape,
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
        # TODO: how would compression work?
        # TODO: sparse input data? COO format? fill rate? → own pipeline! → later!
        # strategy: assume low (20%?) fill rate, read whole partition and apply ROI in-memory
        #           partitioning when opening the dataset, or by having one file per partition

        with fileset:
            if self._enable_readahead:
                self._set_readahead_hints(roi, fileset)
            if not self.need_copy(
                decoder=decoder,
                tiling_scheme=tiling_scheme,
                fileset=fileset,
                roi=roi,
                native_dtype=native_dtype,
                read_dtype=read_dtype,
                sync_offset=sync_offset,
                corrections=corrections,
            ):
                yield from self._get_tiles_straight(
                    tiling_scheme, fileset, read_ranges, sync_offset,
                )
            else:
                yield from self._get_tiles_w_copy(
                    tiling_scheme=tiling_scheme,
                    fileset=fileset,
                    read_ranges=read_ranges,
                    read_dtype=read_dtype,
                    native_dtype=native_dtype,
                    decoder=decoder,
                    corrections=corrections,
                )

    def _prefetch_for_tile(self, fileset, tile_ranges):
        prefr = _get_prefetch_ranges(len(fileset), tile_ranges)
        prefr = prefr[~np.all(prefr == 0, axis=1)]
        for mi, ma, fidx in prefr:
            f = fileset[fidx]
            os.posix_fadvise(
                f.fileno(),
                mi,
                ma - mi,
                os.POSIX_FADV_WILLNEED
            )

    def _set_readahead_hints(self, roi, fileset):
        if not hasattr(os, 'posix_fadvise'):
            return
        if any([f.fileno() is None
                for f in fileset]):
            return
        for f in fileset:
            os.posix_fadvise(
                f.fileno(),
                0,
                0,
                os.POSIX_FADV_WILLNEED
            )
