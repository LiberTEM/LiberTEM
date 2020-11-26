import os
import mmap
import logging

import numba
import numpy as np
from numba.typed import List

from libertem.common import Shape, Slice
from libertem.common.buffers import BufferPool
from libertem.corrections import CorrectionSet
from .tiling import DataTile
from .decode import DtypeConversionDecoder
from .file import File

log = logging.getLogger(__name__)


class IOBackend:
    pass  # FIXME: add interface methods here


_r_n_d_cache = {}


def _make_mmap_reader_and_decoder(decode):
    """
    decode: from inp, in bytes, possibly interpreted as native_dtype, to out.dtype
    """
    @numba.njit(boundscheck=False)
    def _tilereader_w_copy(outer_idx, mmaps, sig_dims, tile_read_ranges,
                           out_decoded, native_dtype, do_zero,
                           origin, shape, ds_shape):
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
    return _tilereader_w_copy


class LocalFSMMapBackend(IOBackend):
    def __init__(self, decoder=None, corrections: CorrectionSet = None):
        self._decoder = decoder
        self._corrections = corrections
        self._buffer_pool = BufferPool()

    def need_copy(self, roi, native_dtype, read_dtype, tiling_scheme=None, fileset=None):
        # checking conditions in which "straight mmap" is not possible
        # straight mmap means our dataset can just return views into the underlying mmap object
        # as tiles and use them as they are in the UDFs

        # 1) if a roi is given, straight mmap doesn't work because there are gaps in the navigation
        # axis:
        if roi is not None:
            log.debug("have roi, need copy")
            return True

        # 2) if we need to decode data, or do dtype conversion, we can't return
        # views into the underlying file:
        if self._need_decode(native_dtype, read_dtype):
            log.debug("have decode, need copy")
            return True

        # 3) if we have less frames per file than tile depth, we need to copy, too
        if tiling_scheme and fileset:
            fileset_arr = fileset.get_as_arr()
            if np.min(fileset_arr[:, 1] - fileset_arr[:, 0]) < tiling_scheme.depth:
                log.debug("too large for fileset, need copy")
                return True

        # 4) if we apply corrections, we need to copy
        if self._corrections is not None and self._corrections.have_corrections():
            log.debug("have corrections, need copy")
            return True

        return False

    def _need_decode(self, native_dtype, read_dtype):
        # FIXME: even with dtype "mismatch", we can possibly do dtype
        # conversion, if the tile size is small enough! maybe benchmark this
        # vs. _get_tiles_w_copy?
        if native_dtype != read_dtype:
            return True
        if self._decoder is not None:
            return True
        return False

    def _get_tiles_straight(self, tiling_scheme, fileset, read_ranges):
        """
        Parameters
        ----------

        fileset : FileSet
            To ensure best performance, should be limited to the files
            that are part of the current partition (otherwise we will
            spend more time finding the right file for a given frame
            index)

        read_ranges : Tuple[np.ndarray, np.ndarray]
            As returned by `get_read_ranges`
        """

        ds_sig_shape = tiling_scheme.dataset_shape.sig
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges
        for idx in range(slices.shape[0]):
            origin, shape = slices[idx]
            tile_ranges = ranges[idx]
            scheme_idx = scheme_indices[idx]

            # FIXME: for straight mmap, read_ranges must not contain tiles
            # that span multiple files!
            file_idx = tile_ranges[0][0]
            fh = fileset[file_idx]
            memmap = fh.mmap().reshape((fh.num_frames,) + tuple(ds_sig_shape))
            tile_slice = Slice(
                origin=origin,
                shape=Shape(shape, sig_dims=sig_dims)
            )
            data_slice = (
                slice(origin[0] - fh.start_idx, origin[0] - fh.start_idx + shape[0]),
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
        key = (decode,)
        if key in _r_n_d_cache:
            return _r_n_d_cache[key]
        r_n_d = _make_mmap_reader_and_decoder(decode=decode)
        _r_n_d_cache[key] = r_n_d
        return r_n_d

    def preprocess(self, data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(data, tile_slice)

    def _get_tiles_w_copy(self, tiling_scheme, fileset, read_ranges, read_dtype, native_dtype):
        if self._decoder is not None:
            decoder = self._decoder
        else:
            decoder = DtypeConversionDecoder(
            )
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
            ranges = read_ranges[1]
            scheme_indices = read_ranges[2]
            for idx in range(slices.shape[0]):
                origin, shape = slices[idx]
                tile_ranges = ranges[idx]
                scheme_idx = scheme_indices[idx]
                out_cut = out_decoded[:np.prod(shape)].reshape((shape[0], -1))
                data = r_n_d(
                    idx,
                    mmaps, sig_dims, tile_ranges,
                    out_cut, native_dtype, do_zero=need_clear,
                    origin=origin, shape=shape, ds_shape=ds_shape,
                )
                tile_slice = Slice(
                    origin=origin,
                    shape=Shape(shape, sig_dims=sig_dims)
                )
                data = data.reshape(shape)
                self.preprocess(data, tile_slice)
                yield DataTile(
                    data,
                    tile_slice=tile_slice,
                    scheme_idx=scheme_idx,
                )

    def get_tiles(self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype):
        # TODO: how would compression work?
        # TODO: sparse input data? COO format? fill rate? → own pipeline! → later!
        # strategy: assume low (20%?) fill rate, read whole partition and apply ROI in-memory
        #           partitioning when opening the dataset, or by having one file per partition

        with fileset:
            self._set_readahead_hints(roi, fileset)
            if not self.need_copy(
                tiling_scheme=tiling_scheme,
                fileset=fileset,
                roi=roi,
                native_dtype=native_dtype, read_dtype=read_dtype,
            ):
                yield from self._get_tiles_straight(
                    tiling_scheme, fileset, read_ranges
                )
            else:
                yield from self._get_tiles_w_copy(
                    tiling_scheme=tiling_scheme,
                    fileset=fileset,
                    read_ranges=read_ranges,
                    read_dtype=read_dtype,
                    native_dtype=native_dtype,
                )

    def _set_readahead_hints(self, roi, fileset):
        if not hasattr(os, 'posix_fadvise'):
            return
        if any([f.fileno() is None
                for f in fileset]):
            return
        if roi is None:
            for f in fileset:
                os.posix_fadvise(
                    f.fileno(),
                    0,
                    0,
                    os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED
                )
        else:
            for f in fileset:
                os.posix_fadvise(
                    f.fileno(),
                    0,
                    0,
                    os.POSIX_FADV_RANDOM | os.POSIX_FADV_WILLNEED
                )


class LocalFile(File):
    def open(self):
        f = open(self._path, "rb")
        self._file = f
        self._raw_mmap = mmap.mmap(
            fileno=f.fileno(),
            length=0,
            # can't use offset for cutting off file header, as it needs to be
            # aligned to page size...
            offset=0,
            access=mmap.ACCESS_READ,
        )
        # TODO: self._raw_mmap.madvise(mmap.MADV_HUGEPAGE) - benchmark this!
        itemsize = np.dtype(self._native_dtype).itemsize
        assert self._frame_header % itemsize == 0
        assert self._frame_footer % itemsize == 0
        start = self._frame_header // itemsize
        stop = start + int(np.prod(self._sig_shape))
        if self._file_header != 0:
            # FIXME: keep the mmap object around maybe?
            self._raw_mmap = memoryview(self._raw_mmap)[self._file_header:]
        self._mmap = self._mmap_to_array(self._raw_mmap, start, stop)

    def _mmap_to_array(self, raw_mmap, start, stop):
        return np.frombuffer(raw_mmap, dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, start:stop]

    def close(self):
        self._mmap = None
        self._raw_mmap = None
        self._file.close()
        self._file = None

    def mmap(self):
        return self._mmap

    def raw_mmap(self):
        return self._raw_mmap

    def fileno(self):
        return self._file.fileno()
