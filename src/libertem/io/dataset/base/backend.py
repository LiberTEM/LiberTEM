import os
import mmap
import logging

import numpy as np
from numba.typed import List

from libertem.common import Shape, Slice
from libertem.common.buffers import BufferPool
from libertem.common.numba import cached_njit
from .tiling import DataTile
from .decode import DtypeConversionDecoder
from .file import File

log = logging.getLogger(__name__)


_r_n_d_cache = {}


def _make_mmap_reader_and_decoder(decode):
    """
    decode: from inp, in bytes, possibly interpreted as native_dtype, to out.dtype
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


def _make_reader_and_decoder(decode):
    """
    decode: from inp, in bytes, possibly interpreted as native_dtype, to out.dtype
    """
    @cached_njit(boundscheck=False)
    def _tilereader_w_copy(outer_idx, buffers, sig_dims, tile_read_ranges,
                           out_decoded, native_dtype, do_zero,
                           origin, shape, ds_shape, offsets):
        if do_zero:
            out_decoded[:] = 0
        for rr_idx in range(tile_read_ranges.shape[0]):
            rr = tile_read_ranges[rr_idx]
            if rr[1] == rr[2] == 0:
                break
            buffer = buffers[rr[0]]
            offset = offsets[rr[0]]  # per-file offset of start of read buffer
            decode(
                inp=buffer[rr[1] - offset:rr[2] - offset],
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


@cached_njit
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


class IOBackend:
    registry = {}

    def __init_subclass__(cls, id_=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if id_ is not None:
            cls.registry[id_] = cls

    def __init__(self):
        pass

    @classmethod
    def from_json(cls, msg):
        """
        Construct an instance from the already-decoded `msg`.
        """
        raise NotImplementedError()


class IOBackendImpl:
    def __init__(self):
        pass

    def need_copy(
        self, decoder, roi, native_dtype, read_dtype, tiling_scheme=None, fileset=None,
        sync_offset=0, corrections=None,
    ):
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
        if self._need_decode(decoder, native_dtype, read_dtype):
            log.debug("have decode, need copy")
            return True

        # 3) if we have less number of frames per file than tile depth, we need to copy
        if tiling_scheme and fileset:
            fileset_arr = fileset.get_as_arr()
            if np.min(fileset_arr[:, 1] - fileset_arr[:, 0]) < tiling_scheme.depth:
                log.debug("too large for fileset, need copy")
                return True

        # 4) if we apply corrections, we need to copy
        if corrections is not None and corrections.have_corrections():
            log.debug("have corrections, need copy")
            return True

        # 5) if a negative offset is given, we need to copy
        if sync_offset < 0:
            log.debug("negative offset is set, need copy")
            return True

        return False

    def _need_decode(self, decoder, native_dtype, read_dtype):
        # FIXME: even with dtype "mismatch", we can possibly do dtype
        # conversion, if the tile size is small enough! maybe benchmark this
        # vs. _get_tiles_w_copy?
        if native_dtype != read_dtype:
            return True
        if decoder is not None:
            return True
        return False

    def preprocess(self, data, tile_slice, corrections):
        if corrections is None:
            return
        corrections.apply(data, tile_slice)

    def get_tiles(
        self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype, decoder,
        corrections,
    ):
        """
        Read tiles from `fileset`, as specified by the parameters.

        Usually, this is used to read the data for a single partition.

        Parameters
        ----------

        tiling_scheme : TilingScheme
            Specifies how the tiles should be shaped

        fileset : FileSet
            The files that should be read from. Note that the order in the `FileSet` is important,
            it must match the indices on the `read_ranges`.

        read_ranges : np.ndarray
            Read ranges, as generated by :meth:`FileSet.get_read_ranges`

        roi : np.ndarray
            Boolean array specifying which data should be read

        native_dtype : np.dtype
            The native on-disk data type. If there is no direct match to
            a numpy dtype, specify the closest dtype.

        read_dtype : np.dtype
            The data dtype into which the data is converted when reading

        corrections
            A set of corrections to apply in a preprocesing step
        """
        raise NotImplementedError()


class MMapBackend(IOBackend, id_="mmap"):
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
        """
        I/O backend using memory mapped files.

        Parameters
        ----------
        """
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
        self, tiling_scheme, fileset, read_ranges, read_dtype, native_dtype, roi, decoder=None,
        sync_offset=0, corrections=None,
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
                    roi=roi,
                    sync_offset=sync_offset,
                    decoder=decoder,
                    corrections=corrections,
                )

    # @profile
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


class BufferedBackend(IOBackend, id_="buffered"):
    def __init__(self):
        pass  # TODO: paramters: max. buffer size to limit memory usage?

    @classmethod
    def from_json(cls, msg):
        """
        Construct an instance from the already-decoded `msg`.
        """
        raise NotImplementedError("TODO! implement me!")

    def get_impl(self):
        return BufferedBackendImpl()


class BufferedBackendImpl(IOBackendImpl):
    def __init__(self):
        """
        I/O backend using a buffered reading strategy.

        Parameters
        ----------
        """
        super().__init__()
        self._buffer_pool = BufferPool()

    def _get_tiles_straight(self, tiling_scheme, fileset, read_ranges, read_dtype):
        """
        """

        ds_sig_shape = tiling_scheme.dataset_shape.sig
        ds_sig_size = np.prod(ds_sig_shape)
        sig_dims = tiling_scheme.shape.sig.dims
        slices, ranges, scheme_indices = read_ranges
        buf_shape = (tiling_scheme.depth,) + ds_sig_size
        pos = -1
        file_idx = -1

        with self._buffer_pool.empty(buf_shape, dtype=read_dtype) as out_decoded:
            for idx in range(slices.shape[0]):
                origin = slices[idx, 0]
                shape = slices[idx, 1]
                tile_ranges = ranges[idx]
                scheme_idx = scheme_indices[idx]

                if tile_ranges[0, 1] > pos or file_idx != tile_ranges[0, 0]:
                    # NOTE: for straight reading, read_ranges must not contain tiles
                    # that span multiple files. This is ensured in IOBackend.need_copy
                    # (that is, we force copying if files are smaller than tiles)
                    file_idx = tile_ranges[0, 0]  # file index from the first rr for this tile
                    fh = fileset[file_idx]
                    pos = tile_ranges[0, 1]
                    assert tile_ranges[0, 1] == np.min(tile_ranges[:, 1])
                    fh.seek(pos)  # minimum of starting offset for this tile
                    fh.readinto(out_decoded)
                else:
                    assert tile_ranges[0, 2] <= pos + out_decoded.nbytes

                # memmap = fh.mmap().reshape((fh.num_frames,) + tuple(ds_sig_shape))
                tile_slice = Slice(
                    origin=origin,
                    shape=Shape(shape, sig_dims=sig_dims)
                )
                data_slice = tile_slice.get()
                data = memmap[data_slice]
                yield DataTile(
                    data,
                    tile_slice=tile_slice,
                    scheme_idx=scheme_idx,
                )

    def get_read_and_decode(self, decode):
        key = (decode, "read")
        if key in _r_n_d_cache:
            return _r_n_d_cache[key]
        r_n_d = _make_reader_and_decoder(decode=decode)
        _r_n_d_cache[key] = r_n_d
        return r_n_d

    # @profile
    def _get_tiles_w_copy(
        self, tiling_scheme, fileset, read_ranges, read_dtype, native_dtype, decoder=None,
        corrections=None,
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

        with self._buffer_pool.empty(buf_shape, dtype=read_dtype) as out_decoded:
            out_decoded = out_decoded.reshape((-1,))

            slices = read_ranges[0]
            shape_prods = np.prod(slices[..., 1, :], axis=1)
            ranges = read_ranges[1]
            scheme_indices = read_ranges[2]
            for idx in range(slices.shape[0]):
                origin = slices[idx, 0]
                shape = slices[idx, 1]
                tile_ranges = ranges[idx]
                scheme_idx = scheme_indices[idx]
                # if idx < slices.shape[0] - 1:
                #     self._prefetch_for_tile(fileset, ranges[idx + 1])
                #     pass
                out_cut = out_decoded[:shape_prods[idx]].reshape((shape[0], -1))
                data = r_n_d(
                    idx,
                    buffers, sig_dims, tile_ranges,
                    out_cut, native_dtype, do_zero=need_clear,
                    origin=origin, shape=shape, ds_shape=ds_shape, offset=offset,
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

    # @profile
    def get_tiles(
        self, tiling_scheme, fileset, read_ranges, roi, native_dtype, read_dtype, decoder,
        corrections,
    ):
        with fileset:
            yield from self._get_tiles_w_copy(
                tiling_scheme=tiling_scheme,
                fileset=fileset,
                read_ranges=read_ranges,
                read_dtype=read_dtype,
                native_dtype=native_dtype,
                decoder=decoder,
                corrections=corrections,
            )


class LocalFile(File):
    def open(self):
        # NOTE: for `readinto` to work, we must not switch off buffering here!
        # otherwise, `readinto` may return partial results, which can be hard to handle
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
        # self._raw_mmap.madvise(mmap.MADV_HUGEPAGE) # TODO - benchmark this!
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
        """
        Create an array from the raw memory map, stipping away
        frame headers and footers

        Parameters
        ----------

        raw_mmap : np.memmap or memoryview
            The raw memory map, with the file header already stripped away

        start : int
            Number of items cut away at the start of each frame (frame_header // itemsize)

        stop : int
            Number of items per frame (something like start + np.prod(sig_shape))
        """
        return np.frombuffer(raw_mmap, dtype=self._native_dtype).reshape(
            (self.num_frames, -1)
        )[:, start:stop]

    def close(self):
        self._mmap = None
        self._raw_mmap = None
        self._file.close()
        self._file = None

    def mmap(self):
        """
        Memory map for this file, with file header, frame header and frame footer cut off

        Used for reading tiles straight from the filesystem cache
        """
        return self._mmap

    def raw_mmap(self):
        """
        Memory map for this file, with only the file header cut off

        Used for reading tiles with a decoding step, using the read ranges
        """
        return self._raw_mmap

    def readinto(self, out):
        """
        Fill `out` by reading from the current file position
        """
        return self._file.readinto(out)

    def seek(self, pos):
        self._file.seek(pos)

    def tell(self):
        return self._file.tell()

    def fileno(self):
        return self._file.fileno()
