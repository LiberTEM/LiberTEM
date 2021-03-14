from typing import Tuple
import logging

import zarr
import numpy as np

from libertem.common import Slice, Shape
from libertem.common.buffers import zeros_aligned
from libertem.corrections import CorrectionSet
from .base import (
    DataSet, Partition, DataTile, DataSetException, DataSetMeta, _roi_to_nd_indices,
)

from .hdf5 import (
    _partition_shape_for_chunking, _tileshape_for_chunking,
    _have_contig_chunks,
)

logger = logging.getLogger(__name__)


class ZarrDataSet(DataSet):
    def __init__(self, path, sig_dims=2, io_backend=None):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError(
                "ZarrDataSet currently doesn't support alternative I/O backends"
            )
        self._path = path
        self._sig_dims = sig_dims
        self._chunks = None
        self._dtype = None
        self._shape = None
        self._sync_offset = 0

    def _do_initialize(self):
        za = zarr.open(self._path, mode='r')
        self._shape = Shape(za.shape, sig_dims=self._sig_dims)
        self._chunks = za.chunks
        self._dtype = za.dtype
        self._image_count = self._shape.nav.size
        self._meta = DataSetMeta(
            shape=self._shape,
            raw_dtype=self._dtype,
            sync_offset=self._sync_offset,
            image_count=self._image_count,
        )
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        raise NotImplementedError()

    @classmethod
    def get_supported_extensions(cls):
        return set()

    @classmethod
    def _do_detect(cls, path):
        try:
            zarr.open(path, mode='r')
        except Exception as e:
            raise DataSetException(repr(e)) from e

    @classmethod
    def detect_params(cls, path, executor):
        try:
            executor.run_function(cls._do_detect, path)
        except (IOError, OSError, KeyError, ValueError, TypeError, DataSetException):
            # not a h5py file or can't open for some reason:
            return False

        return {
            "parameters": {
                "path": path,
            },
            "info": {}
        }

    @property
    def dtype(self):
        if self._dtype is None:
            raise RuntimeError("please call initialize")
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            raise RuntimeError("please call initialize")
        return self._shape

    def check_valid(self):
        return True

    def get_cache_key(self):
        return {
            "path": self._path,
        }

    def get_diagnostics(self):
        return []

    def get_partitions(self):
        ds_shape = Shape(self.shape, sig_dims=self._sig_dims)
        ds_slice = Slice(origin=[0] * len(self.shape), shape=ds_shape)
        target_size = 1024*1024*1024
        partition_shape = self.partition_shape(
            target_size=target_size,
            dtype=self.dtype,
        ) + tuple(self.shape.sig)

        # if the data is chunked in the navigation axes, choose a compatible
        # partition size (even important for non-compressed data!)
        chunks = self._chunks
        if chunks is not None and not _have_contig_chunks(chunks, ds_shape):
            partition_shape = _partition_shape_for_chunking(chunks, ds_shape)

        for pslice in ds_slice.subslices(partition_shape):
            yield ZarrPartition(
                meta=self._meta,
                path=self._path,
                partition_slice=pslice.flatten_nav(self.shape),
                slice_nd=pslice,
                io_backend=self.get_io_backend(),
                chunks=self._chunks,
            )

    def __repr__(self):
        return "<ZarrDataSet of %s shape=%s>" % (self._dtype, self._shape)


class ZarrPartition(Partition):
    def __init__(self, path, slice_nd, chunks, *args, **kwargs):
        self._slice_nd = slice_nd
        self._corrections = None
        self._chunks = chunks
        self._path = path
        super().__init__(*args, **kwargs)

    def _have_compatible_chunking(self):
        chunks = self._chunks
        if chunks is None:
            return True
        # all-1 in nav dims works:
        nav_dims = self._slice_nd.shape.nav.dims
        chunks_nav = self._chunks[:nav_dims]
        if all(c == 1 for c in chunks_nav):
            return True
        # everything else is problematic and needs special case:
        return False

    def _get_subslices_chunked_full_frame(self, scheme_lookup, nav_dims, tileshape_nd):
        """
        chunked full-frame reading. outer loop goes over the
        navigation coords of each chunk, inner loop is pushed into
        hdf5 by reading full frames need to order slices in a way that
        efficiently uses the chunk cache.
        """
        chunks_nav = self._chunks[:nav_dims]
        chunk_full_frame = chunks_nav + self._slice_nd.shape.sig
        chunk_slices = self._slice_nd.subslices(shape=chunk_full_frame)
        logger.debug(
            "_get_subslices_chunked_full_frame: chunking first by %r, then %r",
            chunk_full_frame, tileshape_nd,
        )
        for chunk_slice in chunk_slices:
            subslices = chunk_slice.subslices(shape=tileshape_nd)
            for subslice in subslices:
                idx = scheme_lookup[(subslice.origin[nav_dims:], subslice.shape[nav_dims:])]
                yield idx, subslice

    def _get_subslices_chunked_tiled(self, tiling_scheme, scheme_lookup, nav_dims, tileshape_nd):
        """
        general tiled reading w/ chunking outer loop is a chunk in
        signal dimensions, inner loop is over "rows in nav"
        """
        slice_nd_sig = self._slice_nd.sig
        slice_nd_nav = self._slice_nd.nav
        chunks_nav = self._chunks[:nav_dims]
        sig_slices = slice_nd_sig.subslices(tiling_scheme.shape.sig)
        logger.debug(
            "_get_subslices_chunked_tiled: chunking first by sig %r, then nav %r, finally %r",
            tiling_scheme.shape.sig, chunks_nav, tileshape_nd
        )
        for sig_slice in sig_slices:
            chunk_slices = slice_nd_nav.subslices(shape=chunks_nav)
            for chunk_slice_nav in chunk_slices:
                chunk_slice = Slice(
                    origin=chunk_slice_nav.origin + sig_slice.origin,
                    shape=chunk_slice_nav.shape + tuple(sig_slice.shape),
                )
                subslices = chunk_slice.subslices(shape=tileshape_nd)
                for subslice in subslices:
                    scheme_key = (subslice.origin[nav_dims:], subslice.shape[nav_dims:])
                    idx = scheme_lookup[scheme_key]
                    yield idx, subslice

    def _get_subslices(self, tiling_scheme):
        """
        Generate partition subslices for the given tiling scheme for the different cases.
        """
        extra_nav_dims = self.meta.shape.nav.dims - tiling_scheme.shape.nav.dims
        tileshape_nd = extra_nav_dims * (1,) + tuple(tiling_scheme.shape)

        nav_dims = self._slice_nd.shape.nav.dims

        # Three cases need to be handled:
        if self._have_compatible_chunking():
            # 1) no chunking, or compatible chunking. we are free to use
            #    whatever access pattern we deem efficient:
            logger.debug("using simple tileshape_nd slicing")
            subslices = self._slice_nd.subslices(shape=tileshape_nd)
            scheme_len = len(tiling_scheme)
            for idx, subslice in enumerate(subslices):
                scheme_idx = idx % scheme_len
                yield scheme_idx, subslice
        else:
            scheme_lookup = {
                (s.discard_nav().origin, tuple(s.discard_nav().shape)): idx
                for idx, s in tiling_scheme.slices
            }
            if len(tiling_scheme) == 1:
                logger.debug("using full-frame subslicing")
                yield from self._get_subslices_chunked_full_frame(
                    scheme_lookup, nav_dims, tileshape_nd
                )
            else:
                logger.debug("using chunk-adaptive subslicing")
                yield from self._get_subslices_chunked_tiled(
                    tiling_scheme, scheme_lookup, nav_dims, tileshape_nd
                )

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_normal(self, tiling_scheme, dest_dtype):
        dataset = zarr.open(self._path, mode='r')
        # because the dtype conversion done by HDF5 itself can be quite slow,
        # we need to use a buffer for reading in hdf5 native dtype:
        data_flat = zeros_aligned(tiling_scheme.shape, dtype=dataset.dtype).reshape((-1,))

        subslices = self._get_subslices(
            tiling_scheme=tiling_scheme,
        )
        for scheme_idx, tile_slice in subslices:
            tile_slice_flat = tile_slice.flatten_nav(self.meta.shape)
            # cut buffer into the right size
            buf_size = tile_slice.shape.size
            buf = data_flat[:buf_size].reshape(tile_slice.shape)
            dataset.get_basic_selection(tile_slice.get(), out=buf)
            tile_data = buf.reshape(tile_slice_flat.shape)
            self._preprocess(tile_data, tile_slice_flat)
            yield DataTile(
                tile_data,
                tile_slice=tile_slice_flat,
                scheme_idx=scheme_idx,
            )

    def _get_tiles_with_roi(self, roi, dest_dtype, tiling_scheme):
        # we currently don't chop up the frames when reading with a roi, so
        # the tiling scheme also must not contain more than one slice:
        assert len(tiling_scheme) == 1, "incompatible tiling scheme!"

        flat_roi = roi.reshape((-1,))
        roi = roi.reshape(self.meta.shape.nav)

        result_shape = Shape((1,) + tuple(self.meta.shape.sig), sig_dims=self.meta.shape.sig.dims)
        sig_origin = tuple([0] * self.meta.shape.sig.dims)
        frames_read = 0
        start_at_frame = self.slice.origin[0]
        frame_offset = np.count_nonzero(flat_roi[:start_at_frame])

        indices = _roi_to_nd_indices(roi, self._slice_nd)

        tile_data = np.zeros(result_shape, dtype=dest_dtype)

        dataset = zarr.open(self._file)
        tile_data_raw = np.zeros(result_shape, dtype=dataset.dtype)
        for idx in indices:
            tile_slice = Slice(
                origin=(frames_read + frame_offset,) + sig_origin,
                shape=result_shape,
            )
            dataset.get_basic_selection(idx, out=tile_data_raw)
            tile_data[:] = tile_data_raw  # extra copy for dtype/endianess conversion
            self._preprocess(tile_data, tile_slice)
            yield DataTile(
                tile_data,
                tile_slice=tile_slice,
                # there is only a single slice in the tiling scheme, so our
                # scheme_idx is constant 0
                scheme_idx=0,
            )
            frames_read += 1

    def get_base_shape(self, roi):
        if roi is not None:
            return (1,) + self.shape.sig
        if self._chunks is not None:
            sig_chunks = self._chunks[-self.shape.sig.dims:]
            return (1,) + sig_chunks
        return (1, 1,) + (self.shape[-1],)

    def get_min_sig_size(self):
        if self._chunks is not None:
            return 1024  # allow for tiled processing w/ small-ish chunks
        # un-chunked HDF5 seems to prefer larger signal slices, so we aim for 32 4k blocks:
        return 32 * 4096 // np.dtype(self.meta.raw_dtype).itemsize

    def adjust_tileshape(self, tileshape, roi):
        chunks = self._chunks
        if roi is not None:
            new_tileshape = (1,) + self.shape.sig
            print(new_tileshape)
            return new_tileshape
        if not _have_contig_chunks(chunks, self.shape):
            sig_chunks = chunks[-self.shape.sig.dims:]
            sig_ts = tileshape[-self.shape.sig.dims:]
            # if larger signal chunking is requested in the negotiation,
            # switch to full frames:
            if any(t > c for t, c in zip(sig_ts, sig_chunks)):
                # try to keep total tileshape size:
                tileshape_size = np.prod(tileshape, dtype=np.int64)
                depth = max(1, tileshape_size // self.shape.sig.size)
                new_tileshape = (depth,) + self.shape.sig
            else:
                # depth needs to be limited to prod(chunks.nav)
                new_tileshape = _tileshape_for_chunking(chunks, self.meta.shape)
        else:
            new_tileshape = chunks[-self.shape.sig.dims-1:]
        print(new_tileshape)
        return new_tileshape

    def need_decode(self, roi, read_dtype, corrections):
        return True

    def set_corrections(self, corrections: CorrectionSet):
        self._corrections = corrections

    def get_max_io_size(self):
        return 2 * np.prod(self._chunks) * np.dtype(self.meta.raw_dtype).itemsize

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        import numcodecs
        numcodecs.blosc.use_threads = False
        if roi is not None:
            yield from self._get_tiles_with_roi(roi, dest_dtype, tiling_scheme)
        else:
            yield from self._get_tiles_normal(tiling_scheme, dest_dtype)

    def get_locations(self):
        return None

    def get_macrotile(self, dest_dtype="float32", roi=None):
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct
        dask arrays from datasets.

        Note
        ----

        This can be inefficient if the dataset is compressed and chunked in the
        navigation axis, because you can either have forced-large macrotiles,
        or you can have read amplification effects, where a much larger amount
        of data is read from the HDF5 file than necessary.

        For example, if your chunking is :code:`(32, 32, 32, 32)`, in a
        dataset that is :code:`(128, 128, 256, 256)`, the partition must
        cover the whole of :code:`(32, 128, 256, 256)` - this is because
        partitions are contiguous in the navigation axis.

        The other possibility is to keep the partition smaller, for example
        only :code:`(3, 128, 256, 256)`. That would mean when reading a chunk
        from HDF5, we can only use 3*32 frames of the total 32*32 frames,
        a whopping ~10x read amplification.
        '''
        import numcodecs
        numcodecs.blosc.use_threads = False

        data = zeros_aligned(self.slice.adjust_for_roi(roi).shape, dtype=dest_dtype)

        za = zarr.open(self._path, mode='r')
        if roi is not None:
            roi = roi.reshape(self.meta.shape.nav)
            roi = roi[self._slice_nd.get(nav_only=True)]
            za.get_mask_selection(roi, out=data)
        else:
            za.get_basic_selection(self._slice_nd, out=data)

        tile_slice = Slice(
            origin=(self.slice.origin[0], 0, 0),
            shape=Shape(data.shape, sig_dims=2),
        )
        self._preprocess(data, tile_slice.flatten_nav())

        return DataTile(
            data,
            tile_slice=tile_slice,
            scheme_idx=0,
        )
