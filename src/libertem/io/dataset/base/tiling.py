import math
from typing import List, Tuple
import logging
import warnings

import numba
from numba.typed import List as NumbaList
import numpy as np

from libertem.common import Slice, Shape
from libertem.common.numba import numba_ravel_multi_index_single as _ravel_multi_index, cached_njit
from libertem.corrections import CorrectionSet
from .roi import _roi_to_indices

log = logging.getLogger(__name__)


class TilingScheme:
    def __init__(self, slices: List[Slice],
                 tileshape: Shape, dataset_shape: Shape, debug=None):
        self._slices = slices
        self._tileshape = tileshape
        self._dataset_shape = dataset_shape
        self._debug = debug

        if tileshape.nav.dims > 1:
            raise ValueError("tileshape should have flat navigation dimensions")

    @classmethod
    def make_for_shape(cls, tileshape: Shape, dataset_shape: Shape, debug=None):
        """
        Make a TilingScheme from `tileshape` and `dataset_shape`.

        Note that both in signal and navigation direction there are border
        effects, i.e. if the depth doesn't evenly divide the number of
        frames in the partition (simplified, ROI also applies...), or if
        the signal dimensions of `tileshape` doesn't evenly divide the signal
        dimensions of the `dataset_shape`.

        Parameters
        ----------
        tileshape
            Uniform shape of all tiles. Should have flat navigation axis
            (meaning tileshape.nav.dims == 1) and be contiguous in signal
            dimensions.

        dataset_shape
            Shape of the whole data set. Only the signal part is used.
        """
        # FIXME: validate navigation part of the tileshape to be contiguous
        # (i.e. a shape like (1, 1, ..., 1, X1, ..., XN))
        # where X1 is <= the dataset shape at that index, and X2, ..., XN are
        # equal to the dataset shape at that index

        sig_slice = Slice(
            origin=tuple([0] * dataset_shape.sig.dims),
            shape=dataset_shape.sig
        )
        subslices = list(sig_slice.subslices(tileshape.sig))
        return cls(
            slices=subslices,
            tileshape=tileshape,
            dataset_shape=dataset_shape,
            debug=debug,
        )

    def __getitem__(self, idx):
        return self._slices[idx]

    def __len__(self):
        return len(self._slices)

    def __repr__(self):
        unique_shapes = list({tuple(slice_.shape) for slice_ in self._slices})
        return "<TilingScheme (depth=%d) shapes=%r len=%d>" % (
            self.depth, unique_shapes, len(self._slices),
        )

    @property
    def slices(self):
        """
        signal-only slices for all possible positions
        """
        return list(enumerate(self._slices))

    @property
    def slices_array(self):
        """
        Returns the slices from the schema as a numpy ndarray
        `a` of shape `(n, 2, sig_dims)` with:
        `a[i, 0]` are origins for slice `i`
        `a[i, 1]` are shapes for slice `i`
        """
        sig_dims = self._tileshape.sig.dims
        slices = np.zeros((len(self), 2, sig_dims), dtype=np.int64)
        for idx, slice_ in self.slices:
            slices[idx] = (tuple(slice_.origin), tuple(slice_.shape))
        return slices

    @property
    def shape(self):
        """
        tileshape. note that some border tiles can be smaller!
        """
        return self._tileshape

    @property
    def dataset_shape(self):
        return self._dataset_shape

    @property
    def depth(self):
        return self._tileshape.nav[0]


@numba.njit(inline='always')
def _default_px_to_bytes(
    bpp, frame_in_file_idx, slice_sig_size, sig_size, sig_origin,
    frame_footer_bytes, frame_header_bytes,
    file_idx, read_ranges,
):
    # we are reading a part of a single frame, so we first need to find
    # the offset caused by headers and footers:
    footer_offset = frame_footer_bytes * frame_in_file_idx
    header_offset = frame_header_bytes * (frame_in_file_idx + 1)
    byte_offset = footer_offset + header_offset

    # now let's figure in the current frame index:
    # (go down into the file by full frames; `sig_size`)
    offset = byte_offset + frame_in_file_idx * sig_size * bpp

    # offset in px in the current frame:
    sig_origin_bytes = sig_origin * bpp

    start = offset + sig_origin_bytes

    # size of the sig part of the slice:
    sig_size_bytes = slice_sig_size * bpp

    stop = start + sig_size_bytes

    read_ranges.append((file_idx, start, stop))


@numba.njit(boundscheck=True)
def _find_file_for_frame_idx(fileset_arr, frame_idx):
    """
    Find the file in `fileset_arr` that contains
    `frame_idx` and return its index using binary search.

    Worst case: something like 2**20 files, each containing
    a single frame.

    `fileset_arr` is an array of shape (number_files, 4)
    where `fileset_arr[i]` is:

        (start_idx, end_idx, file_header_size, file_idx)

    It must be sorted by `start_idx` and the defined intervals must not overlap.
    """
    while True:
        num_files = fileset_arr.shape[0]
        mid = num_files // 2
        mid_file = fileset_arr[mid]

        if mid_file[0] <= frame_idx and mid_file[1] > frame_idx:
            return mid_file[2]
        elif mid_file[0] > frame_idx:
            fileset_arr = fileset_arr[:mid]
        else:
            fileset_arr = fileset_arr[mid + 1:]


@numba.njit(inline='always')
def _default_read_ranges_tile_block(
    slices_arr, fileset_arr, slice_sig_sizes, sig_origins,
    inner_indices_start, inner_indices_stop, frame_indices, sig_size,
    px_to_bytes, bpp, frame_header_bytes, frame_footer_bytes, file_idxs,
    slice_offset, extra, sig_shape,
):
    result = NumbaList()

    # positions in the signal dimensions:
    for slice_idx in range(slices_arr.shape[0]):
        # (offset, size) arrays defining what data to read (in pixels)
        # NOTE: assumes contiguous tiling scheme
        # (i.e. a shape like (1, 1, ..., 1, X1, ..., XN))
        # where X1 is <= the dataset shape at that index, and X2, ..., XN are
        # equal to the dataset shape at that index
        slice_origin = slices_arr[slice_idx][0]
        slice_shape = slices_arr[slice_idx][1]
        slice_sig_size = slice_sig_sizes[slice_idx]
        sig_origin = sig_origins[slice_idx]

        read_ranges = NumbaList()

        # inner "depth" loop along the (flat) navigation axis of a tile:
        for i, inner_frame_idx in enumerate(range(inner_indices_start, inner_indices_stop)):
            inner_frame = frame_indices[inner_frame_idx]

            file_idx = file_idxs[i]
            f = fileset_arr[file_idx]

            frame_in_file_idx = inner_frame - f[0]

            px_to_bytes(
                bpp=bpp,
                frame_in_file_idx=frame_in_file_idx,
                slice_sig_size=slice_sig_size,
                sig_size=sig_size,
                sig_origin=sig_origin,
                frame_footer_bytes=frame_footer_bytes,
                frame_header_bytes=frame_header_bytes,
                file_idx=file_idx,
                read_ranges=read_ranges,
            )

        # the indices are compressed to the selected frames
        compressed_slice = np.array([
            [slice_offset + inner_indices_start] + [i for i in slice_origin],
            [inner_indices_stop - inner_indices_start] + [i for i in slice_shape],
        ])
        result.append((slice_idx, compressed_slice, read_ranges))

    return result


def make_get_read_ranges(
    px_to_bytes=_default_px_to_bytes,
    read_ranges_tile_block=_default_read_ranges_tile_block,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Translate the `TilingScheme` combined with the `roi` into (pixel)-read-ranges,
    together with their tile slices.

    Parameters
    ----------

    start_at_frame
        Dataset-global first frame index to read

    stop_before_frame
        Stop before this frame index

    tiling_scheme
        Description on how the data should be tiled

    fileset_arr
        Array of shape (number_of_files, 3) where the last dimension contains
        the following values: `(start_idx, end_idx, file_idx)`, where
        `[start_idx, end_idx)` defines which frame indices are contained
        in the file.

    roi
        Region of interest (for the full dataset)

    Returns
    -------

    (tile_slice, read_ranges)
        read_ranges is an ndarray with shape (number_of_tiles, depth, 3)
        where the last dimension contains: file index, start_byte, stop_byte
    """

    @cached_njit(boundscheck=True, cache=True)
    def _get_read_ranges_inner(
        start_at_frame, stop_before_frame, roi, depth,
        slices_arr, fileset_arr, sig_shape,
        bpp, sync_offset=0, extra=None, frame_header_bytes=0, frame_footer_bytes=0,
    ):
        result = NumbaList()

        sig_size = np.prod(np.array(sig_shape).astype(np.int64))

        if roi is None:
            frame_indices = np.arange(max(0, start_at_frame), stop_before_frame)
            # in case of a negative sync_offset, start_at_frame can be negative
            if start_at_frame < 0:
                slice_offset = abs(sync_offset)
            else:
                slice_offset = start_at_frame - sync_offset
        else:
            frame_indices = _roi_to_indices(
                roi, max(0, start_at_frame), stop_before_frame, sync_offset
            )
            # in case of a negative sync_offset, start_at_frame can be negative
            if start_at_frame < 0:
                slice_offset = np.count_nonzero(roi.reshape((-1,))[:abs(sync_offset)])
            else:
                slice_offset = np.count_nonzero(roi.reshape((-1,))[:start_at_frame - sync_offset])

        num_indices = frame_indices.shape[0]

        # indices into `frame_indices`:
        inner_indices_start = 0
        inner_indices_stop = min(depth, num_indices)

        # this should be `np.prod(..., axis=-1)``, which is not supported by numba yet:
        # slices that divide the signal dimensions:
        slice_sig_sizes = np.array([
            np.prod(slices_arr[slice_idx, 1, :].astype(np.int64))
            for slice_idx in range(slices_arr.shape[0])
        ])

        sig_origins = np.array([
            _ravel_multi_index(slices_arr[slice_idx][0], sig_shape)
            for slice_idx in range(slices_arr.shape[0])
        ])

        # outer "depth" loop skipping over `depth` frames at a time:
        while inner_indices_start < num_indices:
            file_idxs = np.array([
                _find_file_for_frame_idx(fileset_arr, frame_indices[inner_frame_idx])
                for inner_frame_idx in range(inner_indices_start, inner_indices_stop)
            ])

            for slice_idx, compressed_slice, read_ranges in read_ranges_tile_block(
                slices_arr, fileset_arr, slice_sig_sizes, sig_origins,
                inner_indices_start, inner_indices_stop, frame_indices, sig_size,
                px_to_bytes, bpp, frame_header_bytes, frame_footer_bytes, file_idxs,
                slice_offset, extra=extra, sig_shape=sig_shape,
            ):
                result.append((compressed_slice, read_ranges, slice_idx))

            inner_indices_start = inner_indices_start + depth
            inner_indices_stop = min(inner_indices_stop + depth, num_indices)

        result_slices = np.zeros((len(result), 2, 1 + len(sig_shape)), dtype=np.int64)
        for tile_idx, res in enumerate(result):
            result_slices[tile_idx] = res[0]

        if len(result) == 0:
            return (
                result_slices,
                np.zeros((len(result), depth, 3), dtype=np.int64),
                np.zeros((len(result)), dtype=np.int64),
            )

        max_rr_per_tile = max([len(res[1])
                               for res in result])

        slice_indices = np.zeros(len(result), dtype=np.int64)

        # read_ranges_tile_block can decide how many entries there are per read range,
        # so we need to generate a result array with the correct size:
        rr_num_entries = max(3, len(result[0][1][0]))
        result_ranges = np.zeros((len(result), max_rr_per_tile, rr_num_entries), dtype=np.int64)
        for tile_idx, res in enumerate(result):
            for depth_idx, read_range in enumerate(res[1]):
                result_ranges[tile_idx][depth_idx] = read_range
            slice_indices[tile_idx] = res[2]

        return result_slices, result_ranges, slice_indices
    return _get_read_ranges_inner


default_get_read_ranges = make_get_read_ranges()


class DataTile(np.ndarray):
    def __new__(cls, input_array, tile_slice, scheme_idx):
        obj = np.asarray(input_array).view(cls)
        obj.tile_slice = tile_slice
        obj.scheme_idx = scheme_idx

        if tile_slice.shape.nav.dims != 1:
            raise ValueError("DataTile should be flat in navigation axis")

        if obj.shape != tuple(tile_slice.shape):
            raise ValueError(
                "shape mismatch: data=%s, tile_slice=%s" % (obj.shape, tile_slice.shape)
            )
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.tile_slice = getattr(obj, 'tile_slice', None)
        self.scheme_idx = getattr(obj, 'scheme_idx', None)

        # invalidate `tile_slice` in case `obj` is modified/reshaped
        # such that it doesn't match the `tile_slice` anymore
        if self.tile_slice is not None:
            if tuple(self.tile_slice.shape) != self.shape:
                self.tile_slice = None

    def reshape(self, *args, **kwargs):
        # NOTE: "shedding" our DataTile class on reshape, as we can't properly update
        # the slice to keep it aligned with the reshape process.
        return np.asarray(self).view(np.ndarray).reshape(*args, **kwargs)

    @property
    def flat_data(self):
        """
        Flatten the data.

        The result is a 2D array where each row contains pixel data
        from a single frame. It is just a reshape, so it is a view into
        the original data.
        """
        shape = self.tile_slice.shape
        tileshape = (
            shape.nav.size,    # stackheight, number of frames we process at once
            shape.sig.size,    # framesize, number of pixels per tile
        )
        return self.reshape(tileshape)

    def __repr__(self):
        return "<DataTile %r scheme_idx=%d>" % (self.tile_slice, self.scheme_idx)


class Negotiator:
    """
    Tile shape negotiator. The main functionality is in `get_scheme`,
    which, given a `udf`, `partition` and `read_dtype` will generate
    a `TilingScheme` that is compatible with both the `UDF` and the
    `DataSet`, possibly even optimal.
    """

    def validate(self, shape, partition, size, io_max_size, itemsize, base_shape, corrections):
        partition_shape = partition.shape

        # we need some wiggle room with the size, because there may be a harder
        # lower size value for some cases (for example HDF5, which overrides
        # some of the sizing negotiation we are doing here)
        size_px = max(size, io_max_size) // itemsize
        if any(s > ps for s, ps in zip(shape, partition_shape)):
            raise ValueError("generated tileshape does not fit the partition")
        if np.prod(shape, dtype=np.int64) > size_px:
            message = "shape %r (%d) does not fit into size %d" % (
                shape, np.prod(shape, dtype=np.int64), size_px
            )
            # The shape might be exceeded if dead pixel correction didn't find a
            # valid tiling scheme. In that case it falls back to by-frame processing.
            if corrections.get_excluded_pixels() is not None and shape[0] == 1:
                warnings.warn(message)
            else:
                raise ValueError(message)
        for dim in range(len(base_shape)):
            if shape[dim] % base_shape[dim] != 0:
                raise ValueError(
                    f"The tileshape {shape} is incompatible with base "
                    f"shape {base_shape} in dimension {dim}."
                )

    def get_scheme(
            self, udfs, partition, read_dtype: np.dtype, roi: np.ndarray,
            corrections: CorrectionSet = None):
        """
        Generate a :class:`TilingScheme` instance that is
        compatible with both the given `udf` and the
        :class:~`libertem.io.dataset.base.DataSet`.

        Parameters
        ----------

        udfs : List[UDF]
            The concrete UDF to optimize the tiling scheme for.
            Depending on the method (tile, frame, partition)
            and preferred total input size and depth.

        partition : Partition
            The `TilingScheme` is created specifically
            for the given `Partition`, so it can adjust
            even in the face of different partition sizes/shapes.

        read_dtype
            The dtype in which the data will be fed into the UDF

        roi : np.ndarray
            Region of interest

        corrections : CorrectionSet
            Correction set to consider in negotiation
        """
        itemsize = np.dtype(read_dtype).itemsize

        # FIXME: let the UDF define upper bound for signal size (lower bound, too?)
        # (signal buffers should fit into the L2 cache)
        # try not to waste page faults:
        min_sig_size = partition.get_min_sig_size()

        # This already takes corrections into account through a different pathway
        need_decode = partition.need_decode(roi=roi, read_dtype=read_dtype, corrections=corrections)

        io_max_size = self._get_io_max_size(partition, itemsize, need_decode)

        depths = [
            self._get_min_depth(udf, partition)
            for udf in udfs
        ]
        depth = max(depths)  # take the largest min-depth
        base_shape = self._get_base_shape(udfs, partition, roi)

        sizes = [
            self._get_size(
                io_max_size, udf, itemsize, partition, base_shape,
            )
            for udf in udfs
        ]
        if any(
            udf.get_method() == "partition"
            for udf in udfs
        ):
            size = max(sizes)  # by partition wants to be big, ...
        else:
            size = min(sizes)
        size_px = size // itemsize

        if corrections is not None and corrections.have_corrections():
            # The correction has to make sure that there are no excluded pixels
            # at tile boundaries
            base_shape = corrections.adjust_tileshape(
                tile_shape=base_shape,
                sig_shape=tuple(partition.shape.sig),
                base_shape=base_shape,
            )

        # first, scale `base_shape` up to contain at least `min_sig_size` items:
        min_factors = self._get_scale_factors(
            base_shape,
            containing_shape=partition.shape.sig,
            size=min_sig_size,
        )

        min_base_shape = self._scale_base_shape(base_shape, min_factors)

        # considering the min size, calculate the max depth:
        max_depth = max(1, size_px // np.prod(min_base_shape, dtype=np.int64))
        if depth > max_depth:
            depth = max_depth

        full_base_shape = (1,) + tuple(base_shape)
        min_factors = (depth,) + tuple(min_factors)

        if roi is None:
            containing_shape = partition.shape
        else:
            containing_shape = partition.slice.adjust_for_roi(roi.reshape(-1)).shape

        factors = self._get_scale_factors(
            full_base_shape,
            containing_shape=containing_shape,
            size=size_px,
            min_factors=min_factors,
        )
        tileshape = self._scale_base_shape(full_base_shape, factors)

        # the partition has a "veto" on the tileshape:
        # FIXME: this veto may break if the base shape was adjusted
        # above, and we need to be careful not to break corrections with this,
        # and also fulfill requests of per-frame reading
        log.debug("tileshape before adjustment: %r", (tileshape,))
        tileshape = partition.adjust_tileshape(tileshape, roi=roi)
        log.debug("tileshape after adjustment: %r", (tileshape,))

        self.validate(
            tileshape, partition, size, io_max_size, itemsize, full_base_shape, corrections,
        )
        return TilingScheme.make_for_shape(
            tileshape=Shape(tileshape, sig_dims=partition.shape.sig.dims),
            dataset_shape=partition.meta.shape,
            debug={
                "min_factors": min_factors,
                "factors": factors,
                "tileshape": tileshape,
                "size": size,
                "size_px": size_px,
                "full_base_shape": full_base_shape,
                "need_decode": need_decode,
                "depth": depth,
            }
        )

    def _get_io_max_size(self, partition, itemsize, need_decode):
        if need_decode:
            io_max_size = partition.get_max_io_size()
            if io_max_size is None:
                io_max_size = 2**20
        else:
            io_max_size = itemsize * np.prod(partition.shape, dtype=np.int64)
        return io_max_size

    def _get_scale_factors(self, shape, containing_shape, size, min_factors=None):
        """
        Generate scaling factors to scale `shape` up to `size` elements,
        while being constrained to `containing_shape`.
        """
        log.debug(
            "_get_scale_factors in: shape=%r, containing_shape=%r, size=%r, min_factors=%r",
            shape, containing_shape, size, min_factors
        )
        assert len(shape) == len(containing_shape)
        if min_factors is None:
            factors = [1] * len(shape)
        else:
            factors = list(min_factors)
        max_factors = tuple(
            cs // s
            for s, cs in zip(shape, containing_shape)
        )
        prelim_shape = self._scale_base_shape(shape, factors)
        rest = size / np.prod(prelim_shape, dtype=np.int64)
        if rest < 1:
            rest = 1
        for idx in range(len(shape)):
            max_factor = max_factors[idx]
            factor = int(math.floor(rest * factors[idx]))
            if factor < factors[idx]:
                factor = factors[idx]
            if factor > max_factor:
                factor = max_factor
            factors[idx] = factor
            prelim_shape = self._scale_base_shape(shape, factors)
            rest = max(1, math.floor(size / np.prod(prelim_shape, dtype=np.int64)))
        log.debug(
            "_get_scale_factors out: %r",
            factors,
        )
        return factors

    def _scale_base_shape(self, base_shape, factors):
        assert len(factors) == len(base_shape)
        return tuple(
            f * bs
            for f, bs in zip(factors, base_shape)
        )

    def _get_default_size(self):
        # FIXME: adjust size to L3 // number of workers per node
        return 1*2**20

    def _get_udf_size_pref(self, udf):
        from libertem.udf import UDF
        udf_prefs = udf.get_tiling_preferences()
        size = udf_prefs.get("total_size", np.inf)
        if size is UDF.TILE_SIZE_BEST_FIT:
            size = self._get_default_size()
        return size

    def _get_size(self, io_max_size, udf, itemsize, partition, base_shape):
        """
        Calculate the maximum tile size in bytes
        """
        udf_method = udf.get_method()
        partition_size = itemsize * np.prod(partition.shape, dtype=np.int64)
        partition_size_sig = itemsize * np.prod(partition.shape.sig, dtype=np.int64)
        if udf_method == "frame":
            size = max(self._get_default_size(), partition_size_sig)
        elif udf_method == "partition":
            size = partition_size
        elif udf_method == "tile":
            # start with the UDF size preference:
            size = self._get_udf_size_pref(udf)

            # constrain to maximum read size
            size = min(size, io_max_size)

            # if the base_shape is larger than the current maximum size,
            # we need to increase the size:
            base_size = itemsize * np.prod(base_shape, dtype=np.int64)
            size = max(base_size, size)
        return size

    def _get_base_shape(self, udfs, partition, roi):
        methods = [
            udf.get_method()
            for udf in udfs
        ]
        if any(m == "frame" or m == "partition" for m in methods):
            base_shape = partition.shape.sig
        else:
            # only by tile:
            base_shape = Shape(
                partition.get_base_shape(roi=roi),
                sig_dims=partition.shape.sig.dims
            ).sig
        return base_shape

    def _get_udf_depth_pref(self, udf, partition):
        from libertem.udf import UDF
        udf_prefs = udf.get_tiling_preferences()
        depth = udf_prefs.get("depth", UDF.TILE_DEPTH_DEFAULT)
        if depth is UDF.TILE_DEPTH_DEFAULT:
            depth = 32
        if depth > partition.shape[0]:
            depth = partition.shape[0]
        return depth

    def _get_min_depth(self, udf, partition):
        udf_method = udf.get_method()

        if udf_method == "partition":
            return partition.shape[0]
        elif udf_method == "tile":
            return self._get_udf_depth_pref(udf, partition)
        return 1
