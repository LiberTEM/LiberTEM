import logging

import numba
from numba.typed import List as NumbaList
import numpy as np
from sparseconverter import check_shape

from libertem.common.numba import numba_ravel_multi_index_single as _ravel_multi_index, cached_njit


log = logging.getLogger(__name__)


@numba.njit(inline='always')
def _default_px_to_bytes(
    bpp, frame_in_file_idx, slice_sig_size, sig_size, sig_origin,
    frame_footer_bytes, frame_header_bytes, file_header_bytes,
    file_idx, read_ranges,
):
    """
    Convert from a slice (sig_origin, slice_sig_size) to a byte slice (start, stop)
    and append the result to the read_ranges numba List
    """
    # we are reading a part of a single frame, so we first need to find
    # the offset caused by headers and footers:
    footer_offset = frame_footer_bytes * frame_in_file_idx
    header_offset = frame_header_bytes * (frame_in_file_idx + 1)
    byte_offset = file_header_bytes + footer_offset + header_offset

    # now let's figure in the current frame index:
    # (go down into the file by full frames; `sig_size`)
    offset = byte_offset + frame_in_file_idx * sig_size * bpp // 8

    # offset in px in the current frame:
    sig_origin_bytes = sig_origin * bpp // 8

    start = offset + sig_origin_bytes

    # size of the sig part of the slice:
    sig_size_bytes = slice_sig_size * bpp // 8

    stop = start + sig_size_bytes

    read_ranges.append((file_idx, start, stop))


@numba.njit(boundscheck=True, nogil=True)
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
            file_header_bytes = f[3]

            # px_to_bytes is the format-specific translation of pixel
            # coordinates (slice_sig_size, sig_size, sig_origin)
            # to bytes, which are appended as tuples (file_idx, start, stop)
            # to the `read_ranges` list.
            px_to_bytes(
                bpp=bpp,
                frame_in_file_idx=frame_in_file_idx,
                slice_sig_size=slice_sig_size,
                sig_size=sig_size,
                sig_origin=sig_origin,
                frame_footer_bytes=frame_footer_bytes,
                frame_header_bytes=frame_header_bytes,
                file_header_bytes=file_header_bytes,
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
):
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

    bpp : int
        Bits per pixel, including padding

    Returns
    -------

    (tile_slice, read_ranges)
        read_ranges is an ndarray with shape (number_of_tiles, depth, 3)
        where the last dimension contains: file index, start_byte, stop_byte
    """

    @cached_njit(boundscheck=True, cache=True, nogil=True)
    def _get_read_ranges_inner(
        start_at_frame, stop_before_frame, roi_nonzero, depth,
        slices_arr, fileset_arr, sig_shape,
        bpp, sync_offset=0, extra=None, frame_header_bytes=0, frame_footer_bytes=0,
    ):
        result = NumbaList()
        # Use NumPy prod for Numba compilation
        sig_size = np.prod(np.array(sig_shape).astype(np.int64))

        if roi_nonzero is None:
            frame_indices = np.arange(max(0, start_at_frame), stop_before_frame)
            # in case of a negative sync_offset, start_at_frame can be negative
            if start_at_frame < 0:
                slice_offset = abs(sync_offset)
            else:
                slice_offset = start_at_frame - sync_offset
        else:
            shifted_roi = roi_nonzero + sync_offset
            roi_mask = np.logical_and(shifted_roi >= max(0, start_at_frame),
                                      shifted_roi < stop_before_frame)
            frame_indices = shifted_roi[roi_mask]

            if start_at_frame < 0:
                slice_offset = np.sum(roi_nonzero < abs(sync_offset))
            else:
                slice_offset = np.sum(roi_nonzero < start_at_frame - sync_offset)

        num_indices = frame_indices.shape[0]

        # indices into `frame_indices`:
        inner_indices_start = 0
        inner_indices_stop = min(depth, num_indices)

        # this should be `prod(..., axis=-1)``, which is not supported by numba yet:
        # slices that divide the signal dimensions:
        slice_sig_sizes = np.array([
            # Use NumPy prod for Numba compilation
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

        lengths = [len(res[1])
                   for res in result]
        max_rr_per_tile = max(lengths)

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


class DataTile:
    def __init__(self, data, tile_slice, scheme_idx):
        if isinstance(data, DataTile):
            data = data.data
        check_shape(data, tile_slice.shape)
        self._data = data
        self.tile_slice = tile_slice
        self.scheme_idx = scheme_idx

    @property
    def flat_data(self) -> np.ndarray:
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
        return self._data.reshape(tileshape)

    @property
    def dtype(self):
        return self._data.dtype

    @property
    def shape(self):
        return tuple(self.tile_slice.shape)

    @property
    def size(self):
        return self.tile_slice.shape.size

    @property
    def data(self):
        return self._data

    @property
    def c_contiguous(self):
        try:
            return self._data.flags.c_contiguous
        except AttributeError:
            return None

    def __repr__(self):
        return "<DataTile %r scheme_idx=%d>" % (self.tile_slice, self.scheme_idx)
