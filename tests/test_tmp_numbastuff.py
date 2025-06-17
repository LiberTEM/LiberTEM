import numpy as np
from numba.typed import List as NumbaList

from libertem.common.numba import numba_ravel_multi_index_single as _ravel_multi_index, cached_njit
from libertem.io.dataset.base.tiling import (
    _default_px_to_bytes, _default_read_ranges_tile_block, _find_file_for_frame_idx
)


def make_get_read_ranges(
    px_to_bytes=_default_px_to_bytes,
    read_ranges_tile_block=_default_read_ranges_tile_block,
):
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
            num_indices = int(stop_before_frame - max(0, start_at_frame))
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
            num_indices = int(len(frame_indices))

            if start_at_frame < 0:
                slice_offset = np.sum(roi_nonzero < abs(sync_offset))
            else:
                slice_offset = np.sum(roi_nonzero < start_at_frame - sync_offset)

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


def test_numbastuff():
    read_ranges = make_get_read_ranges()
    read_ranges(
        start_at_frame=0,
        stop_before_frame=128,
        roi_nonzero=None,
        depth=1,
        slices_arr=np.array([[[0,  0], [16, 16]]]),
        fileset_arr=np.array([[0, 256, 0, 0]]),
        sig_shape=(16, 16),
        bpp=16,
        sync_offset=0,
        frame_header_bytes=0,
        frame_footer_bytes=0,
    )
