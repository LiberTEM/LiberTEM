import numpy as np


from libertem.common.numba import cached_njit
from libertem.io.dataset.base.tiling import (
    _default_px_to_bytes, _default_read_ranges_tile_block,
)


def make_get_read_ranges(
    px_to_bytes=_default_px_to_bytes,
    read_ranges_tile_block=_default_read_ranges_tile_block,
):
    @cached_njit(boundscheck=True, cache=True, nogil=True)
    def _get_read_ranges_inner(
        start_at_frame, stop_before_frame, depth,
        slices_arr, fileset_arr, sig_shape,
        bpp, sync_offset=0, extra=None, frame_header_bytes=0, frame_footer_bytes=0,
    ):
        # Use NumPy prod for Numba compilation

        num_indices = int(stop_before_frame - max(0, start_at_frame))
        # in case of a negative sync_offset, start_at_frame can be negative
        if start_at_frame < 0:
            slice_offset = abs(sync_offset)
        else:
            slice_offset = start_at_frame - sync_offset

        # indices into `frame_indices`:
        inner_indices_start = 0

        return num_indices, inner_indices_start, slice_offset

    return _get_read_ranges_inner


def test_numbastuff():
    read_ranges = make_get_read_ranges()
    read_ranges(
        start_at_frame=0,
        stop_before_frame=128,
        depth=1,
        slices_arr=np.array([[[0,  0], [16, 16]]]),
        fileset_arr=np.array([[0, 256, 0, 0]]),
        sig_shape=(16, 16),
        bpp=16,
        sync_offset=0,
        frame_header_bytes=0,
        frame_footer_bytes=0,
    )
