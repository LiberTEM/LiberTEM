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
        start_at_frame, stop_before_frame, roi_nonzero, depth,
        slices_arr, fileset_arr, sig_shape,
        bpp, sync_offset=0, extra=None, frame_header_bytes=0, frame_footer_bytes=0,
    ):
        frame_indices = np.arange(max(0, start_at_frame), stop_before_frame)
        num_indices = frame_indices.shape[0]
        return num_indices
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
