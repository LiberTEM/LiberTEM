import numpy as np
import numba


@numba.njit(boundscheck=True, nogil=True)
def numba_ravel_multi_index_single(multi_index, dims):
    # only supports the "single index" case
    idxs = range(len(dims) - 1, -1, -1)
    res = 0
    for idx in idxs:
        stride = 1
        for dimidx in range(idx + 1, len(dims)):
            stride *= dims[dimidx]
        res += multi_index[idx] * stride
    return res


def make_get_read_ranges():
    @numba.njit(boundscheck=True, cache=True, nogil=True)
    def _get_read_ranges_inner(
        start_at_frame, stop_before_frame, depth,
        slices_arr, sig_shape, sync_offset=0,
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
        inner_indices_stop = min(depth, num_indices)

        sig_origins = np.array([
            numba_ravel_multi_index_single(slices_arr[slice_idx][0], sig_shape)
            for slice_idx in range(slices_arr.shape[0])
        ])

        return inner_indices_start, slice_offset, inner_indices_stop, sig_origins

    return _get_read_ranges_inner


def test_numbastuff():
    read_ranges = make_get_read_ranges()
    read_ranges(
        start_at_frame=0,
        stop_before_frame=128,
        depth=1,
        slices_arr=np.array([[[0,  0], [16, 16]]]),
        sig_shape=(16, 16),
        sync_offset=0,
    )


if __name__ == "__main__":
    print("running tests")
    test_numbastuff()
    print("Test passed successfully.")
