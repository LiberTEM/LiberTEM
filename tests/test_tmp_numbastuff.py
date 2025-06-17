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


def test_numbastuff():
    slices_arr = np.array([[[0,  0], [16, 16]]])
    sig_shape = (16, 16)

    sig_origins = np.array([
        numba_ravel_multi_index_single(slices_arr[slice_idx][0], sig_shape)
        for slice_idx in range(slices_arr.shape[0])
    ])
    print(sig_origins)


if __name__ == "__main__":
    print("running tests")
    test_numbastuff()
    print("Test passed successfully.")
