import pytest
import numpy as np

from libertem.utils.devices import detect
from libertem.common.array_backends import (
    CUDA, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, NUMPY,
    BACKENDS, CUDA_BACKENDS, ND_BACKENDS, for_backend, get_backend
)

from utils import _mk_random


d = detect()
has_cupy = d['cudas'] and d['has_cupy']


@pytest.mark.parametrize(
    'left', BACKENDS
)
@pytest.mark.parametrize(
    'right', BACKENDS
)
@pytest.mark.parametrize(
    'dtype', [
        bool, float, int,
        np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64,
        np.float32, np.float64, np.complex64, np.complex128,
    ]
)
def test_as_format(left, right, dtype):
    CUPY_SPARSE_DTYPES = {
        np.float32, np.float64, np.complex64, np.complex128
    }
    CUPY_SPARSE_FORMATS = {
        CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
    }
    print(left, right, dtype)
    if not has_cupy and (left in CUDA_BACKENDS or right in CUDA_BACKENDS):
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    if left in CUPY_SPARSE_FORMATS and dtype not in CUPY_SPARSE_DTYPES:
        pytest.skip(f"Dtype {dtype} not supported for left format {left}")
    if left in ND_BACKENDS:
        shape = (7, 11, 13, 17)
    else:
        shape = (7, 11)
    data = _mk_random(shape, dtype=dtype, array_backend=left)
    if left == CUDA:
        assert get_backend(data) == NUMPY
    else:
        assert get_backend(data) == left
    left_ref = for_backend(data, NUMPY)
    assert isinstance(left_ref, np.ndarray)
    converted = for_backend(data, right)
    if right == CUDA:
        assert get_backend(converted) == NUMPY
    else:
        assert get_backend(converted) == right

    converted_back = for_backend(converted, NUMPY)
    assert isinstance(converted_back, np.ndarray)

    if right not in ND_BACKENDS:
        left_ref = left_ref.reshape((left_ref.shape[0], -1))

    assert np.allclose(left_ref, converted_back)
