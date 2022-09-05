import pytest
import numpy as np

from libertem.utils.devices import detect
from libertem.common.array_backends import (
    CUDA, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, NUMPY,
    BACKENDS, CUDA_BACKENDS, ND_BACKENDS, check_shape, for_backend, get_backend
)
from libertem.common.math import prod

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
def test_for_backend(left, right, dtype):
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
    shape = (7, 11, 13, 17)
    left_ref = _mk_random(shape, dtype=dtype, array_backend=NUMPY)
    assert isinstance(left_ref, np.ndarray)
    data = for_backend(left_ref, left)
    if left == CUDA:
        assert get_backend(data) == NUMPY
    else:
        assert get_backend(data) == left

    converted = for_backend(data, right)
    if right == CUDA:
        assert get_backend(converted) == NUMPY
    else:
        assert get_backend(converted) == right

    converted_back = for_backend(converted, NUMPY)
    assert isinstance(converted_back, np.ndarray)

    if left in ND_BACKENDS and right in ND_BACKENDS:
        target_shape = shape
    else:
        target_shape = (shape[0], prod(shape[1:]))

    if left in ND_BACKENDS:
        check_shape(converted, shape)
    else:
        check_shape(converted, target_shape)

    assert converted.shape == target_shape
    assert converted_back.shape == target_shape

    assert np.allclose(left_ref.reshape(target_shape), converted_back)
