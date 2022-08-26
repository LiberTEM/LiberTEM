import pytest
import numpy as np

from libertem.utils.devices import detect
from libertem.common.array_formats import (
    CUDA, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, NUMPY,
    FORMATS, CUDA_FORMATS, NDFORMATS, as_format, array_format
)

from utils import _mk_random


d = detect()
has_cupy = d['cudas'] and d['has_cupy']


@pytest.mark.parametrize(
    'left', FORMATS
)
@pytest.mark.parametrize(
    'right', FORMATS
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
    if not has_cupy and (left in CUDA_FORMATS or right in CUDA_FORMATS):
        pytest.skip("No CUDA device or no CuPy, skipping CuPy test")
    if left in CUPY_SPARSE_FORMATS and dtype not in CUPY_SPARSE_DTYPES:
        pytest.skip(f"Dtype {dtype} not supported for left format {left}")
    if left in NDFORMATS:
        shape = (7, 11, 13, 17)
    else:
        shape = (7, 11)
    data = _mk_random(shape, dtype=dtype, format=left)
    if left == CUDA:
        assert array_format(data) == NUMPY
    else:
        assert array_format(data) == left
    left_ref = as_format(data, NUMPY)
    assert isinstance(left_ref, np.ndarray)
    converted = as_format(data, right)
    if right == CUDA:
        assert array_format(converted) == NUMPY
    else:
        assert array_format(converted) == right

    converted_back = as_format(converted, NUMPY)
    assert isinstance(converted_back, np.ndarray)

    if right not in NDFORMATS:
        left_ref = left_ref.reshape((left_ref.shape[0], -1))

    assert np.allclose(left_ref, converted_back)
