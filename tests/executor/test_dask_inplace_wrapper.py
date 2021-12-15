import sys
import pytest

import dask.array as da
import numpy as np

from libertem.executor.utils.dask_inplace import DaskInplaceWrapper
from utils import _mk_random


pytestmark = pytest.mark.skipif(sys.version_info < (3, 7),
                                reason="Cannot inplace assign for Dask on Python3.6")


def get_wrapped_data(shape, dtype):
    data = _mk_random(shape, dtype=dtype)
    data_dask = da.from_array(data.copy())
    dask_wrapped = DaskInplaceWrapper(data_dask)
    return data, dask_wrapped


def test_inplace_methods():
    shape = (16, 8, 32, 64)
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    assert dask_wrapped.flags.c_contiguous
    assert np.allclose(dask_wrapped.data.compute(), data)
    assert np.allclose(dask_wrapped.unwrap_sliced().compute(), data)
    assert dask_wrapped.shape == data.shape
    assert dask_wrapped.dtype == dtype
    assert dask_wrapped.size == data.size


def test_inplace_raw():
    shape = (16, 8, 32, 64)
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)
    sl = np.s_[4, :, :, :]

    assert np.allclose(dask_wrapped[sl].compute(), data[sl])
    dask_wrapped[sl] = 55.
    data[sl] = 55.
    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_get_with_slice(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4, :, :, :]
    subslice = np.s_[5, 0, 4]

    dask_wrapped.set_slice(sl)
    assert np.allclose(dask_wrapped.unwrap_sliced().compute(), data[sl])
    assert dask_wrapped[subslice].compute() == data[sl][subslice]


@pytest.mark.parametrize(
    "shape", ((8, 8),))
def test_inplace_set_with_ints(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4, :]
    subslice = np.s_[5]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)

    dask_wrapped.clear_slice()
    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 64),))
def test_inplace_set_with_ints2(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[:, 2, :]
    subslice = np.s_[3, 4]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 16),))
def test_inplace_set_with_neg_ints(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[-3]
    subslice = np.s_[-7]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_colon(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4, :, :, :]

    dask_wrapped.set_slice(sl)
    dask_wrapped[:] = 55.
    data[sl][:] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_ranges(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4:7, :, :, :]
    subslice = np.s_[3:6, 10:22, :]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_ranges2(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[:]
    subslice = np.s_[3:6, 10:22, :]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_ellipsis(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4:7, ...]
    subslice = np.s_[3:6, 10:22, :]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_ellipsis2(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4:7, ...]
    subslice = np.s_[3:6, ...]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = 55.
    data[sl][subslice] = 55.

    assert np.allclose(dask_wrapped.data.compute(), data)


def test_inplace_set_with_ellipsis3():
    _, dask_wrapped = get_wrapped_data((16, 8, 32, 64), np.float32)

    sl = np.s_[4:7, ...]
    # No support for this yet !
    subslice = np.s_[..., 3:6]

    dask_wrapped.set_slice(sl)
    with pytest.raises(NotImplementedError):
        dask_wrapped[subslice] = 55.


@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),))
def test_inplace_set_with_array(shape):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4]
    subslice = np.s_[3:6, 10:22, :]

    target_shape = data[sl][subslice].shape
    set_values = np.random.random(size=target_shape)

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = set_values
    data[sl][subslice] = set_values

    assert np.allclose(dask_wrapped.data.compute(), data)
