import pytest

import dask.array as da
import numpy as np

from libertem.executor.utils.dask_inplace import DaskInplaceWrapper
from utils import _mk_random


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
@pytest.mark.parametrize(
    "set_wrapper", (
        np.asarray,
        da.asarray,
        lambda x: x,
    ))
def test_inplace_set_with_ints(shape, set_wrapper):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = np.s_[4, :]
    subslice = np.s_[5]

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = set_wrapper(55.)
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
    with pytest.raises(IndexError):
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


def int_for_dim(dim):
    return np.random.randint(-dim, dim)


def slice_range_for_dim(dim):
    start = int_for_dim(dim)
    stop = np.random.randint(start, dim)
    possible_steps = [None, 1]
    if stop - start > 1:
        # This avoids a bug in Dask where if step > (stop - start)
        # it is computed to be a zero-length slice, raising a ValueError
        # whereas numpy does a no-op
        possible_steps += [2]
    step = np.random.choice(possible_steps)
    return slice(start, stop, step)


def null_slice(dim=None):
    return slice(None, None, None)


def ellipsis(dim=None):
    return Ellipsis


def random_slice(shape, final_ellipsis=False):
    slices = tuple()
    has_ellipsis = False
    for dim in shape:
        if np.random.choice([True, False], p=[0.2, 0.8]):
            break
        choices = [null_slice]
        if dim > 1:
            choices += [int_for_dim, slice_range_for_dim]
        if not has_ellipsis:
            choices += [ellipsis]
        slicetype = np.random.choice(choices)
        slices = slices + (slicetype(dim),)
        # Handle double Ellipsis case
        if slices[-1] is Ellipsis:
            has_ellipsis = True
            if final_ellipsis:
                break
    if len(slices) == 0:
        slices = null_slice()
    return slices


@pytest.mark.parametrize(
    "repeat_number", range(20))
@pytest.mark.parametrize(
    "shape", ((16, 8, 32, 64),
              (16, 16, 8),
              (16, 16),
              (16,),
              (1,),))
@pytest.mark.parametrize(
    "creator",
    (np.random.uniform, da.random.uniform),
)
def test_random_set_with_array(repeat_number, shape, creator):
    dtype = np.float32
    data, dask_wrapped = get_wrapped_data(shape, dtype)

    sl = random_slice(shape)

    # Encountered a rare bug where my generated slice would
    # be invalid for numpy slicing for an unknown reason
    # as an out of range integer for dimension
    try:
        slice_into = data[sl]
    except IndexError:
        print(f'Skipping due to invalid primary slice {sl} on shape {shape}')
        return
    if np.isscalar(slice_into):
        subslice = None
    else:
        subslice = random_slice(slice_into.shape, final_ellipsis=True)

    # Idem
    try:
        target_shape = data[sl][subslice].shape
    except IndexError:
        print(f'Skipping due to invalid secondary slice {subslice} '
              f'on shape {shape} with sl {sl}')
        return

    try:
        set_values = creator(size=target_shape)
    except ZeroDivisionError:
        # Dask auto-chunking cannot handle a shape with zero dimensions
        assert 0 in target_shape
        print(f'Failed while creating values of shape {target_shape}')
        return

    dask_wrapped.set_slice(sl)
    dask_wrapped[subslice] = set_values

    try:
        set_values = set_values.compute()
    except AttributeError:
        pass
    if set_values.size == 1:
        set_values = set_values.item()

    # Strange-ish case of Dask supporting an assignment but Numpy raises!
    if np.isscalar(slice_into):
        # Would raise TypeError if using subslice even if None
        data[sl] = set_values
    else:
        data[sl][subslice] = set_values
    assert np.allclose(dask_wrapped.data.compute(), data)
