from collections import namedtuple
import itertools
from typing import Union
from numbers import Number

import dask
import numpy as np
import numpy.typing as nt


fake_np_flags = namedtuple('Flags', ['c_contiguous'])


class DaskInplaceWrapper:
    def __init__(self, dask_array):
        """
        Wraps a dask array providing methods such that array
        slicing can be done inplace even when slicing into a slice
        of the array. Works for simple slicing cases, i.e. slicing
        with slice(), integers, : and ...

        The first slice is set using DaskInplaceWrapper.set_slice(slice)
        while secondary slices are done using normal array[slice] syntax

        In the current usage, this wrapper is created and configured
        from DaskBufferWrapper._get_slice
        """
        assert isinstance(dask_array, dask.array.Array)
        self._array = dask_array
        self._slice = None

    @property
    def flags(self):
        return fake_np_flags(c_contiguous=True)

    @property
    def data(self):
        return self._array

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    def check_valid_slices(self, slices):
        if not isinstance(slices, tuple):
            slices = (slices,)
        assert all(isinstance(s, (slice, int)) or s in (Ellipsis, None) for s in slices)
        assert len(slices) <= self.ndim

    def set_slice(self, slices):
        self.check_valid_slices(slices)
        self._slice = slices

    def clear_slice(self):
        self._slice = None

    def unwrap_sliced(self):
        """
        Unwrap the array as a sliced view, if self._slice is set
        """
        if self._slice is None:
            return self._array
        else:
            return self._array[self._slice]

    def __getitem__(self, key):
        if self._slice is None:
            return self._array[key]
        else:
            combined_slice = combine_slices_multid(self._slice, key, self._array.shape)
            return self._array[combined_slice]

    def __setitem__(self, key, value: Union[nt.ArrayLike, Number]):
        if not np.isscalar(value) and value.size == 1 and not isinstance(value, dask.array.Array):
            # Avoids deprecated Numpy behaviour of implicitly
            # extracting a scalar from an array during assignment
            value = value.item()
        try:
            if self._slice is None:
                self._array[key] = value
            else:
                combined_slice = combine_slices_multid(self._slice, key, self._array.shape)
                self._array[combined_slice] = value
        except ValueError as e:
            if not np.isscalar(value) and value.size == 0:
                # Setting a zero-size slice is a no-op, but
                # can raise a ValueError in Dask
                return
            raise e
        except NotImplementedError:
            raise NotImplementedError(
                "Assignment into Dask array failed. This feature requires "
                f"Dask version >= 2021.4.1. Installed is {dask.__version__}."
            )


def combine_slices_multid(slices1, slices2, shape):
    null_slice = slice(None, None, None)

    if not isinstance(slices1, tuple):
        slices1 = (slices1,)
    if not isinstance(slices2, tuple):
        slices2 = (slices2,)

    if all([s == null_slice for s in slices2]):
        return slices1

    if all([s == null_slice for s in slices1]):
        return slices2

    # Handle the Ellipsis case for slice1
    # can handle Ellipsis leading, central and training
    # in slices 1 because it is the primary slice
    # This will pad slice1 to be the same size as shape
    if Ellipsis in slices1:
        assert sum(e is Ellipsis for e in slices1) == 1
        el_idx = slices1.index(Ellipsis)
        before = slices1[:el_idx]
        after = slices1[el_idx + 1:]
        to_add = len(shape) - (len(before) + (len(after)))
        _null = (null_slice,) * to_add
        slices1 = before + _null + after

    # Handle the Ellipsis case for slice2
    # need to work on this to handle leading or central Ellipsis
    # trailing ellipsis is supported as the padding is implemented after
    if Ellipsis in slices2:
        if Ellipsis != slices2[-1] or sum(e is Ellipsis for e in slices1) > 1:
            raise IndexError("No support for Ellipsis in subslice except at end")
        slices2 = slices2[:-1]

    # Must pad slices1 to length of array itself so that
    # we can use it as a reference for padding slices2
    while len(slices1) < len(shape):
        slices1 = slices1 + (null_slice,)

    # If slice1 contains integers we must pad slices2 for combining
    # This section of code is not complete/robust, there are cases
    # where slices1 is all non-integer but slices2 is not of the same length!
    # e.g. slices1 = (slice(0, 64), slice(0, 64)) and slices2 = (slice(0, 32),)
    # need to have logic for slicing only the first dimension here
    s2_gen = iter(slices2)
    _slices2 = []
    for s1 in slices1:
        if isinstance(s1, int):
            _slices2.append(null_slice)
        else:
            _slices2.append(next(s2_gen, null_slice))
    next_el = next(s2_gen, None)
    assert next_el is None
    slices2 = tuple(_slices2)

    combined_slices = []
    for _slice1, _slice2, _dimension in itertools.zip_longest(slices1, slices2, shape,
                                                              fillvalue=null_slice):
        combined_slice = combine_slices(_slice1, _slice2, _dimension)
        combined_slices.append(combined_slice)
    return tuple(combined_slices)


def combine_slices(slice1, slice2, length):
    """
    https://stackoverflow.com/a/26783035
    Modified to support slices which are just element selection
    i.e. an integer for slice1 or slice2


    returns a slice that is a combination of the two slices.
    As in
      x[slice1][slice2]
    becomes
      combined_slice = slice_combine(slice1, slice2, len(x))
      x[combined_slice]

    :param slice1: The first slice
    :param slice2: The second slice
    :param length: The length of the first dimension of data being sliced. (eg len(x))
    """
    if isinstance(slice1, int):
        # by definition we can't slice further on this dimension
        assert slice2 is None or slice2 == slice(None, None, None)
        return slice1

    # First get the step sizes of the two slices.
    slice1_step = (slice1.step if slice1.step is not None else 1)

    # Use slice1.indices to get the actual indices returned from slicing with slice1
    slice1_indices = slice1.indices(length)

    if isinstance(slice2, int):
        # Represents a single element from array[slice1]
        return range(*slice1_indices)[slice2]

    # From this point we are combining two non-integer slices
    slice2_step = (slice2.step if slice2.step is not None else 1)

    # The final step size
    step = slice1_step * slice2_step

    # We calculate the length of the first slice
    slice1_length = (abs(slice1_indices[1] - slice1_indices[0]) - 1) // abs(slice1_indices[2])

    # If we step in the same direction as the start,stop, we get at least one datapoint
    if (slice1_indices[1] - slice1_indices[0]) * slice1_step > 0:
        slice1_length += 1
    else:
        # Otherwise, The slice is zero length.
        return slice(0, 0, step)

    # Use the length after the first slice to get the indices returned from a
    # second slice starting at 0.
    slice2_indices = slice2.indices(slice1_length)

    # if the final range length = 0, return
    if not (slice2_indices[1] - slice2_indices[0]) * slice2_step > 0:
        return slice(0, 0, step)

    # We shift slice2_indices by the starting index in slice1 and the
    # step size of slice1
    start = slice1_indices[0] + slice2_indices[0] * slice1_step
    stop = slice1_indices[0] + slice2_indices[1] * slice1_step

    # slice.indices will return -1 as the stop index when slice.stop should be set to None.
    if start > stop:
        if stop < 0:
            stop = None

    return slice(start, stop, step)
