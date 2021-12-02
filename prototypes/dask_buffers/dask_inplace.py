from collections import namedtuple

fake_np_flags = namedtuple('Flags', ['c_contiguous'])

class DaskInplaceBufferWrapper:
    def __init__(self, dask_array):
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

    def set_slice(self, slices):
        self._slice = slices

    def clear_slice(self):
        self._slice = None

    def __getitem__(self, key):
        if self._slice is None:
            return self._array[key]
        else:
            combined_slice = combine_slices_multid(self._slice, key, self._array.shape)
            return self._array[combined_slice]

    def __setitem__(self, key, value):
        if self._slice is None:
            self._array[key] = value
        else:
            combined_slice = combine_slices_multid(self._slice, key, self._array.shape)
            self._array[combined_slice] = value


def combine_slices_multid(slices1, slices2, shape):
    if not isinstance(slices1, tuple):
        slices1 = (slices1,)
    if not isinstance(slices2, tuple):
        slices2 = (slices2,)
    combined_slices = []
    null_slice = slice(None, None, None)
    for _slice1, _slice2, _dimension in itertools.zip_longest(slices1, slices2, shape, fillvalue=null_slice):
        combined_slice = combine_slices(_slice1, _slice2, _dimension)
        combined_slices.append(combined_slice)
    return tuple(combined_slices)


def combine_slices(slice1, slice2, length):
    """
    https://stackoverflow.com/a/26783035


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

    # First get the step sizes of the two slices.
    slice1_step = (slice1.step if slice1.step is not None else 1)
    slice2_step = (slice2.step if slice2.step is not None else 1)

    # The final step size
    step = slice1_step * slice2_step

    # Use slice1.indices to get the actual indices returned from slicing with slice1
    slice1_indices = slice1.indices(length)

    # We calculate the length of the first slice
    slice1_length = (abs(slice1_indices[1] - slice1_indices[0]) - 1) // abs(slice1_indices[2])

    # If we step in the same direction as the start,stop, we get at least one datapoint
    if (slice1_indices[1] - slice1_indices[0]) * slice1_step > 0:
        slice1_length += 1
    else:
        # Otherwise, The slice is zero length.
        return slice(0,0,step)

    # Use the length after the first slice to get the indices returned from a
    # second slice starting at 0.
    slice2_indices = slice2.indices(slice1_length)

    # if the final range length = 0, return
    if not (slice2_indices[1] - slice2_indices[0]) * slice2_step > 0:
        return slice(0,0,step)

    # We shift slice2_indices by the starting index in slice1 and the 
    # step size of slice1
    start = slice1_indices[0] + slice2_indices[0] * slice1_step
    stop = slice1_indices[0] + slice2_indices[1] * slice1_step

    # slice.indices will return -1 as the stop index when slice.stop should be set to None.
    if start > stop:
        if stop < 0:
            stop = None

    return slice(start, stop, step)


if __name__ == '__main__':
    import dask.array as da
    import numpy as np
    
    tt = da.ones((5, 5))

    dar = DaskInplaceBufferWrapper(tt)
    dar.set_slice(np.s_[0, :])
    dar[:] += 5
    dar[:] *= 5
    dar.set_slice(np.s_[:, 1])
    dar[:] += 5
    dar.clear_slice()

    print(dar.data.compute())
