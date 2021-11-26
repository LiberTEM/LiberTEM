import dask.array as da
import numpy as np


class DaskInplaceBufferWrapper:
    def __init__(self, dask_array):
        self._array = dask_array
        self._slice = None

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
        elif key == slice(None, None, None):
            return self._array[self._slice]
        else:
            raise

    def __setitem__(self, key, value):
        if self._slice is None:
            self._array[key] = value
        elif key == slice(None, None, None):
            self._array[self._slice] = value
        else:
            raise


if __name__ == '__main__':
    tt = da.ones((5, 5))

    dar = DaskInplaceBufferWrapper(tt)
    dar.set_slice(np.s_[0, :])
    dar[:] += 5
    dar[:] *= 5
    dar.set_slice(np.s_[:, 1])
    dar[:] += 5
    dar.clear_slice()

    print(dar.data.compute())
