import numpy as np


class BufferWrapper(object):
    """
    Helper class to automatically allocate buffers, either for partitions or
    the whole dataset, and create views for partitions or single frames.

    This is used as a helper to allow easy merging of results without needing
    to manually handle indexing.
    """
    def __init__(self, kind, extra_shape=(), dtype="float32"):
        self._kind = kind
        self._extra_shape = extra_shape
        self._dtype = np.dtype(dtype)
        self._data = None
        self._shape = None

    def set_shape_partition(self, partition):
        self._shape = self._shape_for_kind(self._kind, partition.shape)

    def set_shape_ds(self, dataset):
        self._shape = self._shape_for_kind(self._kind, dataset.raw_shape)

    def _shape_for_kind(self, kind, orig_shape):
        if self._kind == "nav":
            return tuple(orig_shape.nav) + self._extra_shape
        elif self._kind == "sig":
            return tuple(orig_shape.sig) + self._extra_shape
        else:
            raise ValueError("unknown kind: %s" % kind)

    @property
    def data(self):
        return self._data

    def allocate(self):
        """
        allocate a new buffer, in the shape previously set
        via one of the `set_shape_*` methods.
        """
        # TODO: alignment?
        assert self._shape is not None
        assert self._data is None
        self._data = np.zeros(self._shape, dtype=self._dtype)

    def set_buffer(self, buf):
        """
        Set the underlying buffer to an existing numpy array. The
        shape must match with the shape of nav or sig of the dataset,
        plus extra_shape, as determined by the `kind` and `extra_shape`
        constructor arguments.
        """
        assert self._data is None
        assert buf.shape == self._shape
        assert buf.dtype == self._dtype
        self._data = buf

    def has_data(self):
        return self._data is not None

    def get_view_for_partition(self, partition):
        if self._kind == "nav":
            return self._data[partition.slice.get(nav_only=True)]
        elif self._kind == "sig":
            return self._data[partition.slice.get(sig_only=True)]

    def get_view_for_frame(self, partition, tile, frame_idx):
        if self._kind == "sig":
            return self._data[partition.slice.get(sig_only=True)]
        elif self._kind == "nav":
            ref_slice = partition.slice
            tile_slice = tile.tile_slice.shift(ref_slice)
            start_of_tile = np.ravel_multi_index(
                tile_slice.origin[:-tile_slice.shape.sig.dims],
                tuple(partition.shape.nav),
            )
            result_idx = np.unravel_index(start_of_tile + frame_idx,
                                          partition.shape.nav)
            # shape: (1,) + self._extra_shape
            if len(self._extra_shape) > 0:
                return self._data[result_idx]
            else:
                return self._data[result_idx + (np.newaxis,)]
