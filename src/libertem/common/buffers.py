import mmap
import math
import numpy as np


def _alloc_aligned(size):
    # round up to 4k blocks:
    blocksize = 4096
    blocks = math.ceil(size / blocksize)

    # flags are by default MAP_SHARED, which has to be set
    # to prevent possible corruption (see open(2)). If you
    # are adding flags here, make sure to include MAP_SHARED
    # (and check for windows compat)
    return mmap.mmap(-1, blocksize * blocks)


def bytes_aligned(size):
    buf = _alloc_aligned(size)
    # _alloc_aligned may give us more memory (for alignment reasons), so crop it off the end:
    return memoryview(buf)[:size]


def empty_aligned(size, dtype):
    size_flat = np.product(size)
    dtype = np.dtype(dtype)
    buf = _alloc_aligned(dtype.itemsize * size_flat)
    # _alloc_aligned may give us more memory (for alignment reasons), so crop it off the end:
    npbuf = np.frombuffer(buf, dtype=dtype)[:size_flat]
    return npbuf.reshape(size)


def zeros_aligned(size, dtype):
    if dtype == np.object or np.product(size) == 0:
        res = np.zeros(size, dtype=dtype)
    else:
        res = empty_aligned(size, dtype)
        res[:] = 0
    return res


class BufferWrapper(object):
    """
    Helper class to automatically allocate buffers, either for partitions or
    the whole dataset, and create views for partitions or single frames.

    This is used as a helper to allow easy merging of results without needing
    to manually handle indexing.

    Usually, as a user, you only need to instantiate this class, specifying `kind`,
    `dtype` and sometimes `extra_shape` parameters. Most methods are meant to be called
    from LiberTEM-internal code, for example the UDF functionality.

    This class is array_like, so you can directly use it, for example, as argument
    for numpy functions.
    """
    def __init__(self, kind, extra_shape=(), dtype="float32"):
        """
        Parameters
        ----------
        kind : "nav", "sig" or "single"
            The abstract shape of the buffer, corresponding either to the navigation
            or the signal dimensions of the dataset, or a single value.

        extra_shape : optional, tuple of int
            You can specify additional dimensions for your data. For example, if
            you want to store 2D coords, you would specify (2,) here.

        dtype : string or numpy dtype
            The dtype of this buffer
        """
        self._kind = kind
        self._extra_shape = extra_shape
        self._dtype = np.dtype(dtype)
        self._data = None
        # set to True if the data coords are global ds coords
        self._data_coords_global = False
        self._shape = None
        self._ds_shape = None
        self._roi = None

    def set_roi(self, roi):
        if roi is not None:
            roi = roi.reshape((-1,))
        self._roi = roi

    def set_shape_partition(self, partition, roi=None):
        self.set_roi(roi)
        roi_count = None
        if roi is not None:
            roi_part = self._roi[partition.slice.get(nav_only=True)]
            roi_count = np.count_nonzero(roi_part)
            assert roi_count <= partition.shape[0]
            assert roi_part.shape[0] == partition.shape[0]
        self._shape = self._shape_for_kind(self._kind, partition.shape, roi_count)

    def set_shape_ds(self, dataset, roi=None):
        self.set_roi(roi)
        roi_count = None
        if roi is not None:
            roi_count = np.count_nonzero(self._roi)
        self._shape = self._shape_for_kind(self._kind, dataset.shape.flatten_nav(), roi_count)
        self._ds_shape = dataset.shape

    def _shape_for_kind(self, kind, orig_shape, roi_count=None):
        if self._kind == "nav":
            if roi_count is None:
                nav_shape = tuple(orig_shape.nav)
            else:
                nav_shape = (roi_count,)
            return nav_shape + self._extra_shape
        elif self._kind == "sig":
            return tuple(orig_shape.sig) + self._extra_shape
        elif self._kind == "single":
            if len(self._extra_shape) > 0:
                return self._extra_shape
            else:
                return (1, )
        else:
            raise ValueError("unknown kind: %s" % kind)

    @property
    def data(self):
        """
        Get the buffer contents in shape that corresponds to the
        original dataset shape. If a ROI is set, embed the result into a new
        array; unset values have nan value, if supported by the underlying dtype.
        """
        if self._roi is None or self._kind != 'nav':
            return self._data.reshape(self._shape_for_kind(self._kind, self._ds_shape))
        shape = self._shape_for_kind(self._kind, self._ds_shape)
        wrapper = np.full(shape, np.nan, dtype=self._dtype)
        wrapper[self._roi.reshape(self._ds_shape.nav)] = self._data
        return wrapper

    @property
    def raw_data(self):
        """
        Get the raw data underlying this buffer, which is flattened and
        may be even filtered to a ROI
        """
        return self._data

    def __array__(self):
        """
        returns the "wrapped"/reshaped array, see above
        """
        return self.data

    def allocate(self):
        """
        Allocate a new buffer, in the shape previously set
        via one of the `set_shape_*` methods.
        """
        assert self._shape is not None
        assert self._data is None
        if self.roi_is_zero:
            self._data = zeros_aligned(1, dtype=self._dtype)
        else:
            self._data = zeros_aligned(self._shape, dtype=self._dtype)

    def has_data(self):
        return self._data is not None

    @property
    def roi_is_zero(self):
        return np.product(self._shape) == 0

    def _slice_for_partition(self, partition):
        """
        Get a Slice into self._data for `partition`, taking the current ROI into account.

        Because _data is "compressed" if a ROI is set, we can't directly index and must
        calculate a new slice from the ROI.
        """
        return partition.slice.adjust_for_roi(self._roi)

    def get_view_for_partition(self, partition):
        """
        get a view for a single partition in a whole-result-sized buffer
        """
        if self._kind == "nav":
            slice_ = self._slice_for_partition(partition)
            return self._data[slice_.get(nav_only=True)]
        elif self._kind == "sig":
            return self._data[partition.slice.get(sig_only=True)]
        elif self._kind == "single":
            return self._data

    def get_view_for_frame(self, partition, tile, frame_idx):
        """
        get a view for a single frame in a partition- or dataset-sized buffer
        (partition-sized here means the reduced result for a whole partition,
        not the partition itself!)
        """
        assert partition.shape.dims == partition.shape.sig.dims + 1
        if self.roi_is_zero:
            raise ValueError("cannot get view for frame with zero ROI")
        if self._kind == "sig":
            return self._data[tile.tile_slice.get(sig_only=True)]
        elif self._kind == "nav":
            partition_slice = self._slice_for_partition(partition)
            if self._data_coords_global:
                offset = 0
            else:
                offset = partition_slice.origin[0]
            result_idx = (tile.tile_slice.origin[0] + frame_idx - offset,)
            # shape: (1,) + self._extra_shape
            if len(self._extra_shape) > 0:
                return self._data[result_idx]
            else:
                return self._data[result_idx + (np.newaxis,)]
        elif self._kind == "single":
            return self._data

    def get_view_for_tile(self, partition, tile):
        """
        get a view for a single tile in a partition-sized buffer
        (partition-sized here means the reduced result for a whole partition,
        not the partition itself!)
        """
        assert partition.shape.dims == partition.shape.sig.dims + 1
        if self.roi_is_zero:
            raise ValueError("cannot get view for tile with zero ROI")
        if self._kind == "sig":
            return self._data[tile.tile_slice.get(sig_only=True)]
        elif self._kind == "nav":
            partition_slice = self._slice_for_partition(partition)
            tile_slice = tile.tile_slice
            if self._data_coords_global:
                offset = 0
            else:
                offset = partition_slice.origin[0]
            result_start = tile_slice.origin[0] - offset
            result_stop = result_start + tile_slice.shape[0]
            # shape: (1,) + self._extra_shape
            if len(self._extra_shape) + tile_slice.shape[0] > 1:
                return self._data[result_start:result_stop]
            else:
                return self._data[result_start:result_stop, np.newaxis]
        elif self._kind == "single":
            return self._data

    def __repr__(self):
        return "<BufferWrapper kind=%s dtype=%s extra_shape=%s>" % (
            self._kind, self._dtype, self._extra_shape
        )


class AuxBufferWrapper(BufferWrapper):
    def new_for_partition(self, partition, roi):
        """
        Return a new AuxBufferWrapper for a specific partition,
        slicing the data accordingly and reducing it to the selected roi.

        This assumes to be called on an AuxBufferWrapper that was not created
        by this method, that is, it should have global coordinates without
        having the ROI applied.
        """
        # FIXME: right now creates a view for the partition slice, which
        # AFAIK means we serialize the whole array; we could optimize here
        # and only send over the partition slice. But maybe, later, when we
        # actually properly scatter and share data, this becomes obsolete anyways,
        # as we would scatter most likely for all partitions (to be flexible in node
        # assignment, for example for availability)
        assert self._data_coords_global
        ps = partition.slice.get(nav_only=True)
        buf = self.__class__(self._kind, self._extra_shape, self._dtype)
        if roi is not None:
            roi_part = roi.reshape(-1)[ps]
            new_data = self._data[ps][roi_part]
        else:
            new_data = self._data[ps]
        buf.set_buffer(new_data, is_global=False)
        buf.set_roi(roi)
        assert np.product(new_data.shape) > 0
        assert not buf._data_coords_global
        return buf

    def set_buffer(self, buf, is_global=True):
        """
        Set the underlying buffer to an existing numpy array.

        If is_global is True, the shape must match with the shape of nav or sig
        of the dataset, plus extra_shape, as determined by the `kind` and
        `extra_shape` constructor arguments.
        """
        assert self._data is None
        assert buf.dtype == self._dtype
        extra = self._extra_shape
        if not extra:
            extra = (1,)
        shape = (-1,) + extra
        self._data = buf.reshape(shape).squeeze()
        self._data_coords_global = is_global

    def __repr__(self):
        return "<AuxBufferWrapper kind=%s dtype=%s extra_shape=%s>" % (
            self._kind, self._dtype, self._extra_shape
        )
