from typing import Any, Optional, Union, TYPE_CHECKING, Generic, TypeVar
from collections.abc import Iterable
from typing_extensions import Literal
import mmap
import math
import functools
from contextlib import contextmanager
import collections
import itertools

try:
    from itertools import batched
except ImportError:
    def batched(iterable, n):
        # batched('ABCDEFG', 3) --> ABC DEF G
        if n < 1:
            raise ValueError('n must be at least one')
        it = iter(iterable)
        while batch := tuple(itertools.islice(it, n)):
            yield batch

import numba
import numpy as np

from libertem.common.math import prod, count_nonzero, flat_nonzero
from libertem.common.slice import Slice, Shape
from .backend import get_use_cuda

if TYPE_CHECKING:
    from numpy import typing as nt

BufferKind = Literal['nav', 'sig', 'single']
BufferLocation = Optional[Literal['device']]
BufferUse = Literal['private', 'result_only']
BufferSize = Union[int, tuple[int, ...]]


def _alloc_aligned(size: int, blocksize: int = 4096) -> mmap.mmap:
    # round up to (default 4k) blocks:
    blocks = math.ceil(size / blocksize)

    # flags are by default MAP_SHARED, which has to be set
    # to prevent possible corruption (see open(2)). If you
    # are adding flags here, make sure to include MAP_SHARED
    # (and check for windows compat)
    buf = mmap.mmap(-1, blocksize * blocks)

    # FIXME: if `blocksize` > PAGE_SIZE, we may have to align the start
    # address here, too.

    return buf


def bytes_aligned(size: int) -> memoryview:
    buf = _alloc_aligned(size)
    # _alloc_aligned may give us more memory (for alignment reasons), so crop it off the end:
    return memoryview(buf)[:size]


def empty_aligned(size: BufferSize, dtype: "nt.DTypeLike") -> np.ndarray:
    size_flat = prod(size)
    dtype = np.dtype(dtype)
    buf = _alloc_aligned(dtype.itemsize * size_flat)
    # _alloc_aligned may give us more memory (for alignment reasons), so crop it off the end:
    npbuf: np.ndarray = np.frombuffer(buf, dtype=dtype)[:size_flat]
    return npbuf.reshape(size)


def zeros_aligned(size: BufferSize, dtype: "nt.DTypeLike") -> np.ndarray:
    if dtype == object or prod(size) == 0:
        res = np.zeros(size, dtype=dtype)
    else:
        res = empty_aligned(size, dtype)
        res[:] = 0
    return res


# FIXME: type annotation for cupy.ndarray without importing?
def to_numpy(a: Union[np.ndarray, Any]) -> np.ndarray:
    # .. versionadded:: 0.6.0
    cuda_device = get_use_cuda()
    if isinstance(a, np.ndarray):
        return a
    elif cuda_device is not None:
        # Try to avoid importing unless necessary
        import cupy
        if isinstance(a, cupy.ndarray):
            return cupy.asnumpy(a)
    # Falling through
    raise TypeError(f"I don't know how to convert {type(a)} here.")


def reshaped_view(a: np.ndarray, shape):
    '''
    Like :meth:`numpy.ndarray.reshape`, just guaranteed to
    return a view or throw an :class:`AttributeError` if
    no view can be created.

    .. versionadded:: 0.5.0

    Parameters
    ----------

    a : numpy.ndarray
        Array to create a view of
    shape : tuple
        Shape of the view to create

    Returns
    -------

    view : numpy.ndarray
        View into :code:`a` with shape :code:`shape`

    '''
    res = a.view()
    res.shape = shape
    return res


def disjoint(sl: Slice, slices: Iterable[Slice]):
    return all(sl.intersection_with(s2).is_null() for s2 in slices)


class BufferPool:
    """
    allocation pool for explicitly re-using (aligned) allocations
    """
    def __init__(self):
        self._buffers = collections.defaultdict(list)

    @contextmanager
    def zeros(self, size, dtype, alignment=4096):
        if dtype == object or prod(size) == 0:
            yield np.zeros(size, dtype=dtype)
        else:
            with self.empty(size, dtype, alignment) as res:
                res[:] = 0
                yield res

    @contextmanager
    def empty(self, size, dtype, alignment=4096):
        size_flat = prod(size)
        dtype = np.dtype(dtype)
        with self.bytes(dtype.itemsize * size_flat, alignment) as buf:
            # self.bytes may give us more memory (for alignment reasons), so
            # crop it off the end:
            npbuf = np.frombuffer(buf, dtype=dtype)[:size_flat]
            yield npbuf.reshape(size)

    @contextmanager
    def bytes(self, size, alignment=4096):
        buf = self.checkout_bytes(size, alignment)
        yield buf
        self.checkin_bytes(size, alignment, buf)

    def checkout_bytes(self, size, alignment):
        buffers = self._buffers[(size, alignment)]
        try:
            buf = buffers.pop()
        except IndexError:
            buf = _alloc_aligned(size, blocksize=alignment)
        return buf

    def checkin_bytes(self, size, alignment, buf):
        self._buffers[(size, alignment)].insert(0, buf)


class ManagedBuffer:
    """
    Allocate `size` bytes from `pool`, and return them to the pool once we are GC'd
    """
    def __init__(self, pool, size, alignment):
        self.pool = pool
        self.buf = pool.checkout_bytes(size, alignment)
        self.size = size
        self.alignment = alignment

    def __del__(self):
        self.pool.checkin_bytes(self.size, self.alignment, self.buf)


T = TypeVar('T', bound=np.generic)


class InvalidMaskError(Exception):
    """
    The mask is not compatible, raised for example when the shape of
    the mask doesn't match the data.
    """
    pass


class ArrayWithMask(Generic[T]):
    """
    An opaque type representing an array together with a
    mask (for example, defining a region in the array
    where there are valid entries)

    This is meant for usage in :meth:`UDF.get_results`,
    and you can use the convenience method :meth:`UDF.with_mask`
    to instantiate it.
    """

    def __init__(self, arr: T, mask: Union[np.ndarray, bool]):
        """
        :meta private:
        """
        if isinstance(mask, bool):
            mask = np.array([mask])
        try:
            mask = np.broadcast_to(mask, arr.shape)
        except ValueError:
            raise InvalidMaskError(
                f"`arr` and `mask` must have compatible shapes "
                f"(arr.shape={arr.shape} vs mask.shape={mask.shape})"
            )
        if mask.dtype != np.dtype('bool'):
            raise InvalidMaskError(
                f"`mask` should have `dtype=bool` (have {mask.dtype})"
            )
        self._arr = arr
        self._mask = mask

    @property
    def mask(self) -> np.ndarray:
        return np.broadcast_to(self._mask, self._arr.shape)

    @property
    def arr(self) -> T:
        return self._arr


def get_inner_slice(arr: np.ndarray, axis: int = 0) -> tuple[slice, ...]:
    """
    Get a slice into `arr` across `axis` where the values on all other axes are non-zero.
    This will be the first contiguous slice, not necessarily the largest.

    :meta private:
    """
    ax_min = arr.shape[axis]
    ax_max = 0
    state = 0

    other_axes = tuple([
        i for i in range(arr.ndim)
        if i != axis
    ])
    non_zero = np.all(arr != 0, axis=other_axes)

    # find the first contiguous slice:
    for i in range(non_zero.shape[0]):
        if non_zero[i]:
            if state == 0:
                state = 1
                ax_min = i
                ax_max = i
            elif state == 1:
                ax_max = i
        else:
            if state == 1:
                break
    slice_ = tuple([
        slice(ax_min, ax_max + 1) if dim == axis else slice(None, None, None)
        for dim in range(arr.ndim)
    ])
    return slice_


@numba.njit(cache=True)
def get_bbox_2d(arr, eps=1e-8) -> tuple[int, ...]:
    """
    :meta private:
    """
    xmin = arr.shape[1]
    ymin = arr.shape[0]
    xmax = 0
    ymax = 0

    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            value = arr[y, x]
            if abs(value) < eps:
                continue
            # got a non-zero value, update indices
            if x < xmin:
                xmin = x
            if x > xmax:
                xmax = x
            if y < ymin:
                ymin = y
            if y > ymax:
                ymax = y
    return int(ymin), int(ymax), int(xmin), int(xmax)


def get_bbox(arr: np.ndarray) -> tuple[int, ...]:
    """
    :meta private:
    """
    if arr.ndim == 2:
        # 4x faster specialization in numba for 2D:
        return get_bbox_2d(arr)
    else:
        # from https://stackoverflow.com/a/31402351/540644
        N = arr.ndim
        out: list[int] = []
        for ax in itertools.combinations(reversed(range(N)), N - 1):
            nonzero = np.any(arr, axis=ax)
            out.extend(np.where(nonzero)[0][[0, -1]])
        return tuple(out)


def get_bbox_slice(arr: np.ndarray) -> tuple[slice, ...]:
    """
    :meta private:
    """
    bbox = get_bbox(arr)
    res = []
    for pair in batched(bbox, 2):
        res.append(slice(pair[0], pair[1] + 1, None))
    return tuple(res)


class BufferWrapper:
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

    .. versionchanged:: 0.6.0
        Add option to specify backend, for example CuPy

    Parameters
    ----------
    kind : "nav", "sig" or "single"
        The abstract shape of the buffer, corresponding either to the navigation
        or the signal dimensions of the dataset, or a single value.

    extra_shape : optional, tuple of int or a Shape object
        You can specify additional dimensions for your data. For example, if
        you want to store 2D coords, you would specify (2,) here.
        For a Shape object, sig_dims is discarded and the entire shape is used.

    dtype : string or numpy dtype
        The dtype of this buffer

    where : string or None
        :code:`None` means NumPy array, :code:`device` to use a back-end specified
        in :meth:`allocate`.

        .. versionadded:: 0.6.0

    use : "private", "result_only" or None
        If you specify :code:`"private"` here, the result will only be made available
        to internal functions, like :meth:`process_frame`, :meth:`merge` or
        :meth:`get_results`. It will not be available to the user of the UDF, which means
        you can use this to hide implementation details that are likely to change later.

        Specify :code:`"result_only"` here if the buffer is only used in :meth:`get_results`,
        this means we don't have to allocate and return it on the workers without actually
        needing it.

        :code:`None` means the buffer is used both as a final and intermediate result.

        .. versionadded:: 0.7.0

    """
    def __init__(
        self,
        kind: Literal['nav', 'sig', 'single'],
        extra_shape: tuple[int, ...] = (),
        dtype="float32",
        where: Union[None, Literal['device']] = None,
        use: Optional[Literal['private', 'result_only']] = None,
    ) -> None:
        self._extra_shape = tuple(extra_shape)
        self._kind = kind
        self._dtype = np.dtype(dtype)
        self._where = where
        self._data: Optional[np.ndarray] = None
        # set to True if the data coords are global ds coords
        self._data_coords_global = False
        self._shape = None
        self._ds_shape: Optional[Shape] = None
        self._ds_partitions = None
        self._roi = None
        self._roi_is_zero = None
        self._contiguous_cache = dict()
        self.use = use

    def set_roi(self, roi):
        """
        :meta private:
        """
        if roi is not None:
            roi = roi.reshape((-1,))
        self._roi = roi

    def add_partitions(self, partitions):
        """
        Add a list of dataset partitions to the buffer such that
        self.allocate() can make use of the structure

        :meta private:
        """
        self._ds_partitions = partitions

    def set_shape_partition(self, partition, roi=None):
        """
        :meta private:
        """
        self.set_roi(roi)
        roi_count = None
        if roi is not None:
            roi_part = self._roi[partition.slice.get(nav_only=True)]
            roi_count = count_nonzero(roi_part)
            assert roi_count <= partition.shape[0]
            assert roi_part.shape[0] == partition.shape[0]
        self._shape = self._shape_for_kind(self._kind, partition.shape, roi_count)
        self._update_roi_is_zero()

    def set_shape_ds(self, dataset_shape: Shape, roi=None):
        """
        :meta private:
        """
        self.set_roi(roi)
        roi_count = None
        if roi is not None:
            roi_count = count_nonzero(self._roi)
        self._shape = self._shape_for_kind(self._kind, dataset_shape.flatten_nav(), roi_count)
        self._update_roi_is_zero()
        self._ds_shape = dataset_shape

    @property
    def shape(self):
        # precondition: _shape_for_kind has been called
        return self._shape

    def _shape_for_kind(self, kind, orig_shape, roi_count=None):
        """
        :meta private:
        """
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
        array; unset values have NaN value for floating point types,
        False for boolean, 0 for integer types and structs,
        '' for string types and None for objects.

        .. versionchanged:: 0.7.0
            Better initialization values for dtypes other than floating point.
        """
        if self._contiguous_cache:
            raise RuntimeError("Cache is not empty, has to be flushed")
        if self._roi is None or self._kind != 'nav':
            return self._data.reshape(self._shape_for_kind(self._kind, self._ds_shape))
        shape = self._shape_for_kind(self._kind, self._ds_shape)
        if shape == self._data.shape:
            # preallocated and already wrapped
            return self._data
        # Integer types and "void" (structs and such)
        if self.dtype.kind in ('i', 'u', 'V'):
            fill = 0
        # Bytes and Unicode strings
        elif self.dtype.kind in ('S', 'U'):
            fill = ''
        else:
            # 'b' (boolean): False
            # 'f', 'c': NaN
            # 'm', 'M' (datetime, timedelta): NaT
            # 'O' (object): None
            fill = None
        flat_shape_with_extra = (prod(shape) // prod(self._extra_shape),) + self._extra_shape
        wrapper = np.full(flat_shape_with_extra, fill, dtype=self._dtype)
        wrapper[flat_nonzero(self._roi), ...] = self._data
        return wrapper.reshape(shape)

    @property
    def dtype(self):
        """
        Get the declared dtype of this buffer.

        .. versionadded:: 0.7.0
        """
        return self._dtype

    @property
    def raw_data(self) -> Optional[np.ndarray]:
        """
        Get the raw data underlying this buffer, which is flattened and
        may be even filtered to a ROI
        """
        return self._data

    def make_default_mask(
        self,
        valid_nav_mask: np.ndarray,
        dataset_shape: Shape,
        roi: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Create the default valid mask for this buffer kind, given the "upstream" `valid_nav_mask`.

        :meta private:
        """
        roi_count = None
        if roi is not None:
            roi_count = count_nonzero(roi)
        shape = self._shape_for_kind(self._kind, dataset_shape.flatten_nav(), roi_count)
        if self._kind == 'nav':
            # kind=nav by default uses the `valid_nav_mask`, which we need to
            # broadcast to take care of `extra_shape`:
            mask = np.zeros(shape, dtype=bool)
            extra_compat = tuple([
                1
                for _ in range(len(self._extra_shape))
            ])
            valid_nav_mask_compat = valid_nav_mask.reshape(valid_nav_mask.shape + extra_compat)
            mask[:] = valid_nav_mask_compat
        else:
            mask = np.ones(shape, dtype=bool)
        return mask

    @property
    def valid_mask(self) -> np.ndarray:
        """
        Returns a boolean array with a mask that indicates which parts of the
        buffer have valid contents.
        """
        if self._ds_shape is None:
            raise RuntimeError("`valid_mask` called without setting the dataset shape")

        if self._valid_mask is None:
            raise RuntimeError("`valid_mask` must be set")

        if self._kind == 'nav':
            # for `nav`, we need to do some gymnastics to un-flatten the nav
            # part of the shape, and possibly handle `roi`:
            full_shape = tuple(self._ds_shape.nav) + self._extra_shape
            if self._roi is not None:
                flat_shape = tuple(self._ds_shape.flatten_nav().nav) + self._extra_shape
                valid_mask = np.zeros(full_shape, dtype=bool)
                valid_mask.reshape(flat_shape)[self._roi] = self._valid_mask
                return valid_mask
            return self._valid_mask.reshape(full_shape)

        return self._valid_mask

    @valid_mask.setter
    def valid_mask(self, valid_nask: np.ndarray):
        """
        :meta private:
        """
        self._valid_mask = valid_nask

    @property
    def valid_slice_bounding(self) -> tuple[slice, ...]:
        """
        Returns a slice that bounds the valid elements in :attr:`data`.

        Note that this includes all valid elements, but also may contain
        invalid elements. If you instead want to get a slice that
        contains no invalid elements (which may cut off some valid elements),
        use :meth:`get_valid_slice_inner`.
        """
        return get_bbox_slice(self.valid_mask)

    def get_valid_slice_inner(self, axis: int = 0) -> tuple[slice, ...]:
        """
        Return a slice into :attr:`data` across `axis` that contains only
        elements marked as valid. This is the "inner" slice, meaning all
        elements selected by the slice are valid according to :attr:`valid_mask`.

        Parameters
        ----------

        axis
            The axis across which we should cut. Other axes are unrestricted in
            the returned slice. For example, if you have a :code:`(y, x)`
            result, specifying :code:`axis=0` will give you a slice like
            :code:`np.s_[y0:y1, :]` where for y in y0..y1, :code:`data[y, :]` is valid
            according to :attr:`valid_mask`.
        """
        return get_inner_slice(self.valid_mask, axis=axis)

    @property
    def masked_data(self) -> np.ma.MaskedArray:
        """
        Same as :attr:`data`, but masked to the valid data, as a numpy
        masked array.
        """
        mask = ~self.valid_mask
        return np.ma.array(self.data, mask=mask)

    @property
    def raw_masked_data(self) -> np.ma.MaskedArray:
        """
        Same as :attr:`raw_data`, but masked to the valid data, as a numpy
        masked array.
        """
        # NOTE: using `self._valid_mask` instead of `self.valid_mask` to get the
        # raw flat mask, not the one expanded to the full
        mask = ~self._valid_mask
        return np.ma.array(self.raw_data, mask=mask)

    @property
    def kind(self) -> Literal['nav', 'sig', 'single']:
        """
        Get the kind of this buffer.

        .. versionadded:: 0.5.0
        """
        return self._kind

    @property
    def extra_shape(self):
        """
        Get the :code:`extra_shape` of this buffer.

        .. versionadded:: 0.5.0
        """
        return self._extra_shape

    @property
    def where(self):
        """
        Get the place where this buffer is to be allocated.

        .. versionadded:: 0.6.0
        """
        return self._where

    def __array__(self):
        """
        returns the "wrapped"/reshaped array, see above
        """
        return self.data

    def allocate(self, lib=None):
        """
        Allocate a new buffer, in the shape previously set
        via one of the `set_shape_*` methods.

        .. versionchanged:: 0.6.0
            Support for allocating on device

        :meta private:
        """
        if self._shape is None:
            raise RuntimeError("cannot allocate: no shape set")
        if self._data is not None:
            raise RuntimeError("cannot allocate: data is already set")
        if self._where == 'device' and lib is not None:
            _z = lib.zeros
        else:
            _z = zeros_aligned
        self._data = _z(self._shape, dtype=self._dtype)

    def has_data(self):
        """
        :meta private:
        """
        return self._data is not None

    @property
    def roi_is_zero(self):
        """
        :meta private:
        """
        return self._roi_is_zero

    def _update_roi_is_zero(self):
        """
        :meta private:
        """
        self._roi_is_zero = prod(self._shape) == 0

    def _slice_for_partition(self, partition):
        """
        Get a Slice into self._data for `partition`, taking the current ROI into account.

        Because _data is "compressed" if a ROI is set, we can't directly index and must
        calculate a new slice from the ROI.

        :meta private:
        """
        if self._roi is None:
            return partition.slice
        return partition.slice.adjust_for_roi(self._roi)

    def get_view_for_dataset(self, dataset):
        """
        :meta private:
        """
        if self._contiguous_cache:
            raise RuntimeError("Cache is not empty, has to be flushed")
        return self._data

    def _get_slice(self, slice: Slice):
        """
        :meta private:
        """
        real_slice = slice._get()
        shape = tuple(slice.shape) + self.extra_shape
        return self._get_slice_direct(real_slice, shape)

    def _get_slice_direct(self, real_slice: slice, shape):
        """
        :meta private:
        """
        result = self._data[real_slice]
        # Defend against #1026 (internal bugs), allow deactivating in
        # optimized builds for performance
        assert result.shape == shape
        return result

    def get_view_for_partition(self, partition):
        """
        get a view for a single partition in a whole-result-sized buffer

        :meta private:
        """
        if self._contiguous_cache:
            raise RuntimeError("Cache is not empty, has to be flushed")
        if self._kind == "nav":
            return self._get_slice(self._slice_for_partition(partition).nav)
        elif self._kind == "sig":
            return self._get_slice(partition.slice.sig)
        elif self._kind == "single":
            return self._data

    def get_view_for_frame(self, partition, tile, frame_idx):
        """
        get a view for a single frame in a partition- or dataset-sized buffer
        (partition-sized here means the reduced result for a whole partition,
        not the partition itself!)

        :meta private:
        """
        if partition.shape.dims != partition.shape.sig.dims + 1:
            raise RuntimeError("partition shape should be flat, is %s" % partition.shape)
        if self._contiguous_cache:
            raise RuntimeError("Cache is not empty, has to be flushed")
        if self._kind == "sig":
            return self._get_slice(tile.tile_slice.sig)
        elif self._kind == "nav":
            partition_slice = self._slice_for_partition(partition)
            if self._data_coords_global:
                offset = 0
            else:
                offset = partition_slice.origin[0]
            # Defense against #1026: if it is an int, there is no empty or out
            # of range slice, but an index error
            result_idx = (int(tile.tile_slice.origin[0] + frame_idx - offset),)
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

        :meta private:
        """
        if self._contiguous_cache:
            raise RuntimeError("Cache is not empty, has to be flushed")
        if self.roi_is_zero:
            raise ValueError("cannot get view for tile with zero ROI")
        if self._kind == "sig":
            return self._get_slice(tile.tile_slice.sig)
        elif self._kind == "nav":
            partition_slice = self._slice_for_partition(partition)
            tile_slice = tile.tile_slice
            if self._data_coords_global:
                offset = 0
            else:
                offset = partition_slice.origin[0]
            result_start = tile_slice.origin[0] - offset
            result_stop = result_start + tile_slice.shape[0]
            # Defend against #1026 (internal bugs), allow deactivating in
            # optimized builds for performance
            assert result_start < len(self._data)
            assert result_stop <= len(self._data)
            return self._data[result_start:result_stop]
        elif self._kind == "single":
            return self._data

    def get_contiguous_view_for_tile(self, partition, tile):
        '''
        Make a cached contiguous copy of the view for a single tile
        if necessary.

        Currently this is only necessary for :code:`kind="sig"` buffers.
        Use :meth:`flush` to write back the cache.

        Boundary condition: :code:`tile.tile_slice.get(sig_only=True)`
        does not overlap for different tiles while the cache is active,
        i.e. the tiles follow LiberTEM slicing for
        :meth:`libertem.udf.base.UDFTileMixing.process_tile()`.

        .. versionadded:: 0.5.0

        Returns
        -------

        view : np.ndarray
            View into data or contiguous copy if necessary

        :meta private:
        '''
        if self._kind == "sig":
            key = tile.tile_slice._discard_nav_key()
            if key in self._contiguous_cache:
                view = self._contiguous_cache[key]
            else:
                real_slice, expected_shape = self._slice_from_key(key, self.extra_shape)
                view = self._get_slice_direct(real_slice, expected_shape)
                # See if the signal dimension can be flattened
                # https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html
                if not view.flags.c_contiguous:
                    view = view.copy()
                    self._contiguous_cache[key] = view
            return view
        else:
            return self.get_view_for_tile(partition, tile)

    @staticmethod
    @functools.lru_cache(maxsize=512)
    def _slice_from_key(key, extra_shape):
        """
        :meta private:
        """
        origin, shape, sig_dims = key
        expected_shape = shape[-sig_dims:]
        return (
            tuple(slice(o, (o + s)) for (o, s) in zip(origin[-sig_dims:], expected_shape)),
            expected_shape + extra_shape
        )

    def flush(self, debug=False):
        '''
        Write back any cached contiguous copies

        .. versionadded:: 0.5.0

        :meta private:
        '''
        if self._kind == "sig":
            for key, view in self._contiguous_cache.items():
                sl, _ = self._slice_from_key(key, tuple())
                _, _, sig_dims = key
                sl = sl[-sig_dims:]
                self._data[sl] = view
                # FIXME disjoint() for tuple keys
                # if debug and not disjoint(key, self._contiguous_cache.keys()):
                #     raise RuntimeError(
                #         "`key` %r should be disjoint with existing keys" % key
                #     )
            self._contiguous_cache = dict()
        else:
            if self._contiguous_cache:
                raise RuntimeError(
                    f"Contiguous cache not implemented for kind {self._kind}."
                )

    def export(self):
        '''
        Convert device array to NumPy array for pickling and merging

        :meta private:
        '''
        self._data = to_numpy(self._data)

    def __repr__(self):
        return "<BufferWrapper kind={} dtype={} extra_shape={}>".format(
            self._kind, self._dtype, self._extra_shape
        )

    def replace_array(self, data: "nt.ArrayLike") -> None:
        """
        Set the data backing the BufferWrapper even if the BufferWrapper
        already has self._data allocated

        data should be any array-like object

        Will perform checking for shape and dtype.

        :meta private:
        """
        if self._data is None:
            shape = self._shape
            dtype = self._dtype
        else:
            shape = self._data.shape
            dtype = self._data.dtype
        if data.dtype != dtype:
            raise ValueError(f"dtype mismatch: buffer is {dtype}, data is {data.dtype}.")
        if data.shape != shape:
            raise ValueError(f"Shape mismatch: buffer is {shape}, data is {data.shape}.")
        self._contiguous_cache = dict()
        self._data = data

    def result_buffer_type(self):
        """
        Define the type of Buffer used to return final UDF results

        More specialised buffers can override this

        :meta private:
        """
        return PreallocBufferWrapper


class PlaceholderBufferWrapper(BufferWrapper):
    """
    A declaration-only version of :code:`BufferWrapper` that doesn't
    actually allocate a buffer. Meant as a placeholder for results
    that are only materialized in :code:`UDF.get_results`.
    """
    def allocate(self, lib=None):
        return None

    def has_data(self):
        return False

    def export(self):
        return None

    def get_view_for_partition(self, partition):
        return None

    def get_view_for_frame(self, partition, tile, frame_idx):
        return None

    def get_view_for_tile(self, partition, tile):
        return None

    def get_contiguous_view_for_tile(self, partition, tile):
        return None

    @property
    def data(self):
        raise ValueError(
            "this BufferWrapper doesn't have a value associated with it"
        )

    @property
    def raw_data(self):
        raise ValueError(
            "this BufferWrapper doesn't have a value associated with it"
        )


class PreallocBufferWrapper(BufferWrapper):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data


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
        assert prod(new_data.shape) > 0
        assert not buf._data_coords_global
        return buf

    def get_view_for_dataset(self, dataset):
        return self._data[self._roi]

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
        shape = (-1,)
        if extra and extra != (1,):
            shape = shape + extra
        self._data = buf.reshape(shape)
        self._data_coords_global = is_global

    def __repr__(self):
        return "<AuxBufferWrapper kind={} dtype={} extra_shape={}>".format(
            self._kind, self._dtype, self._extra_shape
        )
