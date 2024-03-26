import dask.array as da
import numpy as np

from ...common.math import prod
from ...common.slice import Slice
from ...common.buffers import BufferWrapper
from .dask_inplace import DaskInplaceWrapper


class DaskBufferWrapper(BufferWrapper):
    @classmethod
    def from_buffer(cls, buffer: BufferWrapper):
        """
        Convert an existing BufferWrapper into a DaskBufferWrapper
        without copying its data, only its parameters
        """
        if buffer.has_data():
            RuntimeWarning('Buffer is already allocated, not copying current contents or view')
        result = cls(kind=buffer.kind,
                   extra_shape=buffer.extra_shape,
                   dtype=buffer.dtype,
                   where=buffer.where,
                   use=buffer.use)
        result._shape = buffer._shape
        result._ds_shape = buffer._ds_shape
        result._ds_partitions = buffer._ds_partitions
        result._roi = buffer._roi
        result._roi_is_zero = buffer._roi_is_zero
        return result

    @property
    def _extra_chunking(self):
        return tuple((s,) for s in self._extra_shape)

    def allocate(self, lib=None):
        """
        Allocate a buffer as a dask array

        nav buffer chunking is made identical to the partition structure
        in the flat nav dimension via the attribute self._ds_partitions
        sig buffers are currently unchunked as there is no mechanism
        to have partitions which cover part of the signal dimension

        extra_shape dimensions are currently unchunked
        """
        # self._ds_partitions set in UDFData.allocate_for_full()
        nav_chunking = (tuple(self._slice_for_partition(p).shape[0]
                              for p in self._ds_partitions),)

        # At this time there is no interest in chunking the sig dimension
        # This could be modified in the future if partitions are more flexible
        sig_shape = self._ds_shape.sig
        sig_chunking = (-1,) * len(sig_shape) + self._extra_chunking

        if self._kind == 'nav':
            flat_nav_shape = self._shape[0]
            if self._shape == (flat_nav_shape,) + self._extra_shape:
                _buf_chunking = nav_chunking + self._extra_chunking
            elif self._shape[0] in nav_chunking:
                # This branch should never be taken if we are only allocating
                # with dask on the main node and not inside UDFTasks
                _buf_chunking = self._shape
            else:
                raise RuntimeError('Unrecognized buffer size relative to ds/chunk globals')
        elif self._kind == 'sig':
            _buf_chunking = sig_chunking
        elif self._kind == 'single':
            _buf_chunking = self._shape
        else:
            raise NotImplementedError('Unrecognized buffer kind')
        self._data = da.zeros(self._shape, dtype=self._dtype, chunks=_buf_chunking)

    @property
    def data(self):
        """
        Get the buffer contents in shape that corresponds to the
        original dataset shape, using a lazy Dask array.

        Copied largely from BufferWrapper with modifications to ensure
        Dask arrays are correctly unpacked into the result array.

        #TODO consider if this needs to be cached to avoid creating
        multiple copies in the task graph ?

        If a ROI is set, embed the result into a new
        array; unset values have NaN value for floating point types,
        False for boolean, 0 for integer types and structs,
        '' for string types and None for objects.
        """
        if isinstance(self._data, DaskInplaceWrapper):
            self._data = self._data.data
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

        flat_chunking = tuple(p.slice.shape[0] for p in self._ds_partitions)
        flat_chunking = (flat_chunking,) + self._extra_chunking
        flat_shape = (prod(self._ds_shape.nav),) + self._extra_shape
        flat_wrapper = da.full(flat_shape, fill, dtype=self._dtype, chunks=flat_chunking)
        flat_wrapper[self._roi, ...] = self._data
        wrapper = flat_wrapper.reshape(self._ds_shape.nav + self._extra_shape)
        return wrapper

    def _get_slice(self, slice: Slice):
        """
        Get a view of the buffer which enable inplace assignment
        to the underlying Dask array, for simple inplace assignments

        This is not robust to doing complex operations on the
        buffer such as changing its shape
        """
        real_slice = slice.get()
        inplace_wrapper = DaskInplaceWrapper(self._data)
        inplace_wrapper.set_slice(real_slice)
        return inplace_wrapper

    def export(self):
        """
        No requirement to move dask arrays between devices
        This functionality is embedded in the .compute() call
        """
        self._data = self._data

    def __repr__(self):
        return (f"<{self.__class__.__name__} kind={self._kind} "
                f"extra_shape={self._extra_shape} backing={self._data}>")

    def result_buffer_type(self):
        return self.new_as_prealloc

    def new_as_prealloc(self, *args, **kwargs):
        buffer = DaskPreallocBufferWrapper(*args, **kwargs)
        buffer.add_partitions(self._ds_partitions)
        return buffer


class DaskPreallocBufferWrapper(DaskBufferWrapper):
    def __init__(self, data, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data = data


class DaskResultBufferWrapper(DaskPreallocBufferWrapper):
    """
    A DaskBufferWrapper that is meant to be used for results.

    In the LiberTEM internals, we need to make sure everything is still lazy/delayed.

    Once we leave LiberTEM internals, the external interface is used, where
    both :meth:`data` and :meth:`raw_data` eagerly return numpy array-like
    objects.
    """
    @classmethod
    def from_buffer_wrapper(cls, buffer: BufferWrapper):
        assert buffer.has_data()
        result = cls(
            data=buffer._data,
            kind=buffer.kind,
            extra_shape=buffer.extra_shape,
            dtype=buffer.dtype,
            where=buffer.where,
            use=buffer.use
        )
        result._shape = buffer._shape
        result._ds_shape = buffer._ds_shape
        result._ds_partitions = buffer._ds_partitions
        result._roi = buffer._roi
        result._roi_is_zero = buffer._roi_is_zero
        result._valid_mask = buffer._valid_mask
        return result

    def __array__(self):
        return super().__array__()

    @property
    def raw_data(self):
        """
        Eager version of :meth:`delayed_raw_data`
        """
        return np.array(super().raw_data)

    @property
    def delayed_raw_data(self):
        return super().raw_data

    @property
    def data(self):
        """
        Eager version of :meth:`delayed_data`
        """
        return np.array(super().data)

    @property
    def delayed_data(self):
        return super().data
