import dask.array as da

from ...common.slice import Slice
from ...common.buffers import BufferWrapper
from .dask_inplace import DaskInplaceBufferWrapper


class DaskBufferWrapper(BufferWrapper):
    @classmethod
    def from_buffer(cls, buffer: BufferWrapper):
        """
        Convert an existing BufferWrapper into a DaskBufferWrapper
        without copying its data, only its parameters
        """
        if buffer.has_data():
            RuntimeWarning('Buffer is already allocated, not copying current contents or view')
        return cls(buffer.kind,
                   extra_shape=buffer.extra_shape,
                   dtype=buffer.dtype,
                   where=buffer.where,
                   use=buffer.use)

    def allocate(self, lib=None):
        """
        Allocate a buffer as a dask array

        nav buffer chunking is made identical to the partition structure
        in the flat nav dimension via the attribute self._ds_partitions
        sig buffers are currently unchunked as there is no mechanism
        to have partitions which cover part of the signal dimension
        """
        # self._ds_partitions set in UDFData.allocate_for_full()
        nav_chunking = tuple(self._slice_for_partition(p).shape[0]
                             for p in self._ds_partitions)
        # At this time there is no interest in chunking the sig dimension
        # This could be modified in the future if partitions are more flexible
        sig_shape = self._ds_shape.sig
        sig_chunking = (-1,) * len(sig_shape)

        if self._kind == 'nav':
            flat_nav_shape = self._shape[0]
            if self._shape == (flat_nav_shape,):
                _buf_chunking = nav_chunking
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
        if isinstance(self._data, DaskInplaceBufferWrapper):
            self._data = self._data.unwrap()
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

        if self._kind == 'nav':
            flat_chunking = tuple(p.slice.shape[0] for p in self._ds_partitions)
            flat_shape = self._ds_shape.flatten_nav()[0]
            flat_wrapper = da.full(flat_shape, fill, dtype=self._dtype, chunks=flat_chunking)
            flat_wrapper[self._roi] = self._data
            wrapper = flat_wrapper.reshape(self._ds_shape.nav)
        else:
            chunking = ((-1,),) * len(shape)
            wrapper = da.full(shape, fill, dtype=self._dtype, chunks=chunking)
            wrapper[self._roi.reshape(self._ds_shape.nav)] = self._data
        return wrapper

    def _get_slice(self, slice: Slice):
        """
        Get a view of the buffer which enable inplace assignment
        to the underlying Dask array, for simple inplace assignments

        This is not robust to doing complex operations on the
        buffer such as changing its shape
        """
        real_slice = slice.get()
        inplace_wrapper = DaskInplaceBufferWrapper(self._data)
        inplace_wrapper.set_slice(real_slice)
        return inplace_wrapper

    def export(self):
        """
        No requirement to move dask arrays between devices
        This functionality is embedded in the .compute() call
        """
        self._data = self._data

    def __repr__(self):
        return (f"<DaskBufferWrapper kind={self._kind}"
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
