import logging

try:
    import torch
except ImportError:
    torch = None
import sparse
import scipy.sparse
import numpy as np

from libertem.io.dataset.base import DataTile, Partition
from libertem.udf.base import Task
from .base import BaseJob, ResultTile
from libertem.common import Slice
from libertem.common.buffers import zeros_aligned
from libertem.common.container import MaskContainer


log = logging.getLogger(__name__)


class JobMaskContainer(MaskContainer):
    def get(self, key, dtype=None, sparse_backend=None, transpose=True):
        if isinstance(key, Partition):
            slice_ = key.slice
        elif isinstance(key, DataTile):
            slice_ = key.tile_slice
        elif isinstance(key, Slice):
            slice_ = key
        else:
            raise TypeError(
                "JobMaskContainer.get() can only be called with "
                "DataTile/Slice/Partition instances"
            )
        return super().get(
            key=slice_, dtype=dtype, sparse_backend=sparse_backend, transpose=transpose
        )


class ApplyMasksJob(BaseJob):
    """
    Apply masks to signals/frames in the dataset.

    Running a :class:`ApplyMaskJob` with
    :meth:`~libertem.api.Context.run` returns a :class:`numpy.ndarray`
    with shape (n_masks, prod(ds.shape.nav)).

    .. deprecated:: 0.4.0
        Use :class:`libertem.udf.masks.ApplyMasksUDF` instead. See also :ref:`job deprecation`
    """
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, mask_count=None,
                mask_dtype=None, dtype=None, *args, **kwargs):
        '''
        use_sparse can be None, True, 'scipy.sparse', 'scipy.sparse.csc' or 'sparse.pydata'
        '''
        super().__init__(*args, **kwargs)
        # Choose default back-end
        # If None, decide in the mask container
        if use_sparse is True:
            use_sparse = 'scipy.sparse'
        self.masks = JobMaskContainer(mask_factories, dtype=mask_dtype,
            use_sparse=use_sparse, count=mask_count)

        self.dtype = dtype
        self.use_torch = use_torch

    def get_tasks(self):
        for idx, partition in enumerate(self.dataset.get_partitions()):
            yield ApplyMasksTask(
                partition=partition,
                masks=self.masks,
                use_torch=self.use_torch,
                idx=idx,
                dtype=self.get_result_dtype()
            )

    def get_result_shape(self):
        return (len(self.masks),) + tuple(self.dataset.shape.flatten_nav().nav)

    def get_result_dtype(self):
        if self.dtype is None:
            return np.result_type(np.float32, self.dataset.dtype, self.masks.dtype)
        else:
            return self.dtype


class ApplyMasksTask(Task):
    def __init__(self, masks, use_torch, dtype, *args, **kwargs):
        """
        Parameters
        ----------
        partition : libertem.dataset.base.Partition instance
            the partition to work on
        masks : JobMaskContainer
            the masks to apply to the partition
        use_torch :
            Setting to False disables torch. Setting to True doesn't enforce
            torch. Torch will be disabled if it is not installed
            or if the dtypes are unsuitable. It works only for float32 or float64,
            and only if both mask and data have the same dtype.
        dtype :
            dtype to use for the calculation and the result.
        """
        super().__init__(*args, **kwargs)
        self.masks = masks
        self.use_torch = use_torch
        self.dtype = np.dtype(dtype)
        self.read_dtype = self._input_dtype(self.partition.dtype)
        self.mask_dtype = self._input_dtype(self.masks.dtype)
        if torch is None or self.dtype.kind != 'f' or self.read_dtype != self.mask_dtype:
            self.use_torch = False

    def _input_dtype(self, dtype):
        '''
        Determine which dtype to request for masks or input data based on their native
        dtype and self.dtype.

        A dot product with floats is significantly faster than doing the same processing with
        integer data types, because Numpy uses its internal implementation of the dot product
        for integers, while it uses optimized libraries like OpenBLAS for floats. Furthermore,
        floats allow using torch.

        For that reason, we use floats by default. If floats are used, we request native integer
        mask data and native integer input data to be converted to floats already at the source.
        That helps to avoid an additional conversion step. As an example, the mask container can
        cache a float version with get_mask_for_slice(), and the K2IS reader can convert its 12 bit
        packed uints to floats directly.

        In case a conversion is requested, we decide if float32 or float64 is best suited based on
        the itemsize of the source data. In particular, float64 will be used for int64 data.

        In case the native dtype is not integer or in case the processing is not done with floating
        point numbers, return the native dtype.

        FIXME It should be tested in more detail which dtype combinations (32 bit vs 64 bit, complex
        vs real) are ideal for the dot product and what impact the conversion has on overall
        performance. In particular, the impact of size vs conversion overhead is not trivial to
        predict and might depend on the CPU type and load.
        The decision logic in this function should be adapted accordingly.
        '''
        dtype = np.dtype(dtype)
        # Convert integer data to floats if we want to produce floats or complex
        if dtype.kind in ('u', 'i', 'b') and self.dtype.kind in ('f', 'c'):
            # We have int64 or similar, use float64 to fit as much information as possible.
            if dtype.itemsize > 4:
                dtype = np.float64
            else:
                dtype = np.float32
        return dtype

    def reshaped_data(self, data, dest_slice):
        """
        Reshapes the result from the flattened and interleaved version to a shape
        that fits the result array (masks, ...nav_dims)
        """

        num_masks = data.shape[1]

        deinterleaved = np.stack(
            [data.ravel()[idx::num_masks]
             for idx in range(num_masks)],
            axis=0,
        )
        return deinterleaved.reshape((num_masks,) + tuple(dest_slice.shape.nav))

    def __call__(self):
        num_masks = len(self.masks)
        part = zeros_aligned((num_masks,) + tuple(self.partition.shape.nav), dtype=self.dtype)
        for data_tile in self.partition.get_tiles(mmap=True, dest_dtype=self.read_dtype):
            flat_data = data_tile.flat_data
            masks = self.masks.get(data_tile, self.mask_dtype)
            if isinstance(masks, sparse.SparseArray):
                result = sparse.dot(flat_data, masks)
            elif scipy.sparse.issparse(masks):
                # This is scipy.sparse using the old matrix interface
                # where "*" is the dot product
                result = flat_data * masks
            elif self.use_torch:
                result = torch.mm(
                    torch.from_numpy(flat_data),
                    torch.from_numpy(masks),
                ).numpy()
            else:
                result = flat_data.dot(masks)
            dest_slice = data_tile.tile_slice.shift(self.partition.slice)
            reshaped = self.reshaped_data(data=result, dest_slice=dest_slice)
            # Ellipsis to match the "number of masks" part of the result
            part[(...,) + dest_slice.get(nav_only=True)] += reshaped
        return [
            MaskResultTile(
                data=part,
                dest_slice=self.partition.slice.get(nav_only=True),
            )
        ]


class MaskResultTile(ResultTile):
    def __init__(self, data, dest_slice):
        self.data = data
        self.dest_slice = dest_slice

    def __repr__(self):
        return "<ResultTile for slice=%r>" % self.dest_slice

    @property
    def dtype(self):
        return self.data.dtype

    def reduce_into_result(self, result):
        result[(Ellipsis,) + self.dest_slice] += self.data
        return result
