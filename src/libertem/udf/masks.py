import numpy as np

from libertem.udf import UDF
from libertem.common.container import MaskContainer
from libertem.common.numba import rmatmul


class ApplyMasksUDF(UDF):
    '''
    Apply masks to signals/frames in the dataset.

    .. versionadded:: 0.4.0
    '''
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, mask_count=None,
                mask_dtype=None, preferred_dtype=None, backends=None):
        '''
        Parameters
        ----------

        mask_factories : Union[Callable[[], array_like], Iterable[Callable[[], array_like]]]
            Function or list of functions that take no arguments and create masks. The returned
            masks can be
            numpy arrays, scipy.sparse or sparse https://sparse.pydata.org/ matrices. The mask
            factories should not reference large objects because they can create significant
            overheads when they are pickled and unpickled. Each factory function should, when
            called, return a numpy array with the same shape as frames in the dataset
            (so dataset.shape.sig).
        use_torch : bool, optional
            Use pytorch back-end if available. Default True
        use_sparse : Union[None, False, True, 'scipy.sparse', 'scipy.sparse.csc', \
                'sparse.pydata'], optional
            Which sparse back-end to use.
            * None (default): Use sparse matrix multiplication if all factory functions return a \
                sparse mask, otherwise convert all masks to dense matrices and use dense matrix \
                multiplication
            * True: Convert all masks to sparse matrices and use default sparse back-end.
            * False: Convert all masks to dense matrices.
            * 'scipy.sparse': Use scipy.sparse.csr_matrix (default sparse)
            * 'scipy.sparse.csc': Use scipy.sparse.csc_matrix
            * 'sparse.pydata': Use sparse.pydata COO matrix
        mask_count : int, optional
            Specify the number of masks if a single factory function is used so that the
            number of masks can be determined without calling the factory function.
        mask_dtype : numpy.dtype, optional
            Specify the dtype of the masks so that mask dtype
            can be determined without calling the mask factory functions. This can be used to
            override the mask dtype in the result dtype determination. As an example, setting
            this to np.float32 means that masks of type float64 will not switch the calculation
            and result dtype to float64 or complex128.
        preferred_dtype : numpy.dtype, optional
            Let :meth:`get_preferred_input_dtype` return the specified type instead of the
            default `float32`. This can perform the calculation with integer types if both input
            data and mask data are compatible with this.
        backends : Iterable containing strings "numpy" and/or "cupy", or None
            Control which back-ends are used. Default is numpy and cupy
        '''
        if use_sparse is True:
            use_sparse = 'scipy.sparse'

        if backends is None:
            backends = ('numpy', 'cupy')

        self._mask_container = None

        super().__init__(
            mask_factories=mask_factories,
            use_torch=use_torch,
            use_sparse=use_sparse,
            mask_count=mask_count,
            mask_dtype=mask_dtype,
            preferred_dtype=preferred_dtype,
            backends=backends
        )

    def get_preferred_input_dtype(self):
        ''
        if self.params.preferred_dtype is None:
            return super().get_preferred_input_dtype()
        else:
            return self.params.preferred_dtype

    def get_mask_dtype(self):
        if self.params.mask_dtype is None:
            return self.masks.dtype
        else:
            return self.params.mask_dtype

    def get_mask_count(self):
        if self.params.mask_count is None:
            return len(self.masks)
        else:
            return self.params.mask_count

    @property
    def masks(self):
        if self._mask_container is None:
            self._mask_container = self._make_mask_container()
        return self._mask_container

    @property
    def backend(self):
        if self.meta.device_class == 'cuda':
            backend = 'cupy'
        else:
            backend = 'numpy'
        return backend

    def _make_mask_container(self):
        p = self.params
        return MaskContainer(
            p.mask_factories, dtype=p.mask_dtype, use_sparse=p.use_sparse, count=p.mask_count,
            backend=self.backend
        )

    def get_task_data(self):
        ''
        try:
            import torch
        except ImportError:
            torch = None
        m = self.meta
        use_torch = self.params.use_torch
        if (torch is None or m.input_dtype.kind != 'f' or m.input_dtype != self.get_mask_dtype()
                or self.meta.device_class != 'cpu' or self.masks.use_sparse):
            use_torch = False
        return {
            'use_torch': use_torch,
            'masks': self.masks,
        }

    def get_result_buffers(self):
        ''
        dtype = np.result_type(self.meta.input_dtype, self.get_mask_dtype())
        count = self.get_mask_count()
        return {
            'intensity': self.buffer(
                kind='nav', extra_shape=(count, ), dtype=dtype, where='device'
            )
        }

    def get_backends(self):
        return self.params.backends

    def process_tile(self, tile):
        ''
        flat_data = tile.reshape((tile.shape[0], -1))
        if self.task_data.use_torch:
            import torch
            masks = self.task_data.masks.get(self.meta.slice, transpose=True)
            # CuPy back-end disables torch in get_task_data
            # FIXME use GPU torch with CuPy array?
            result = torch.mm(
                torch.from_numpy(flat_data),
                torch.from_numpy(masks),
            ).numpy()
        # Required due to https://github.com/cupy/cupy/issues/4072
        elif self.backend == 'cupy' and self.task_data.masks.use_sparse:
            masks = self.task_data.masks.get(self.meta.slice, transpose=False)
            result = masks.dot(flat_data.T).T
        # Required due to https://github.com/scipy/scipy/issues/13211
        elif (self.backend == 'numpy'
              and self.task_data.masks.use_sparse
              and 'scipy.sparse' in self.task_data.masks.use_sparse):
            masks = self.task_data.masks.get(self.meta.slice, transpose=True)
            result = rmatmul(flat_data, masks)
        else:
            masks = self.task_data.masks.get(self.meta.slice, transpose=True)
            result = flat_data @ masks
        # '+' is the correct merge for dot product
        self.results.intensity[:] += result
