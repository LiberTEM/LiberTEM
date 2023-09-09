from typing import Optional, Tuple
from typing_extensions import Literal
from libertem.common.math import prod
import numpy as np

from libertem.common.udf import UDFMethodEnum
from libertem.udf import UDF, UDFMeta
from libertem.common.container import MaskContainer
from libertem.common.numba import rmatmul


class ApplyMasksEngine:
    def __init__(self, masks: MaskContainer, meta: UDFMeta, use_torch: bool):
        self.masks = masks
        self.meta = meta

        try:
            import torch
        except ImportError:
            torch = None

        torch_incompatible = (
            torch is None
            or self.meta.input_dtype.kind != 'f'
            or self.meta.input_dtype != self.masks.dtype
            or self.meta.device_class != 'cpu'
            or self.meta.array_backend != UDF.BACKEND_NUMPY
            or self.masks.use_sparse
        )

        self.needs_transpose = True
        if use_torch and (not torch_incompatible):
            self.process_flat = self._process_flat_torch
        elif (
            self.meta.array_backend == UDF.BACKEND_NUMPY
            and self.masks.use_sparse
            and 'scipy.sparse' in self.masks.use_sparse
        ):
            # Due to https://github.com/scipy/scipy/issues/13211
            self.process_flat = self._process_flat_spsp
        elif (
            self.meta.array_backend in (
                UDF.BACKEND_SCIPY_COO,
                UDF.BACKEND_SCIPY_CSR,
                UDF.BACKEND_SCIPY_CSC
            ) and self.masks.use_sparse
            and 'sparse.pydata' in self.masks.use_sparse
        ):
            self.process_flat = self._process_flat_sparsepyd
            self.needs_transpose = False
        else:
            self.process_flat = self._process_flat_standard

    def _get_masks(self):
        return self.masks.get_for_sig_slice(
            self.meta.sig_slice, transpose=self.needs_transpose
        )

    def _process_flat_torch(self, flat_tile, masks):
        import torch
        if masks is None:
            masks = self._get_masks()
        # CuPy back-end disables torch in get_task_data
        # FIXME use GPU torch with CuPy array?
        return torch.mm(
            torch.from_numpy(flat_tile),
            torch.from_numpy(masks),
        ).numpy()

    def _process_flat_spsp(self, flat_tile, masks):
        if masks is None:
            masks = self._get_masks()
        return rmatmul(flat_tile, masks)

    def _process_flat_sparsepyd(self, flat_tile, masks):
        if masks is None:
            masks = self._get_masks()
        # Make sure the sparse.pydata mask comes first
        # to choose the right multiplication method
        return (masks @ flat_tile.T).T

    def _process_flat_standard(self, flat_tile, masks):
        if masks is None:
            masks = self._get_masks()
        return flat_tile @ masks

    def process_tile(self, tile, masks: Optional[np.ndarray] = None):
        flat_shape = (tile.shape[0], prod(tile.shape[1:]))
        # Avoid reshape since older versions of scipy.sparse don't support it
        flat_data = tile.reshape(flat_shape) if tile.shape != flat_shape else tile
        return self.process_flat(flat_data, masks)
    
    def process_frame_shifted(self, frame, shifts: Tuple[int, ...]):
        masks = self._get_masks()
        mask_transposed = self.needs_transpose
        shifted_slice = self.meta.sig_slice.shift_by(shifts)
        left, right = self.meta.sig_slice.intersection_pair(shifted_slice)
        data = left.get(frame)
        if mask_transposed:
            mask_slice = right.get() + (slice(None), )
            num_masks = masks.shape[-1]
            masks = masks.reshape((*frame.shape[::-1], masks.shape[-1]))
            masks = masks[mask_slice].reshape((-1, num_masks))
        else:
            mask_slice = (slice(None), ) + right.get()
            num_masks = masks.shape[0]
            masks = masks.reshape((masks.shape[0], *frame.shape))
            masks = masks[mask_slice].reshape((num_masks, -1))

        # FIXME This reshape is incompatible with older versions of scipy.sparse
        flat_data = data.reshape(1, prod(data.shape))
        return self.process_flat(flat_data, masks)[0]


class ApplyMasksUDF(UDF):
    '''
    Apply masks to signals/frames in the dataset. This can not only be used to integrate
    over regions with a binary mask - the integration can be weighted by using
    float or complex valued masks.

    The result will be returned in a single sig-shaped buffer called intensity.
    Its shape will be :code:`(*nav_shape, len(masks))`.

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

    Examples
    --------

    >>> dataset.shape
    (16, 16, 32, 32)
    >>> def my_masks():
    ...     return [np.ones((32, 32)), np.zeros((32, 32))]
    >>> udf = ApplyMasksUDF(mask_factories=my_masks)
    >>> res = ctx.run_udf(dataset=dataset, udf=udf)['intensity']
    >>> res.data.shape
    (16, 16, 2)
    >>> np.allclose(res.data[..., 1], 0)  # same order as in the mask factory
    True

    Mask factories can also return all masks as a single array, stacked on the first axis:

    >>> def my_masks_2():
    ...     masks = np.zeros((2, 32, 32))
    ...     masks[1, ...] = 1
    ...     return masks
    >>> udf = ApplyMasksUDF(mask_factories=my_masks_2)
    >>> res_2 = ctx.run_udf(dataset=dataset, udf=udf)['intensity']
    >>> np.allclose(res_2.data, res.data)
    True

    .. versionadded:: 0.4.0
    '''
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, mask_count=None,
                mask_dtype=None, preferred_dtype=None, backends=None, shifts=None, **kwargs):

        if shifts is not None:
            if use_sparse is True:
                use_sparse = 'sparse.pydata'
            elif isinstance(use_sparse, str) and use_sparse.startswith('scipy.sparse'):
                raise ValueError(
                    f'Sparse backend {use_sparse} not supported for shifts, use sparse.pydata instead.'
                )
        elif use_sparse is True:
            use_sparse = 'scipy.sparse'

        if backends is None:
            backends = self.BACKEND_ALL

        self._mask_container = None

        super().__init__(
            mask_factories=mask_factories,
            use_torch=use_torch,
            use_sparse=use_sparse,
            mask_count=mask_count,
            mask_dtype=mask_dtype,
            preferred_dtype=preferred_dtype,
            backends=backends,
            shifts=shifts,
            **kwargs
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

    def _make_mask_container(self):
        p = self.params
        if self.meta.array_backend in self.CUPY_BACKENDS:
            backend = self.BACKEND_CUPY
        else:
            backend = self.BACKEND_NUMPY
        return MaskContainer(
            p.mask_factories, dtype=p.mask_dtype, use_sparse=p.use_sparse, count=p.mask_count,
            backend=backend
        )

    def get_task_data(self):
        ''
        engine = ApplyMasksEngine(self.masks, self.meta, self.params.use_torch)
        return {
            'engine': engine,
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
        ''
        return self.params.backends

    def get_method(self) -> Literal[UDFMethodEnum.FRAME, UDFMethodEnum.TILE]:
        if self.params.get('shifts') is not None:
            return UDFMethodEnum.FRAME
        else:
            return UDFMethodEnum.TILE

    def process_tile(self, tile):
        """
        Used for simple mask application, without shifts
        """
        self.results.intensity[:] += self.forbuf(
            self.task_data.engine.process_tile(tile),
            self.results.intensity,
        )

    def process_frame(self, frame):
        """
        Apply shifted masks to a frame
        """
        shifts = self.params.shifts.astype(int)
        self.results.intensity[:] += self.forbuf(
            self.task_data.engine.process_frame_shifted(frame, shifts),
            self.results.intensity,
        )
