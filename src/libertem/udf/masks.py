from typing_extensions import Literal
from libertem.common.math import prod
import numpy as np

from libertem.common.udf import UDFMethod
from libertem.udf import UDF, UDFMeta
from libertem.common.buffers import AuxBufferWrapper
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
            and isinstance(self.masks.use_sparse, str)
            and self.masks.use_sparse.startswith('scipy.sparse')
        ):
            # Due to https://github.com/scipy/scipy/issues/13211
            self.process_flat = self._process_flat_spsp
        elif (
            self.meta.array_backend in (
                UDF.BACKEND_SCIPY_COO,
                UDF.BACKEND_SCIPY_CSR,
                UDF.BACKEND_SCIPY_CSC
            ) and isinstance(self.masks.use_sparse, str)
            and self.masks.use_sparse.startswith('sparse.pydata')
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
        # CuPy back-end disables torch in get_task_data
        # FIXME use GPU torch with CuPy array?
        return torch.mm(
            torch.from_numpy(flat_tile),
            torch.from_numpy(masks),
        ).numpy()

    def _process_flat_spsp(self, flat_tile, masks):
        return rmatmul(flat_tile, masks)

    def _process_flat_sparsepyd(self, flat_tile, masks):
        # Make sure the sparse.pydata mask comes first
        # to choose the right multiplication method
        return (masks @ flat_tile.T).T

    def _process_flat_standard(self, flat_tile, masks):
        return flat_tile @ masks

    def process_tile(self, tile):
        flat_shape = (tile.shape[0], prod(tile.shape[1:]))
        # Avoid reshape since older versions of scipy.sparse don't support it
        flat_data = tile.reshape(flat_shape) if tile.shape != flat_shape else tile
        return self.process_flat(flat_data, self._get_masks())

    def process_frame_shifted(self, frame, shifts: tuple[int, ...]):
        sig_shape = self.meta.dataset_shape.sig
        masks = self._get_masks()
        num_masks = len(self.masks)
        shifted_slice = self.meta.sig_slice.shift_by(shifts)
        inverse_shifted_slice = self.meta.sig_slice.shift_by(-1 * shifts)
        left = self.meta.sig_slice.intersection_with(shifted_slice)
        right = self.meta.sig_slice.intersection_with(inverse_shifted_slice)
        if left.is_null():
            # Zero overlap after shifts, shortcut return
            return np.zeros((num_masks,), dtype=np.float32)
        mask_slice = right.get()
        if self.needs_transpose:
            # expects masks in shape (sig_size, num_masks)
            mask_slice = mask_slice + (slice(None), )
            masks = masks.reshape((*sig_shape, num_masks))
            final_mask_shape = (-1, num_masks)
        else:
            # expects masks in shape (num_masks, sig_size)
            # NOTE unexpectedly don't need to reverse sig_shape or mask_slice ?
            masks = masks.reshape((num_masks, *sig_shape))
            mask_slice = (slice(None), ) + mask_slice
            final_mask_shape = (num_masks, -1)

        sliced_masks = masks[mask_slice].reshape(final_mask_shape)
        # shift slicing requires sig_shape frames
        # sparse array backend can provide flat frame
        frame = frame.reshape(sig_shape)
        try:
            data = left.get(frame)
        except TypeError as e:
            # frame is in a form which doesn't support slicing
            # the only recognized case is scipy.sparse.coo
            if not hasattr(frame, 'getformat'):
                raise e  # pragma: no cover
            assert frame.getformat() == 'coo'
            frame = frame.tocsr()
            data = left.get(frame)
        flat_data = data.reshape((1, -1))
        return self.process_flat(flat_data, sliced_masks).reshape((num_masks,))


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
        * None (default): Where possible use sparse matrix multiplication if all factory \
            functions return a sparse mask, otherwise convert all masks to dense matrices \
            and use dense matrix multiplication
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
    shifts : Union[Tuple[int, int], AuxBufferWrapper], optional
        (Y/X)-shifts to apply to all masks before multiplying with each frame. Can be either a
        length-2 array-like for a constant :code:`(Y, X)` shift, or an
        :class:`~libertem.common.buffers.AuxBufferWrapper` of
        :code:`(kind='nav', extra_shape=(2,), dtype=int)` defining a per-frame shift to apply.

        A positive y-shift moves the mask 'down' relative to the frame, while a positive
        x-shift moves the mask 'right' relative to the frame. Elements of the mask and frame
        which do not overlap after the shift are discarded. A shift resulting in no overlap
        at all will return a sum of :code:`0.` for that frame.

        .. note::
            Float shift values are cast to integers internally; round values before
            passing the shifts argument to better control the exact shifts used.

        .. note::
            The :code:`shifts` parameter requires frame-by-frame processing to function, and so
            adds a performance penalty compared to unshifted mask application. If applying a
            constant shift it may be worthwhile to manually create a new, pre-shifted mask
            rather than relying on this feature.

            Shifting is also currently incompatible with :code:`scipy.sparse` masks. If sparse
            processing is required then where possible :code:`scipy.sparse` masks are converted to
            :code:`sparse.pydata` equivalents. A consequence of this is that sparse processing
            is not yet supported through CuPy when shifts are enabled, as :code:`sparse.pydata`
            has no current CuPy implementation. If sparse masks are supplied on a CuPy backend
            when :code:`use_sparse=None` (the default) they will be densified to allow the
            calculation to take place.

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

    Masks can be shifted relative to the data using the :code:`shifts` parameter,
    this can either be a constant shift for all frames:

    >>> udf = ApplyMasksUDF(mask_factories=my_masks, shifts=(2, -5))
    >>> res_shift_constant = ctx.run_udf(dataset=dataset, udf=udf)['intensity']

    or a per-frame shift supplied using an :class:`~libertem.common.buffers.AuxBufferWrapper`
    created using :meth:`~libertem.udf.base.UDF.aux_data`:

    >>> shifts = np.random.randint(-8, 8, size=(16, 16, 2)).ravel()
    >>> udf = ApplyMasksUDF(
    ...         mask_factories=my_masks,
    ...         shifts=ApplyMasksUDF.aux_data(
    ...             shifts,
    ...             kind='nav',
    ...             extra_shape=(2,),
    ...             dtype=shifts.dtype,
    ...         )
    ...     )
    >>> res_shift_aux = ctx.run_udf(dataset=dataset, udf=udf)['intensity']

    .. versionadded:: 0.4.0

    .. versionchanged:: 0.13.0
        Added the :code:`shifts` parameter
    '''
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, mask_count=None,
                mask_dtype=None, preferred_dtype=None, backends=None, shifts=None, **kwargs):

        _backends = backends
        not_supported = (
            self.BACKEND_SCIPY_COO_ARRAY,
            self.BACKEND_SCIPY_CSR_ARRAY,
            self.BACKEND_SCIPY_CSC_ARRAY,
        )
        supported_backends = tuple(b for b in self.BACKEND_ALL if b not in not_supported)
        if backends is None:
            backends = supported_backends
        backends = tuple(b for b in backends if b in supported_backends)

        if shifts is not None:
            if isinstance(use_sparse, str) and use_sparse.startswith('scipy.sparse'):
                # This is 'unsupported' because we need to slice the mask stack
                # in the signal dimensions to shift it, and sig is flattened
                # to give to 2D matrix in scipy.sparse
                raise ValueError(
                    f'Sparse backend {use_sparse} not supported for '
                    'shifts, use sparse.pydata instead.'
                )
            if not isinstance(shifts, AuxBufferWrapper):
                shifts = np.asarray(shifts)

            backends = tuple(
                b for b in backends
                if b not in (
                    # Here we are doing frame-by-frame processing, so we can
                    # accept scipy.sparse-style frames, however we need to
                    # perform a reshape into sig-shaped frames which normally
                    # casts the frame into coo_matrix form, which then
                    # has to be re-cast into csr_matrix form to be sliced (i.e. shifted)
                    self.BACKEND_SCIPY_COO,  # cannot be sliced
                    self.BACKEND_CUPY_SCIPY_COO,  # cannot be sliced
                )
            )

        if len(backends) == 0:
            raise ValueError(f'No compatible backend found in {_backends}')

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
        # In the default case defer to default kwarg on MaskContainer
        default_sparse = {}
        if p.shifts is None:
            default_sparse['default_sparse'] = 'scipy.sparse'
        else:
            default_sparse['default_sparse'] = 'sparse.pydata'
        return MaskContainer(
            p.mask_factories, dtype=p.mask_dtype, use_sparse=p.use_sparse, count=p.mask_count,
            backend=backend, **default_sparse,
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

    def get_method(self) -> Literal[UDFMethod.FRAME, UDFMethod.TILE]:
        """
        :meta private:
        """
        if self.params.get('shifts') is not None:
            return UDFMethod.FRAME
        else:
            return UDFMethod.TILE

    def process_tile(self, tile):
        """
        Used for simple mask application, without shifts

        :meta private:
        """
        self.results.intensity[:] += self.forbuf(
            self.task_data.engine.process_tile(tile),
            self.results.intensity,
        )

    def process_frame(self, frame):
        """
        Apply shifted masks to a frame

        :meta private:
        """
        shifts = self.params.shifts.astype(int)
        self.results.intensity[:] += self.forbuf(
            self.task_data.engine.process_frame_shifted(frame, shifts),
            self.results.intensity,
        )
