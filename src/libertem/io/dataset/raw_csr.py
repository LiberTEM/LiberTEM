import typing

import scipy.sparse
import numpy as np
import numba

from libertem.common import Slice, Shape
from libertem.common.array_backends import SCIPY_CSR, ArrayBackend
from libertem.io.corrections.corrset import CorrectionSet
from libertem.io.dataset.base import (
    DataTile, DataSet
)
from libertem.io.dataset.base.meta import DataSetMeta
from libertem.io.dataset.base.partition import Partition
from libertem.io.dataset.base.tiling_scheme import TilingScheme

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.backend import IOBackend
    from libertem.common.messageconverter import MessageConverter
    from libertem.common.executor import JobExecutor
    import numpy.typing as nt


class CSRDescriptor(typing.NamedTuple):
    indptr_file: str
    indptr_dtype: np.dtype
    coords_file: str
    coords_dtype: np.dtype
    values_file: str
    values_dtype: np.dtype


class CSRTriple(typing.NamedTuple):
    indptr: np.ndarray
    coords: np.ndarray
    values: np.ndarray


class RawCSRDataSet(DataSet):
    """
    Read sparse data in CSR format from a triple of files
    that contain the index pointers, the coordinates and the values.

    Both the navigation and signal axis are flattened, such that existing
    CSR libraries like scipy.sparse can be used.
    """

    def __init__(
        self,
        descriptor: CSRDescriptor,
        nav_shape: typing.Tuple[int, ...],
        sig_shape: typing.Tuple[int, ...],
        io_backend: typing.Optional["IOBackend"],
    ):
        assert io_backend is None
        super().__init__(io_backend=io_backend)
        self._descriptor = descriptor
        self._nav_shape = nav_shape
        self._sig_shape = sig_shape

    def initialize(self, executor: "JobExecutor") -> "DataSet":
        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape))
        descriptor = self._descriptor
        triple = executor.run_function(get_triple, descriptor)
        image_count = triple.indptr.shape[0]
        sync_offset = 0  # TODO
        self._meta = DataSetMeta(
            shape=shape,
            array_backends=[SCIPY_CSR],
            image_count=image_count,
            raw_dtype=self._descriptor.values_dtype,
            dtype=None,
            metadata=None,
            sync_offset=sync_offset,
        )
        return self

    @property
    def dtype(self) -> "nt.DTypeLike":
        assert self._meta is not None
        return self._meta.raw_dtype

    @property
    def shape(self) -> Shape:
        assert self._meta is not None
        return self._meta.shape

    @property
    def array_backends(self) -> typing.Sequence[ArrayBackend]:
        assert self._meta is not None
        return self._meta.array_backends

    def check_valid(self) -> bool:
        return True  # TODO

    @classmethod
    def detect_params(cls, path: str, executor: "JobExecutor"):
        return False  # TODO: can be implemented once we have a sidecar of some sort

    @classmethod
    def get_msg_converter(cls) -> typing.Type["MessageConverter"]:
        raise NotImplementedError()  # TODO: web GUI support once we have a sidecar

    def get_diagnostics(self):
        return []

    @classmethod
    def get_supported_extensions(cls) -> typing.Set[str]:
        return set()  # TODO: extension of the sidecar file

    def get_cache_key(self) -> str:
        raise NotImplementedError()  # TODO

    @classmethod
    def get_supported_io_backends(cls) -> typing.List[str]:
        return []  # FIXME: we may want to read using a backend in the future

    def adjust_tileshape(
        self,
        tileshape: typing.Tuple[int, ...],
        roi: typing.Optional[np.ndarray]
    ) -> typing.Tuple[int, ...]:
        return (tileshape[0],) + tuple(self._sig_shape)

    def need_decode(
        self,
        read_dtype: "nt.DTypeLike",
        roi: typing.Optional[np.ndarray],
        corrections: typing.Optional[CorrectionSet]
    ) -> bool:
        return super().need_decode(read_dtype, roi, corrections)

    def get_partitions(self) -> typing.Generator[Partition, None, None]:
        assert self._meta is not None
        for part_slice, start, stop in self.get_slices():
            yield RawCSRPartition(
                descriptor=self._descriptor,
                meta=self._meta,
                partition_slice=part_slice,
                start_frame=start,
                num_frames=stop - start,
                io_backend=None,
                decoder=None,
            )


class RawCSRPartition(Partition):
    def __init__(
        self,
        descriptor: CSRDescriptor,
        start_frame: int,
        num_frames: int,
        *args,
        **kwargs
    ):
        self._descriptor = descriptor
        self._start_frame = start_frame
        self._num_frames = num_frames
        self._corrections = CorrectionSet()
        self._worker_context = None
        super().__init__(*args, **kwargs)

    def set_corrections(self, corrections: CorrectionSet):
        if corrections.have_corrections():
            raise NotImplementedError("corrections not implemented for raw CSR data set")

    def validate_tiling_scheme(self, tiling_scheme: TilingScheme):
        if len(tiling_scheme) != 1:
            raise ValueError("Cannot slice CSR data in sig dimensions")

    def get_tiles(
        self,
        tiling_scheme: TilingScheme,
        dest_dtype="float32",
        roi=None,
        array_backend: typing.Optional[ArrayBackend] = None
    ):
        assert array_backend == SCIPY_CSR
        tiling_scheme = tiling_scheme.adjust_for_partition(self)
        self.validate_tiling_scheme(tiling_scheme)
        triple = get_triple(self._descriptor)
        if self._corrections is not None and self._corrections.have_corrections():
            raise NotImplementedError(
                "corrections are not yet supported for raw CSR"
            )
        # TODO: sync_offset
        # TODO: dest_dtype
        if roi is None:
            yield from read_tiles_straight(triple, self.slice, tiling_scheme)
        else:
            yield from read_tiles_with_roi(triple, self.slice, tiling_scheme, roi)


def sliced_indptr(triple: CSRTriple, partition_slice: Slice):
    assert len(partition_slice.shape.nav) == 1
    indptr_start = partition_slice.origin[0]
    indptr_stop = indptr_start + partition_slice.shape.nav[0] + 1
    return triple.indptr[indptr_start:indptr_stop]


def get_triple(descriptor: CSRDescriptor) -> CSRTriple:
    values: np.ndarray = np.memmap(descriptor.values_file, dtype=descriptor.values_dtype, mode='r')
    coords: np.ndarray = np.memmap(descriptor.coords_file, dtype=descriptor.coords_dtype, mode='r')
    indptr: np.ndarray = np.memmap(descriptor.indptr_file, dtype=descriptor.indptr_dtype, mode='r')

    return CSRTriple(
        indptr=indptr,
        coords=coords,
        values=values,
    )


def read_tiles_straight(
    triple: CSRTriple,
    partition_slice: Slice,
    tiling_scheme: TilingScheme
):
    assert len(tiling_scheme) == 1

    indptr = sliced_indptr(triple, partition_slice=partition_slice)

    sig_shape = tuple(partition_slice.shape.sig)
    sig_size = partition_slice.shape.sig.size
    sig_dims = len(sig_shape)

    # Technically, one could use the slicing implementation of csr_matrix here.
    # However, it is slower, presumably because it takes a copy
    # Furthermore it provides a template to use an actual I/O backend here
    # instead of memory mapping.
    for indptr_start in range(0, len(indptr) - 1, tiling_scheme.depth):
        indptr_stop = min(indptr_start + tiling_scheme.depth, len(indptr) - 1)

        start = indptr[indptr_start]
        stop = indptr[indptr_stop]
        values = triple.values[start:stop]
        coords = triple.coords[start:stop]
        indptr_slice = indptr[indptr_start:indptr_stop + 1]
        indptr_slice = indptr_slice - indptr_slice[0]
        arr = scipy.sparse.csr_matrix(
            (values, coords, indptr_slice),
            shape=(indptr_stop - indptr_start, sig_size)
        )
        tile_slice = Slice(
            origin=(partition_slice.origin[0] + indptr_start, ) + (0, ) * sig_dims,
            shape=Shape((indptr_stop - indptr_start, ) + sig_shape, sig_dims=sig_dims),
        )
        yield DataTile(
            data=arr,
            tile_slice=tile_slice,
            scheme_idx=0,
        )


@numba.njit(cache=True)
def populate_tile(
    indptr_tile_start: "np.ndarray",
    indptr_tile_stop: "np.ndarray",
    orig_values: "np.ndarray",
    orig_coords: "np.ndarray",
    values_out: "np.ndarray",
    coords_out: "np.ndarray",
    indptr_out: "np.ndarray",
):
    offset = 0
    indptr_out[0] = 0
    for i, (start, stop) in enumerate(zip(indptr_tile_start, indptr_tile_stop)):
        chunk_size = stop - start
        values_out[offset:offset + chunk_size] = orig_values[start:stop]
        coords_out[offset:offset + chunk_size] = orig_coords[start:stop]
        offset += chunk_size
    indptr_out[i + 1] = offset


def read_tiles_with_roi(
    triple: CSRTriple,
    partition_slice: Slice,
    tiling_scheme: TilingScheme,
    roi: np.ndarray,
):
    assert len(tiling_scheme) == 1
    roi = roi.reshape((-1, ))
    part_start = partition_slice.origin[0]
    tile_offset = np.count_nonzero(roi[:part_start])
    part_roi = partition_slice.get(roi, nav_only=True)

    indptr = sliced_indptr(triple, partition_slice=partition_slice)

    sig_shape = tuple(partition_slice.shape.sig)
    sig_size = partition_slice.shape.sig.size
    sig_dims = len(sig_shape)

    start_values = indptr[:-1][part_roi]
    stop_values = indptr[1:][part_roi]

    # Implementing this "by hand" instead of fancy indexing to provide a template to use an
    # actual I/O backend here instead of memory mapping.
    # The native scipy.sparse.csr_matrix implementation of fancy indexing
    # with a boolean mask for nav is very fast.

    for indptr_start in range(0, len(start_values), tiling_scheme.depth):
        indptr_stop = min(indptr_start + tiling_scheme.depth, len(start_values))
        indptr_tile_start = start_values[indptr_start:indptr_stop]
        indptr_tile_stop = stop_values[indptr_start:indptr_stop]
        size = sum(indptr_tile_stop - indptr_tile_start)

        values = np.zeros(dtype=triple.values.dtype, shape=size)
        coords = np.zeros(dtype=triple.coords.dtype, shape=size)
        indptr_slice = np.zeros(
            dtype=indptr.dtype, shape=indptr_stop - indptr_start + 1
        )
        populate_tile(
            indptr_tile_start=indptr_tile_start,
            indptr_tile_stop=indptr_tile_stop,
            orig_values=triple.values,
            orig_coords=triple.coords,
            values_out=values,
            coords_out=coords,
            indptr_out=indptr_slice,
        )

        arr = scipy.sparse.csr_matrix(
            (values, coords, indptr_slice),
            shape=(indptr_stop - indptr_start, sig_size)
        )
        tile_slice = Slice(
            origin=(tile_offset + indptr_start, ) + (0, ) * sig_dims,
            shape=Shape((indptr_stop - indptr_start, ) + sig_shape, sig_dims=sig_dims),
        )
        yield DataTile(
            data=arr,
            tile_slice=tile_slice,
            scheme_idx=0,
        )
