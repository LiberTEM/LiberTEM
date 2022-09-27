import typing

import scipy.sparse
import numpy as np

from libertem.common import Slice, Shape
from libertem.io.dataset.base import (
    DataTile, DataSet
)
from libertem.io.dataset.base.partition import Partition
from libertem.io.dataset.base.tiling_scheme import TilingScheme

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.backend import IOBackend
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

    def initialize(self, executor) -> "DataSet":
        # TODO: self._meta
        return self

    def get_partitions(self) -> typing.Generator[Partition, None, None]:
        return super().get_partitions()

    def get_triple(self):
        return get_triple(self._descriptor)

    def dtype(self) -> "nt.DTypeLike":
        return self._meta.raw_dtype


def sliced_indptr(triple: CSRTriple, partition_slice: Slice):
    assert len(partition_slice.shape.nav) == 1
    indptr_start = partition_slice.origin[0]
    indptr_stop = indptr_start + partition_slice.shape.nav[0] + 1
    return triple.indptr[indptr_start:indptr_stop]


def get_triple(descriptor: CSRDescriptor, partition_slice: Slice) -> CSRTriple:
    values = np.memmap(descriptor.values_file, dtype=descriptor.values_dtype, mode='r')
    coords = np.memmap(descriptor.coords_file, dtype=descriptor.coords_dtype, mode='r')
    indptr = np.memmap(descriptor.indptr_file, dtype=descriptor.indptr_dtype, mode='r')

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
    orig_values = triple.values
    orig_coords = triple.coords

    sig_shape = tuple(partition_slice.shape.sig)
    sig_size = partition_slice.shape.sig.size
    sig_dims = len(sig_shape)

    for indptr_start in range(0, len(indptr) - 1, tiling_scheme.depth):
        indptr_stop = min(indptr_start + tiling_scheme.depth, len(indptr) - 1)

        start = indptr[indptr_start]
        stop = indptr[indptr_stop]
        values = orig_values[start:stop]
        coords = orig_coords[start:stop]
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


"""
def read_part_with_roi(
    descriptor: CSRDescriptor, partition_slice: Slice, roi: typing.Optional[np.ndarray]
):
    indptr = get_indptr()
    indptr_partition = indptr[partition_selector]

    pairs_in_roi = indptr_pairs[roi]

    start_values = pairs_in_roi[0]
    stop_values = pairs_in_roi[1]

    size = sum(stop_values  - start_values)

    values = np.zeros(dtype=read_dtype, shape=size)
    coords = np.zeros(dtype=coord_dtype, shape=size)
    result_indptr = np.zeros(
        dtype=indptr_dtype, shape=pairs_in_roi.shape[1] + 1
    )

    # r_n_d: jit this
    offset = 0
    result_indptr[0] = 0
    for i, (start, stop) in enumerate(pairs_in_roi):
        row_size = stop - start
        result_indptr[i + 1] = offset + row_size

        # ~= read_ranges
        values[offset : offset + row_size] = orig_values[start:stop]
        coords[offset : offset + row_size] = orig_coords[start:stop]

        offset += row_size
    # ...
    return DataTile(...)
"""
