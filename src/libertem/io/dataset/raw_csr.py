import typing
import os

import scipy.sparse
import numpy as np
import numba
import tomli
from sparseconverter import SCIPY_CSR, ArrayBackend, for_backend, NUMPY

from libertem.common import Slice, Shape
from libertem.common.math import prod, count_nonzero
from libertem.io.corrections.corrset import CorrectionSet
from libertem.io.dataset.base import (
    DataTile, DataSet
)
from libertem.io.dataset.base.meta import DataSetMeta
from libertem.io.dataset.base.partition import Partition
from libertem.io.dataset.base.tiling_scheme import TilingScheme
from libertem.common.messageconverter import MessageConverter
from libertem.common.numba import numba_dtypes

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.backend import IOBackend
    from libertem.common.executor import JobExecutor
    import numpy.typing as nt


def load_toml(path: str):
    with open(path, "rb") as f:
        return tomli.load(f)


class RawCSRDatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/RawCSRDatasetParams.schema.json",
        "title": "RawCSRDatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "RAW_CSR"},
            "path": {"type": "string"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
        },
        "required": ["type", "path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


class CSRDescriptor(typing.NamedTuple):
    indptr_file: str
    indptr_dtype: np.dtype
    indices_file: str
    indices_dtype: np.dtype
    data_file: str
    data_dtype: np.dtype


class CSRTriple(typing.NamedTuple):
    indptr: np.ndarray
    indices: np.ndarray
    data: np.ndarray


class RawCSRDataSet(DataSet):
    """
    Read sparse data in compressed sparse row (CSR) format from a triple of files
    that contain the index pointers, the coordinates and the values. See
    `Wikipedia article on the CSR format
    <https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)>`_
    for more information on the format.

    The necessary parameters are specified in a TOML file like this:

    .. code-block::

        [params]

        filetype = "raw_csr"
        nav_shape = [512, 512]
        sig_shape = [516, 516]

        [raw_csr]

        indptr_file = "rowind.dat"
        indptr_dtype = "<i4"

        indices_file = "coords.dat"
        indices_dtype = "<i4"

        data_file = "values.dat"
        data_dtype = "<i4"`

    Both the navigation and signal axis are flattened in the file, so that existing
    CSR libraries like scipy.sparse can be used directly by memory-mapping or
    reading the file contents.

    Parameters
    ----------

    path : str
        Path to the TOML file with file names and other parameters for the sparse dataset.
    nav_shape : Tuple[int, int], optional
        A nav_shape to apply to the dataset overriding the shape
        value read from the TOML file, by default None. This can
        be used to read a subset of the data, or reshape the
        contained data.
    sig_shape : Tuple[int, int], optional
        A sig_shape to apply to the dataset overriding the shape
        value read from the TOML file, by default None.
    sync_offset : int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start
    io_backend : IOBackend, optional
        The I/O backend to use, see :ref:`io backends`, by default None.
    num_partitions: int, optional
        Override the number of partitions. This is useful if the
        default number of partitions, chosen based on common workloads,
        creates partitions which are too large (or small) for the UDFs
        being run on this dataset.

    Examples
    --------

    >>> ds = ctx.load("raw_csr", path='./path_to.toml')  # doctest: +SKIP
    """

    def __init__(
        self,
        path: str,
        nav_shape: typing.Optional[tuple[int, ...]] = None,
        sig_shape: typing.Optional[tuple[int, ...]] = None,
        sync_offset: int = 0,
        io_backend: typing.Optional["IOBackend"] = None,
        num_partitions: typing.Optional[int] = None,
    ):
        if io_backend is not None:
            raise NotImplementedError()
        super().__init__(
            io_backend=io_backend,
            num_partitions=num_partitions,
        )
        self._path = path
        if nav_shape is not None:
            nav_shape = tuple(nav_shape)
        self._nav_shape = nav_shape
        if sig_shape is not None:
            sig_shape = tuple(sig_shape)
        self._sig_shape = sig_shape
        self._sync_offset = sync_offset
        self._conf = None
        self._descriptor = None

    def initialize(self, executor: "JobExecutor") -> "DataSet":
        self._conf = conf = executor.run_function(load_toml, self._path)
        assert conf is not None
        if conf['params']['filetype'].lower() != 'raw_csr':
            raise ValueError(f"Filetype is not CSR, found {conf['params']['filetype']}")
        nav_shape = tuple(conf['params']['nav_shape'])
        sig_shape = tuple(conf['params']['sig_shape'])
        if self._nav_shape is None:
            self._nav_shape = nav_shape
        if self._sig_shape is None:
            self._sig_shape = sig_shape
        else:
            if prod(self._sig_shape) != prod(sig_shape):
                raise ValueError(f"Sig size mismatch between {self._sig_shape} and {sig_shape}.")

        shape = Shape(self._nav_shape + self._sig_shape, sig_dims=len(self._sig_shape))
        self._descriptor = descriptor = executor.run_function(get_descriptor, self._path)
        executor.run_function(
            check,
            descriptor=descriptor,
            nav_shape=self._nav_shape,
            sig_shape=self._sig_shape
        )
        image_count = executor.run_function(get_nav_size, descriptor=descriptor)
        self._image_count = image_count
        self._nav_shape_product = int(prod(self._nav_shape))
        self._sync_offset_info = self.get_sync_offset_info()
        self._meta = DataSetMeta(
            shape=shape,
            array_backends=[SCIPY_CSR],
            image_count=image_count,
            raw_dtype=descriptor.data_dtype,
            dtype=None,
            metadata=None,
            sync_offset=self._sync_offset,
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

    def get_base_shape(self, roi):
        return (1, ) + tuple(self.shape.sig)

    def get_max_io_size(self):
        # High value since referring to dense for the time being
        # Compromise between memory use during densification and
        # performance with native sparse
        return int(1024*1024*20)

    def check_valid(self) -> bool:
        return True  # TODO

    @staticmethod
    def _get_filesize(path):
        return os.stat(path).st_size

    def supports_correction(self):
        return False

    @classmethod
    def detect_params(cls, path: str, executor: "JobExecutor"):
        try:
            _, extension = os.path.splitext(path)
            has_extension = extension.lstrip('.') in cls.get_supported_extensions()
            under_size_lim = executor.run_function(cls._get_filesize, path) < 2**20  # 1 MB
            if not (has_extension or under_size_lim):
                return False
            conf = executor.run_function(load_toml, path)
            if "params" not in conf:
                return False

            if "filetype" not in conf["params"]:
                return False
            if conf["params"]["filetype"].lower() != "raw_csr":
                return False
            descriptor = executor.run_function(get_descriptor, path)
            image_count = executor.run_function(get_nav_size, descriptor=descriptor)
            return {
                "parameters": {
                    'path': path,
                    "nav_shape": conf["params"]["nav_shape"],
                    "sig_shape": conf["params"]["sig_shape"],
                    "sync_offset": 0,
                },
                "info": {
                    "image_count": image_count,
                }
            }
        except (TypeError, UnicodeDecodeError, tomli.TOMLDecodeError, OSError):
            return False

    @classmethod
    def get_msg_converter(cls) -> type["MessageConverter"]:
        return RawCSRDatasetParams

    def get_diagnostics(self):
        return [
            {"name": "data dtype", "value": str(self._descriptor.data_dtype)},
            {"name": "indptr dtype", "value": str(self._descriptor.indptr_dtype)},
            {"name": "indices dtype", "value": str(self._descriptor.indices_dtype)},
        ]  # TODO: nonzero elements?

    @classmethod
    def get_supported_extensions(cls) -> set[str]:
        return {"toml"}

    def get_cache_key(self) -> str:
        raise NotImplementedError()  # TODO

    @classmethod
    def get_supported_io_backends(cls) -> list[str]:
        return []  # FIXME: we may want to read using a backend in the future

    def adjust_tileshape(
        self,
        tileshape: tuple[int, ...],
        roi: typing.Optional[np.ndarray]
    ) -> tuple[int, ...]:
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

    def set_corrections(self, corrections: typing.Optional[CorrectionSet]):
        if corrections is not None and corrections.have_corrections():
            raise NotImplementedError("corrections not implemented for raw CSR data set")

    def validate_tiling_scheme(self, tiling_scheme: TilingScheme):
        if len(tiling_scheme) != 1:
            raise ValueError("Cannot slice CSR data in sig dimensions")

    def get_locations(self):
        # Allow using any worker by default
        return None

    def get_tiles(
        self,
        tiling_scheme: TilingScheme,
        dest_dtype="float32",
        roi=None,
        array_backend: typing.Optional[ArrayBackend] = None
    ):
        assert array_backend == SCIPY_CSR or array_backend is None
        tiling_scheme = tiling_scheme.adjust_for_partition(self)
        self.validate_tiling_scheme(tiling_scheme)
        triple = get_triple(self._descriptor)
        if self._corrections is not None and self._corrections.have_corrections():
            raise NotImplementedError(
                "corrections are not yet supported for raw CSR"
            )
        if roi is None:
            yield from read_tiles_straight(
                triple, self.slice, self.meta.sync_offset, tiling_scheme, dest_dtype
            )
        else:
            yield from read_tiles_with_roi(
                triple, self.slice, self.meta.sync_offset, tiling_scheme, roi, dest_dtype
            )


def sliced_indptr(triple: CSRTriple, partition_slice: Slice, sync_offset: int):
    assert len(partition_slice.shape.nav) == 1
    skip = min(0, partition_slice.origin[0] + sync_offset)
    indptr_start = max(0, partition_slice.origin[0] + sync_offset)
    indptr_stop = max(0, partition_slice.origin[0] + partition_slice.shape.nav[0] + 1 + sync_offset)
    return skip, triple.indptr[indptr_start:indptr_stop]


def get_triple(descriptor: CSRDescriptor) -> CSRTriple:
    data: np.ndarray = np.memmap(
        descriptor.data_file,
        dtype=descriptor.data_dtype,
        mode='r'
    )
    indices: np.ndarray = np.memmap(
        descriptor.indices_file,
        dtype=descriptor.indices_dtype,
        mode='r'
    )
    indptr: np.ndarray = np.memmap(
        descriptor.indptr_file,
        dtype=descriptor.indptr_dtype,
        mode='r'
    )

    return CSRTriple(
        indptr=indptr,
        indices=indices,
        data=data,
    )


def check(descriptor: CSRDescriptor, nav_shape, sig_shape, debug=False):
    triple = get_triple(descriptor)
    if triple.indices.shape != triple.data.shape:
        raise RuntimeError('Shape mismatch between data and indices.')
    if debug:
        assert np.min(triple.indices) >= 0
        assert np.max(triple.indices) < prod(sig_shape)
        assert np.min(triple.indptr) >= 0
        assert np.max(triple.indptr) == len(triple.indices)


def get_descriptor(path: str) -> CSRDescriptor:
    """
    Get a CSRDescriptor from the path to a toml sidecar file
    """
    conf = load_toml(path)
    assert conf is not None
    if conf['params']['filetype'].lower() != 'raw_csr':
        raise ValueError(f"Filetype is not CSR, found {conf['params']['filetype']}")

    base_path = os.path.dirname(path)
    # make sure the key is not case sensitive to follow the convention of
    # the Context.load() function.
    csr_key = conf['params']['filetype']
    csr_conf = conf[csr_key]
    return CSRDescriptor(
        indptr_file=os.path.join(base_path, csr_conf['indptr_file']),
        indptr_dtype=csr_conf['indptr_dtype'],
        indices_file=os.path.join(base_path, csr_conf['indices_file']),
        indices_dtype=csr_conf['indices_dtype'],
        data_file=os.path.join(base_path, csr_conf['data_file']),
        data_dtype=csr_conf['data_dtype'],
    )


def get_nav_size(descriptor: CSRDescriptor) -> int:
    '''
    To run efficiently on a remote worker for dataset initialization
    '''
    indptr = np.memmap(
        descriptor.indptr_file,
        dtype=descriptor.indptr_dtype,
        mode='r',
    )
    return len(indptr) - 1


def read_tiles_straight(
    triple: CSRTriple,
    partition_slice: Slice,
    sync_offset: int,
    tiling_scheme: TilingScheme,
    dest_dtype: np.dtype,
):
    assert len(tiling_scheme) == 1

    skip, indptr = sliced_indptr(
        triple,
        partition_slice=partition_slice,
        sync_offset=sync_offset
    )

    sig_shape = tuple(partition_slice.shape.sig)
    sig_size = partition_slice.shape.sig.size
    sig_dims = len(sig_shape)

    # Technically, one could use the slicing implementation of csr_matrix here.
    # However, it is slower, presumably because it takes a copy
    # Furthermore it provides a template to use an actual I/O backend here
    # instead of memory mapping.
    for indptr_start in range(0, len(indptr) - 1, tiling_scheme.depth):
        tile_start = indptr_start - skip  # skip is a negative value or 0
        indptr_stop = min(indptr_start + tiling_scheme.depth, len(indptr) - 1)
        if indptr_stop - indptr_start <= 0:
            continue

        indptr_slice = indptr[indptr_start:indptr_stop + 1]

        start = indptr[indptr_start]
        stop = indptr[indptr_stop]
        data = triple.data[start:stop]
        if dest_dtype != data.dtype:
            data = data.astype(dest_dtype)
        indices = triple.indices[start:stop]

        indptr_slice = indptr_slice - indptr_slice[0]
        arr = scipy.sparse.csr_matrix(
            (data, indices, indptr_slice),
            shape=(indptr_stop - indptr_start, sig_size)
        )
        tile_slice = Slice(
            origin=(partition_slice.origin[0] + tile_start, ) + (0, ) * sig_dims,
            shape=Shape((arr.shape[0], ) + sig_shape, sig_dims=sig_dims),
        )
        yield DataTile(
            data=arr,
            tile_slice=tile_slice,
            scheme_idx=0,
        )


def populate_tile(
    indptr_tile_start: "np.ndarray",
    indptr_tile_stop: "np.ndarray",
    orig_data: "np.ndarray",
    orig_indices: "np.ndarray",
    data_out: "np.ndarray",
    indices_out: "np.ndarray",
    indptr_out: "np.ndarray",
):
    offset = 0
    indptr_out[0] = 0
    for i, (start, stop) in enumerate(zip(indptr_tile_start, indptr_tile_stop)):
        chunk_size = stop - start
        data_out[offset:offset + chunk_size] = orig_data[start:stop]
        indices_out[offset:offset + chunk_size] = orig_indices[start:stop]
        offset += chunk_size
        indptr_out[i + 1] = offset


populate_tile_numba = numba.njit(populate_tile)


def can_use_numba(triple: CSRTriple) -> bool:
    return all(d in numba_dtypes
        for d in (triple.data.dtype, triple.indices.dtype, triple.indptr.dtype))


def read_tiles_with_roi(
    triple: CSRTriple,
    partition_slice: Slice,
    sync_offset: int,
    tiling_scheme: TilingScheme,
    roi: np.ndarray,
    dest_dtype: np.dtype,
):
    assert len(tiling_scheme) == 1
    roi = roi.reshape((-1, ))
    part_start = max(0, partition_slice.origin[0])
    tile_offset = count_nonzero(roi[:part_start])
    part_roi = partition_slice.get(roi, nav_only=True)

    skip, indptr = sliced_indptr(triple, partition_slice=partition_slice, sync_offset=sync_offset)

    if skip < 0:
        skipped_part_roi = part_roi[-skip:]
    else:
        skipped_part_roi = part_roi

    roi_overhang = max(0, len(skipped_part_roi) - len(indptr) + 1)
    if roi_overhang:
        real_part_roi = skipped_part_roi[:-roi_overhang]
    else:
        real_part_roi = skipped_part_roi

    real_part_roi = for_backend(real_part_roi, NUMPY)

    sig_shape = tuple(partition_slice.shape.sig)
    sig_size = partition_slice.shape.sig.size
    sig_dims = len(sig_shape)

    start_values = indptr[:-1][real_part_roi]
    stop_values = indptr[1:][real_part_roi]

    # Implementing this "by hand" instead of fancy indexing to provide a template to use an
    # actual I/O backend here instead of memory mapping.
    # The native scipy.sparse.csr_matrix implementation of fancy indexing
    # with a boolean mask for nav is very fast.

    if can_use_numba(triple):
        my_populate_tile = populate_tile_numba
    else:
        my_populate_tile = populate_tile

    for indptr_start in range(0, len(part_roi), tiling_scheme.depth):
        indptr_stop = min(indptr_start + tiling_scheme.depth, len(start_values))
        indptr_start = min(indptr_start, indptr_stop)
        # Don't read empty slices
        if indptr_stop - indptr_start <= 0:
            continue
        # Cast to int64 to avoid later upcasting to float64 in case of uint64
        # We can safely assume that files have less than 2**63 entries so that casting
        # from uint64 to int64 should be safe
        indptr_tile_start = start_values[indptr_start:indptr_stop].astype(np.int64)
        indptr_tile_stop = stop_values[indptr_start:indptr_stop].astype(np.int64)
        size = sum(indptr_tile_stop - indptr_tile_start)

        data = np.zeros(dtype=dest_dtype, shape=size)
        indices = np.zeros(dtype=triple.indices.dtype, shape=size)
        indptr_slice = np.zeros(
            dtype=indptr.dtype, shape=indptr_stop - indptr_start + 1
        )
        my_populate_tile(
            indptr_tile_start=indptr_tile_start,
            indptr_tile_stop=indptr_tile_stop,
            orig_data=triple.data,
            orig_indices=triple.indices,
            data_out=data,
            indices_out=indices,
            indptr_out=indptr_slice,
        )

        arr = scipy.sparse.csr_matrix(
            (data, indices, indptr_slice),
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
