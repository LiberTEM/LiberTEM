from collections import defaultdict
from enum import Enum
from typing import (
    Any, AsyncGenerator, Dict, Generator, Iterator, Mapping, Optional, List,
    Tuple, Type, Iterable, TypeVar, Union, Set, TYPE_CHECKING
)
from typing_extensions import Protocol, runtime_checkable, Literal, TypedDict
import warnings
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import numpy as np

from libertem.io.dataset.base.tiling import DataTile

from libertem.warnings import UseDiscouragedWarning
from libertem.exceptions import UDFException
from libertem.common.buffers import (
    BufferWrapper, AuxBufferWrapper, PlaceholderBufferWrapper,
    BufferKind, BufferUse, BufferLocation,
)
from libertem.common import Shape, Slice
from libertem.common.math import prod
from libertem.io.dataset.base import (
    TilingScheme, Negotiator, Partition, DataSet, get_coordinates
)
from libertem.corrections import CorrectionSet
from libertem.common.backend import get_use_cuda, get_device_class
from libertem.utils.async_utils import async_generator_eager
from libertem.executor.inline import InlineJobExecutor
from libertem.executor.base import Environment, JobExecutor, TaskProtocol

if TYPE_CHECKING:
    from typing import OrderedDict
    from numpy import typing as nt

log = logging.getLogger(__name__)

Backend = Literal['numpy', 'cuda', 'cupy']
BackendSpec = Union[Backend, Iterable[Backend]]
ResourceDef = Dict[
    Literal[
        'CPU', 'compute', 'ndarray', 'CUDA',
    ],
    int
]
DeviceClass = Literal['cpu', 'cuda']
UDFKwarg = Union[Any, AuxBufferWrapper]
UDFKwargs = Dict[str, UDFKwarg]


# markers for special values:
class TileDepthEnum(Enum):
    TILE_DEPTH_DEFAULT = object()


class TileSizeEnum(Enum):
    TILE_SIZE_BEST_FIT = object()


class TilingPreferences(TypedDict):
    depth: Union[int, TileDepthEnum]
    total_size: Union[float, int]


class UDFMeta:
    """
    UDF metadata. Makes all relevant metadata accessible to the UDF. Can be different
    for each task/partition.

    .. versionchanged:: 0.4.0
        Added distinction of dataset_dtype and input_dtype

    .. versionchanged:: 0.6.0
        Information on compute backend, corrections, coordinates and tiling scheme added

    .. versionchanged:: 0.9.0
        :code:`tiling_scheme_idx` and `sig_slice` added
    """

    def __init__(
        self,
        partition_slice: Optional[Slice],
        dataset_shape: Shape,
        roi: Optional[np.ndarray],
        dataset_dtype: "nt.DTypeLike",
        input_dtype: "nt.DTypeLike",
        tiling_scheme: TilingScheme = None,
        tiling_index: int = 0,
        corrections: Optional[CorrectionSet] = None,
        device_class: Optional[DeviceClass] = None,
        threads_per_worker: Optional[int] = None
    ):
        self._partition_slice = partition_slice
        self._dataset_shape = dataset_shape
        self._dataset_dtype = dataset_dtype
        self._input_dtype = input_dtype
        self._tiling_scheme = tiling_scheme
        self._tiling_index = tiling_index
        if device_class is None:
            device_class = 'cpu'
        self._device_class = device_class
        if roi is not None:
            roi = roi.reshape(tuple(dataset_shape.nav))
        self._roi = roi
        self._slice: Optional[Slice] = None
        self._cached_coordinates: Optional[np.ndarray] = None
        if corrections is None:
            corrections = CorrectionSet()
        self._corrections = corrections
        self._threads_per_worker = threads_per_worker

    @property
    def slice(self) -> Optional[Slice]:
        """
        Slice : A :class:`~libertem.common.slice.Slice` instance that describes the location
                within the dataset with navigation dimension flattened and reduced to the ROI.
        """
        return self._slice

    @slice.setter
    def slice(self, new_slice: Slice) -> None:
        self._slice = new_slice

    @property
    def partition_shape(self) -> Shape:
        """
        Shape : The shape of the partition this UDF currently works on.
                If a ROI was applied, the shape will be modified accordingly.
        """
        if self._partition_slice is None:
            raise ValueError("cannot get partition_shape if partition_slice is None")
        return self._partition_slice.shape

    @property
    def dataset_shape(self) -> Shape:
        """
        Shape : The original shape of the whole dataset, not influenced by the ROI
        """
        return self._dataset_shape

    @property
    def tiling_scheme(self) -> Optional[TilingScheme]:
        """
        TilingScheme : the tiling scheme that was negotiated

        .. versionadded:: 0.6.0
        """
        return self._tiling_scheme

    @property
    def tiling_scheme_idx(self) -> int:
        '''
        Index of the current tile in :attr:`tiling_scheme`.
        '''
        return self._tiling_index

    @tiling_scheme_idx.setter
    def tiling_scheme_idx(self, new_idx: int) -> None:
        self._tiling_index = new_idx

    @property
    def sig_slice(self) -> Slice:
        '''
        Signal slice of the current tile.

        Since all tiles follow the same tiling scheme, this avoids repeatedly
        calculating the signal part of :attr:`slice`. Instead, the
        appropriate slice from the tiling scheme can be re-used.
        '''
        assert self._tiling_scheme is not None
        return self._tiling_scheme[self._tiling_index]

    @property
    def roi(self) -> Optional[np.ndarray]:
        """
        numpy.ndarray : Boolean array which limits the elements the UDF is working on.
                     Has a shape of :attr:`dataset_shape.nav`.
        """
        return self._roi

    @property
    def dataset_dtype(self) -> "nt.DTypeLike":
        """
        numpy.dtype : Native dtype of the dataset
        """
        return self._dataset_dtype

    @property
    def input_dtype(self) -> "nt.DTypeLike":
        """
        numpy.dtype : dtype of the data that will be passed to the UDF

        This is determined from the dataset's native dtype and
        :meth:`UDF.get_preferred_input_dtype` using :meth:`numpy.result_type`

        .. versionadded:: 0.4.0
        """
        return self._input_dtype

    @property
    def corrections(self) -> CorrectionSet:
        """
        CorrectionSet : correction data that is available, either from the dataset
        or specified by the user

        .. versionadded:: 0.6.0
        """
        return self._corrections

    @property
    def device_class(self) -> str:
        '''
        Which device class is used.

        The back-end library can be accessed as :attr:`libertem.udf.base.UDF.xp`.
        This additional string information is used since that way the back-end can be probed without
        importing them all and testing them against :attr:`libertem.udf.base.UDF.xp`.

        Current values are :code:`cpu` (default) or :code:`cuda`.

        .. versionadded:: 0.6.0
        '''
        return self._device_class

    @property
    def coordinates(self) -> np.ndarray:
        """
        np.ndarray : Array of coordinates that correspond to the frames in the actual
        navigation space which are part of the current tile or partition.

        .. versionadded:: 0.6.0
        """
        assert self._slice is not None
        assert self._partition_slice is not None
        if self._cached_coordinates is None:
            self._cached_coordinates = get_coordinates(
                self._partition_slice,
                self._dataset_shape,
                self._roi
            )
        shifted_slice = self._slice.shift(self._partition_slice).get(nav_only=True)
        return self._cached_coordinates[shifted_slice]

    @property
    def threads_per_worker(self) -> Optional[int]:
        """
        int or None : number of threads that a UDF is allowed to use in the `process_*` method.
                      For Numba, pyfftw, Torch, NumPy and SciPy (OMP, MKL, OpenBLAS), this limit
                      is set automatically; this property can be used for other cases, like manually
                      creating thread pools or setting limits for unsupported modules.
                      :code:`None` means no limit is set, and the UDF can use any number of threads
                      it deems necessary (should be limited to system limits, of course).

        Note
        ----

        .. versionchanged:: 0.8.0
            Since discovery of loaded libraries can be slow with :mod:`threadpoolctl`
            (:issue:`1117`), they are cached now. In case an UDF triggers loading of a new
            library or instance of a library that is supported by :mod:`threadpoolctl`,
            it will only be discovered in the first run on a :class:`~libertem.api.Context`.
            The threading settings for such other libraries or instances can therefore depend on the
            execution order. In such cases the thread count for affected libraries should be
            set in the UDF based on :code:`threads_per_worker`. Numba, pyfftw, Torch, NumPy
            and SciPy should not be affected since they are loaded before the first discovery.

        See also: :func:`libertem.utils.threading.set_num_threads`

        .. versionadded:: 0.7.0
        """
        return self._threads_per_worker


class MergeAttrMapping:
    def __init__(self, dict_input: Dict[str, np.ndarray]):
        self._dict: Dict[str, np.ndarray] = dict_input

    def __iter__(self) -> Iterator[str]:
        return iter(self._dict)

    def __contains__(self, k: str) -> bool:
        return k in self._dict

    def __getattr__(self, k: str) -> np.ndarray:
        return self._dict[k]

    def __setattr__(self, k: str, v: np.ndarray) -> None:
        if k in ['_dict']:
            super().__setattr__(k, v)
        else:
            self._dict[k][:] = v

    def __getitem__(self, k: str) -> np.ndarray:
        warnings.warn(
            "dict-like access is discouraged, as it can be "
            "confusing vs. using attribute access",
            UseDiscouragedWarning,
            stacklevel=2,
        )
        return self._dict[k]


T = TypeVar('T')


class UDFData:
    '''
    Container for result buffers, return value from running UDFs
    '''

    def __init__(self, data: Dict[str, BufferWrapper]):
        self._data = data
        self._views: Dict[str, np.ndarray] = {}

    def __repr__(self) -> str:
        return "<UDFData: %r>" % (
            self._data
        )

    def __getattr__(self, k: str) -> Union[np.ndarray, BufferWrapper]:
        if k.startswith("_"):
            raise AttributeError("no such attribute: %s" % k)
        try:
            return self._get_view_or_data(k)
        except KeyError as e:
            raise AttributeError(str(e))

    def get_buffer(self, name: str) -> BufferWrapper:
        """
        Return the `BufferWrapper` for buffer `name`

        .. versionadded:: 0.7.0
        """
        return self._data[name]

    def set_buffer(self, name: str, buffer: BufferWrapper) -> None:
        """
        Replace or set the `BufferWrapper` for buffer `name`
        """
        assert isinstance(buffer, BufferWrapper)
        self._data[name] = buffer

    def get(
        self, k: str, default: Optional[T] = None
    ) -> Optional[Union[T, np.ndarray, BufferWrapper]]:
        try:
            return self.__getattr__(k)
        except KeyError:
            return default

    def __setattr__(self, k: str, v: "nt.ArrayLike") -> None:
        if not k.startswith("_"):
            # convert UDFData.some_attr = something to array slice assignment
            getattr(self, k)[:] = v
        else:
            super().__setattr__(k, v)

    def _get_view_or_data(self, k: str) -> Union[np.ndarray, BufferWrapper]:
        if k in self._views:
            return self._views[k]
        res = self._data[k]
        if isinstance(res, BufferWrapper) and res.raw_data is not None:
            return res.raw_data
        return res

    def __getitem__(self, k: str) -> BufferWrapper:
        warnings.warn(
            "dict-like access is discouraged, as it can be "
            "confusing vs. using attribute access. Please use `get_buffer` instead, "
            "if you really need the `BufferWrapper` and not the current view",
            UseDiscouragedWarning,
            stacklevel=2,
        )
        return self._data[k]

    def __contains__(self, k: str) -> bool:
        return k in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def values(self):
        return self._data.values()

    def as_dict(self) -> Dict[str, BufferWrapper]:
        return dict(self.items())

    def get_proxy(self) -> MergeAttrMapping:
        return MergeAttrMapping({
            k: (self._views[k] if k in self._views else self._data[k].raw_data)
            for k, v in self.items()
            if v and v.has_data()
        })

    def _get_buffers(
        self, filter_allocated: bool = False
    ) -> Generator[Tuple[str, BufferWrapper], None, None]:
        for k, buf in self._data.items():
            if not hasattr(buf, 'has_data') or (buf.has_data() and filter_allocated):
                continue
            yield k, buf

    def allocate_for_part(self, partition: Partition, roi: Optional[np.ndarray], lib=None) -> None:
        """
        allocate all BufferWrapper instances in this namespace.
        for pre-allocated buffers (i.e. aux data), only set shape and roi
        """
        for k, buf in self._get_buffers():
            buf.set_shape_partition(partition, roi)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate(lib=lib)

    def allocate_for_full(self, dataset: DataSet, roi: Optional[np.ndarray]) -> None:
        ds_partitions = [*dataset.get_partitions()]
        for k, buf in self._get_buffers():
            buf.set_shape_ds(dataset.shape, roi)
            buf.add_partitions(ds_partitions)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate()

    def set_view_for_dataset(self, dataset: DataSet) -> None:
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_dataset(dataset)

    def set_view_for_partition(self, partition: Partition) -> None:
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_partition(partition)

    def set_view_for_tile(self, partition: Partition, tile: DataTile) -> None:
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_tile(partition, tile)

    def set_contiguous_view_for_tile(self, partition: Partition, tile: DataTile) -> None:
        # .. versionadded:: 0.5.0
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_contiguous_view_for_tile(partition, tile)

    def flush(self, debug: bool = False) -> None:
        # .. versionadded:: 0.5.0
        for k, buf in self._get_buffers():
            buf.flush(debug=debug)

    def export(self) -> None:
        # .. versionadded:: 0.6.0
        for k, buf in self._get_buffers():
            buf.export()

    def set_view_for_frame(self, partition: Partition, tile: DataTile, frame_idx: int) -> None:
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_frame(partition, tile, frame_idx)

    def clear_views(self) -> None:
        self._views = {}


class UDFKwargsWrapper(UDFData):
    """
    Wrapper for UDF kwargs, used for slicing/viewing AUX data
    """
    def __init__(self, data: UDFKwargs):
        super().__init__(data)
        self._data = data
        self._views = {}

    def _get_buffers(
        self, filter_allocated: bool = False
    ) -> Generator[Tuple[str, AuxBufferWrapper], None, None]:
        for k, buf in self._data.items():
            if isinstance(buf, AuxBufferWrapper):
                if buf.has_data() and filter_allocated:
                    continue
                yield k, buf

    def new_for_partition(self, partition: Partition, roi: np.ndarray):
        for k, buf in self._get_buffers():
            self._data[k] = buf.new_for_partition(partition, roi)


@runtime_checkable
class UDFFrameMixin(Protocol):
    '''
    Implement :code:`process_frame` for per-frame processing.
    '''

    def process_frame(self, frame: np.ndarray):
        """
        Implement this method to process the data on a frame-by-frame manner.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        - `self.meta`      - meta data about the current operation and data set

        Parameters
        ----------
        frame : numpy.ndarray or cupy.ndarray
            A single frame or signal element from the dataset.
            The shape is the same as `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 2D, for spectra 1D etc.
        """
        raise NotImplementedError()


@runtime_checkable
class UDFTileMixin(Protocol):
    '''
    Implement :code:`process_tile` for per-tile processing.
    '''

    def process_tile(self, tile: np.ndarray):
        """
        Implement this method to process the data in a tiled manner.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        - `self.meta`      - meta data about the current operation and data set

        Parameters
        ----------
        tile : numpy.ndarray or cupy.ndarray
            A small number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()


@runtime_checkable
class UDFPartitionMixin(Protocol):
    '''
    Implement :code:`process_partition` for per-partition processing.
    '''

    def process_partition(self, partition: np.ndarray) -> None:
        """
        Implement this method to process the data partitioned into large
        (100s of MiB) partitions.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        - `self.meta`      - meta data about the current operation and data set

        Note
        ----
        Only use this method if you know what you are doing; especially if
        you are running a processing pipeline with multiple steps, or multiple
        processing pipelines at the same time, performance may be adversely
        impacted.

        Parameters
        ----------
        partition : numpy.ndarray or cupy.ndarray
            A large number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()


@runtime_checkable
class UDFPreprocessMixin(Protocol):
    '''
    Implement :code:`preprocess` to initialize the result buffers of a partition on the worker
    before the partition data is processed.

    .. versionadded:: 0.3.0
    '''

    def preprocess(self) -> None:
        """
        Implement this method to preprocess the result data for a partition.

        This can be useful to initialize arrays of
        :code:`dtype='object'` with the correct container types, for example.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        """
        raise NotImplementedError()


@runtime_checkable
class UDFPostprocessMixin(Protocol):
    '''
    Implement :code:`postprocess` to modify the resulf buffers of a partition on the worker
    after the partition data has been completely processed, but before it is returned to the
    main node for the final merging step.
    '''

    def postprocess(self) -> None:
        """
        Implement this method to postprocess the result data for a partition.

        This can be useful in combination with process_tile() to implement
        a postprocessing step that requires the reduced results for whole frames.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        """
        raise NotImplementedError()


@runtime_checkable
class UDFMergeAllMixin(Protocol):
    def merge_all(
                self, ordered_results: 'OrderedDict[Slice, MergeAttrMapping]'
            ) -> Mapping[str, 'nt.ArrayLike']:
        """
        Combine stack of ordered partial results `ordered_results` to form complete result.

        Combining can be more efficient than direct merging into a result buffer
        for cases where the results are not NumPy arrays.
        Currently this is only applicable for the
        :class:`libertem.executor.delayed.DelayedJobExecutor`
        where it provides an efficient pathway to construct Dask arrays from delayed UDF results.

        The input and the returned arrays are in flattened navigation dimension with ROI applied.

        Data available in this method:

        - `self.params` - the parameters of this UDF

        For UDFs with only :code:`kind='nav'` result buffers a default implementation
        is used automatically.

        Parameters
        ----------

        ordered_results
            Ordered dict mapping partition slice to UDF partial result

        Returns
        -------

        dict[buffername] -> array_like
            Dictionary mapping result buffer name to buffer content

        Note
        ----
        This function is running on the main node, which means `self.results`
        and `self.task_data` are not available.
        """
        raise NotImplementedError()


def _default_merge_all(udf, ordered_results: 'OrderedDict[Slice, MergeAttrMapping]'):
    if udf.requires_custom_merge:
        raise NotImplementedError(
            "Default merging only works for kind='nav' buffers. "
            "Please implement a suitable custom merge_all function."
        )
    result_chunks = defaultdict(lambda: [])
    for b in ordered_results.values():
        for key in b:
            result_chunks[key].append(getattr(b, key))

    result = {
        # checking above assures that we only have kind='nav'
        # where concatenation is the correct method.
        k: np.concatenate(val)
        for k, val in result_chunks.items()
    }
    return result


class UDFBase:
    '''
    Base class for UDFs with helper functions.
    '''
    def __init__(self, *args, **kwargs) -> None:
        # this should not execute - it is mostly for getting the typing right:
        self.params = UDFKwargsWrapper({})
        self.results = UDFData({})

    def get_task_data(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_result_buffers(self) -> Dict[str, BufferWrapper]:
        raise NotImplementedError()

    def allocate_for_part(self, partition: Partition, roi: Optional[np.ndarray]) -> None:
        for ns in [self.results]:
            ns.allocate_for_part(partition, roi, lib=self.xp)

    def allocate_for_full(self, dataset: DataSet, roi: Optional[np.ndarray]) -> None:
        for ns in [self.params, self.results]:
            ns.allocate_for_full(dataset, roi)

    def set_views_for_dataset(self, dataset: DataSet) -> None:
        for ns in [self.params]:
            ns.set_view_for_dataset(dataset)

    def set_views_for_partition(self, partition: Partition) -> None:
        for ns in [self.params, self.results]:
            ns.set_view_for_partition(partition)

    def set_views_for_tile(self, partition: Partition, tile: DataTile) -> None:
        for ns in [self.params, self.results]:
            ns.set_view_for_tile(partition, tile)

    def set_contiguous_views_for_tile(self, partition: Partition, tile: DataTile) -> None:
        # .. versionadded:: 0.5.0
        for ns in [self.params, self.results]:
            ns.set_contiguous_view_for_tile(partition, tile)

    def flush(self, debug: bool = False) -> None:
        # .. versionadded:: 0.5.0
        for ns in [self.params, self.results]:
            ns.flush(debug=debug)

    def set_views_for_frame(self, partition: Partition, tile: DataTile, frame_idx: int):
        for ns in [self.params, self.results]:
            ns.set_view_for_frame(partition, tile, frame_idx)

    def clear_views(self) -> None:
        for ns in [self.params, self.results]:
            ns.clear_views()

    def init_task_data(self) -> None:
        self.task_data = UDFData(self.get_task_data())

    def init_result_buffers(self, executor=None) -> None:
        """
        Create the UDFData instance containing BufferWrappers
        for results of this UDF. Same method used for
        Dataset- and Partition-sized buffers.

        The executor argument is provided on the main node
        only to allow modification of Dataset-sized buffers
        buffers into more specialized forms.

        #TODO use another mechanism to distinguish between
        dataset and parititon-sized buffers when initializing,
        such that we can modify buffers in both cases.
        """
        self.results = UDFData(self.get_result_buffers())
        if executor is not None:
            for name, buffer in self.results.items():
                new_buffer = executor.modify_buffer_type(buffer)
                self.results.set_buffer(name, new_buffer)

    def export_results(self) -> None:
        # .. versionadded:: 0.6.0.dev0
        self.results.export()

    def set_meta(self, meta: UDFMeta) -> None:
        self.meta = meta

    def set_slice(self, slice_: Slice) -> None:
        self.meta.slice = slice_

    def set_tile_idx(self, idx: int) -> None:
        self.meta.tiling_scheme_idx = idx

    def set_backend(self, backend: Backend) -> None:
        assert backend in self.get_backends()
        self._backend = backend

    def get_backends(self) -> BackendSpec:
        raise NotImplementedError()  # see impl in UDF.get_backends

    @property
    def xp(self):
        '''
        Compute back-end library to use.

        Generally, use :code:`self.xp` instead of :code:`np` to use NumPy or CuPy transparently

        .. versionadded:: 0.6.0
        '''
        # Implemented as property and not variable to avoid pickling issues
        # tests/udf/test_simple_udf.py::test_udf_pickle
        if self._backend == 'numpy' or self._backend == 'cuda':
            return np
        elif self._backend == 'cupy':
            # Re-importing should be fast, right?
            # Importing only here to avoid superfluous import
            import cupy
            # mocking for testing without actual CUDA device
            # import numpy as cupy
            return cupy
        else:
            raise ValueError(f"Backend name can be 'numpy', 'cuda' or 'cupy', got {self._backend}")

    def get_method(self) -> Literal['tile', 'frame', 'partition']:
        if hasattr(self, 'process_tile'):
            return 'tile'
        elif hasattr(self, 'process_frame'):
            return 'frame'
        elif hasattr(self, 'process_partition'):
            return 'partition'
        else:
            raise TypeError("UDF should implement one of the `process_*` methods")

    def _check_results(self, decl, arr, name):
        """
        Check results in `arr` for buffer `name` for consistency with the
        declaration in `decl`.

        1) All returned buffers need to be declared
        2) Private buffers can't be returned from `get_results`
        3) The `dtype` of `arr` needs to be compatible with the declared `dtype`
        """
        if name not in decl:
            raise UDFException(
                "buffer '%s' is not declared in `get_result_buffers` "
                "(hint: `self.buffer(..., use='result_only')`" % name
            )
        buf_decl = decl[name]
        if buf_decl.use == "private":
            raise UDFException("Don't return `use='private'` buffers from `get_results`")
        if np.dtype(arr.dtype).kind != np.dtype(buf_decl.dtype).kind:
            raise UDFException(
                "the returned ndarray '%s' has a different dtype kind (%s) "
                "than declared (%s)" % (
                    name, arr.dtype, buf_decl.dtype
                )
            )

    def get_results(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError()

    _default_merge_all = UDFMergeAllMixin.merge_all

    def _do_merge_all(self, ordered_results: 'OrderedDict[Slice, MergeAttrMapping]'):
        if isinstance(self, UDFMergeAllMixin):
            results_tmp = self.merge_all(ordered_results)
        else:
            # This will raise NotImplemementedError if preconditions for default merge
            # are not met
            results_tmp = _default_merge_all(self, ordered_results)

        if not set(results_tmp.keys()).issubset(set(self.results.keys())):
            raise ValueError(f'Returned result names from merge_all ({[*results_tmp.keys()]}) '
                            'are not contained within declared result buffer names '
                            f'({[*self.results.keys()]})')

        for key, value in results_tmp.items():
            # This SHOULD throw errors if sth doesn't match up about
            # buffer name, shape or dtype
            self.results.get_buffer(key).replace_array(value)

    def _do_get_results(self) -> Mapping[str, BufferWrapper]:
        results_tmp = self.get_results()
        decl = self.get_result_buffers()

        # include any results that were not explicitly included, but have non-private `use`:
        results_tmp.update({
            k: getattr(self.results, k)
            for k, v in decl.items()
            if k not in results_tmp and v.use is None
        })

        results_buffer_cls = {
            k: self.results.get_buffer(k).result_buffer_type()
            for k in results_tmp.keys()
        }

        # wrap numpy results into `ResultBuffer`s:
        results = {}
        for name, arr in results_tmp.items():
            self._check_results(decl, arr, name)
            buf_decl = decl[name]
            buf = results_buffer_cls[name](
                kind=buf_decl.kind, extra_shape=buf_decl.extra_shape,
                dtype=buf_decl.dtype,
                data=arr,
            )
            buf.set_shape_ds(self.meta.dataset_shape, self.meta.roi)
            results[name] = buf
        return results


class UDF(UDFBase):
    """
    The main user-defined functions interface. You can implement your functionality
    by overriding methods on this class.

    If you override `__init__`, please take care,
    as it is called multiple times during evaluation of a UDF. You can handle
    some pre-conditioning of parameters, but you also have to accept the results
    as input again.

    Arguments passed as `**kwargs` will be automatically available on `self.params`
    when running the UDF.

    Example
    -------

    >>> class MyUDF(UDF):
    ...     def __init__(self, param1, param2="def2", **kwargs):
    ...         param1 = int(param1)
    ...         if "param3" not in kwargs:
    ...             raise TypeError("missing argument param3")
    ...         super().__init__(param1=param1, param2=param2, **kwargs)

    Parameters
    ----------
    kwargs
        Input parameters. They are scattered to the worker processes and
        available as `self.params` from here on.

        Values can be `BufferWrapper` instances, which, when accessed via
        `self.params.the_key_here`, will automatically return a view corresponding
        to the current unit of data (frame, tile, partition).
    """
    USE_NATIVE_DTYPE = bool
    TILE_SIZE_BEST_FIT = TileSizeEnum.TILE_SIZE_BEST_FIT
    TILE_SIZE_MAX = np.inf
    TILE_DEPTH_DEFAULT = TileDepthEnum.TILE_DEPTH_DEFAULT
    TILE_DEPTH_MAX = np.inf

    def __init__(self, **kwargs: UDFKwarg) -> None:
        self._backend = 'numpy'  # default so that self.xp can always be used
        self._kwargs = kwargs
        self.params = UDFKwargsWrapper(kwargs)
        self.task_data = UDFData({})
        self.results = UDFData({})
        self._requires_custom_merge = None

    def copy(self) -> "UDF":
        return self.__class__(**self._kwargs)

    @classmethod
    def new_for_partition(
        cls,
        kwargs: Dict[str, Any],
        partition: Partition,
        roi: np.ndarray,
    ) -> "UDF":
        new_instance = cls(**kwargs)
        new_instance.params.new_for_partition(partition, roi)
        return new_instance

    def copy_for_partition(self, partition: Partition, roi: np.ndarray) -> "UDF":
        """
        create a copy of the UDF, specifically slicing aux data to the
        specified pratition and roi
        """
        return self.__class__.new_for_partition(self._kwargs, partition, roi)

    def get_task_data(self) -> Dict[str, Any]:
        """
        Initialize per-task data.

        Per-task data can be mutable. Override this function
        to allocate temporary buffers, or to initialize
        system resources.

        If you want to distribute static data, use
        parameters instead.

        Data available in this method:

        - `self.params` - the input parameters of this UDF
        - `self.meta` - relevant metadata, see :class:`UDFMeta` documentation.

        Returns
        -------
        dict
            Flat dict with string keys. Keys should
            be valid python identifiers, which allows
            access via `self.task_data.the_key_here`.
        """
        return {}

    def get_result_buffers(self) -> Dict[str, BufferWrapper]:
        """
        Return result buffer declaration.

        Values of the returned dict should be :class:`~libertem.common.buffers.BufferWrapper`
        instances, which, when accessed via :code:`self.results.key`,
        will automatically return a view corresponding to the
        current unit of data (frame, tile, partition).

        The values also need to be serializable via pickle.

        Data available in this method:

        - :code:`self.params` - the parameters of this UDF
        - :code:`self.meta` - relevant metadata, see :class:`UDFMeta` documentation.
          Please note that partition metadata will not be set when this method is
          executed on the head node.

        Returns
        -------
        dict
            Flat dict with string keys. Keys should
            be valid python identifiers, which allows
            access via `self.results.the_key_here`.
        """
        raise NotImplementedError()

    @property
    def requires_custom_merge(self):
        """
        Determine if buffers with :code:`kind != 'nav'` are present where
        the default merge doesn't work

        .. versionadded:: 0.5.0
        """
        if self._requires_custom_merge is None:
            buffers = self.get_result_buffers()
            self._requires_custom_merge = any(buffer.kind != 'nav' for buffer in buffers.values())
        return self._requires_custom_merge

    def merge(self, dest: MergeAttrMapping, src: MergeAttrMapping):
        """
        Merge a partial result `src` into the current global result `dest`.

        Data available in this method:

        - `self.params` - the parameters of this UDF

        Parameters
        ----------

        dest
            global results; you can access the ndarrays for each
            buffer name (from `get_result_buffers`) by attribute access
            (:code:`dest.your_buffer_name`)

        src
            results for a partition; you can access the ndarrays for each
            buffer name (from `get_result_buffers`) by attribute access
            (:code:`src.your_buffer_name`)

        Note
        ----
        This function is running on the main node, which means `self.results`
        and `self.task_data` are not available.
        """
        if self.requires_custom_merge:
            raise NotImplementedError(
                "Default merging only works for kind='nav' buffers. "
                "Please implement a suitable custom merge function."
            )
        for k in dest:
            check_cast(getattr(dest, k), getattr(src, k))
            getattr(dest, k)[:] = getattr(src, k)

    def get_results(self) -> Dict[str, np.ndarray]:
        """
        Get results, allowing a postprocessing step on the main node after
        a result has been merged. See also: :class:`UDFPostprocessMixin`.

        .. versionadded:: 0.7.0

        Note
        ----
        You should return all values as numpy arrays, they will be wrapped
        in `BufferWrapper` instances before they are returned to the user.

        See the :ref:`udf final post processing` section in the documentation for
        details and examples.

        Returns
        -------

        results : dict
            A `dict` containing the final post-processed results.

        """
        for k in self.results.keys():
            buf = self.results.get_buffer(k)
            if buf.use == "result_only":
                raise UDFException(
                    "don't know how to set use='result_only' buffer '%s';"
                    " please implement `get_results`" % k
                )
        decls = self.get_result_buffers()
        return {
            k: getattr(self.results, k)
            for k in self.results.keys()
            if decls[k].use != "private"
        }

    def get_preferred_input_dtype(self) -> "nt.DTypeLike":
        '''
        Override this method to specify the preferred input dtype of the UDF.

        The default is :code:`float32` since most numerical processing tasks
        perform best with this dtype, namely dot products.

        The back-end uses this preferred input dtype in combination with the
        dataset`s native dtype to determine the input dtype using
        :func:`numpy.result_type`. That means :code:`float` data in a dataset
        switches the dtype to :code:`float` even if this method returns an
        :code:`int` dtype. :code:`int32` or wider input data would switch from
        :code:`float32` to :code:`float64`, and complex data in the dataset will
        switch the input dtype kind to :code:`complex`, following the NumPy
        casting rules.

        In case your UDF only works with specific input dtypes, it should throw
        an error or warning if incompatible dtypes are used, and/or implement a
        meaningful conversion in your UDF's :code:`process_<...>` routine.

        If you prefer to always use the dataset's native dtype instead of
        floats, you can override this method to return
        :attr:`UDF.USE_NATIVE_DTYPE`, which is currently identical to
        :code:`numpy.bool` and behaves as a neutral element in
        :func:`numpy.result_type`.

        .. versionadded:: 0.4.0
        '''
        return np.float32

    def get_tiling_preferences(self) -> TilingPreferences:
        """
        Configure tiling preferences. Return a dictionary with the
        following keys:

        - "depth": number of frames/frame parts to stack on top of each other
        - "total_size": total size of a tile in bytes

        .. versionadded:: 0.6.0
        """
        return {
            "depth": UDF.TILE_DEPTH_DEFAULT,
            "total_size": UDF.TILE_SIZE_MAX,
        }

    def get_backends(self) -> BackendSpec:
        # TODO see interaction with C++ CUDA modules requires a different type than CuPy
        '''
        Signal which computation back-ends the UDF can use.

        :code:`numpy` is the default CPU-based computation.

        :code:`cuda` is CUDA-based computation without CuPy.

        :code:`cupy` is CUDA-based computation through CuPy.

        .. versionadded:: 0.6.0

        Returns
        -------

        backend : Iterable[str]
            An iterable containing possible values :code:`numpy` (default), :code:`'cuda'` and
            :code:`cupy`
        '''
        return ('numpy', )

    def cleanup(self) -> None:  # FIXME: name? implement cleanup as context manager somehow?
        pass

    def buffer(
        self,
        kind: BufferKind,
        extra_shape: Tuple[int, ...] = (),
        dtype: "nt.DTypeLike" = "float32",
        where: BufferLocation = None,
        use: Optional[BufferUse] = None,
    ) -> BufferWrapper:
        '''
        Use this method to create :class:`~ libertem.common.buffers.BufferWrapper` objects
        in :meth:`get_result_buffers`.

        Parameters
        ----------
        kind : "nav", "sig" or "single"
            The abstract shape of the buffer, corresponding either to the navigation
            or the signal dimensions of the dataset, or a single value.

        extra_shape : optional, tuple of int or a Shape object
            You can specify additional dimensions for your data. For example, if
            you want to store 2D coords, you would specify :code:`(2,)` here.
            If this is specified as a Shape object, it is converted to a tuple first.

        dtype : string or numpy dtype
            The dtype of this buffer

        where : string or None
            :code:`None` means NumPy array, specify :code:`'device'` to use a device buffer
            (for example on a CUDA device)

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
        '''
        if use is not None and use.lower() == "result_only":
            return PlaceholderBufferWrapper(kind, extra_shape, dtype, use=use)
        return BufferWrapper(kind, extra_shape, dtype, where, use=use)

    @classmethod
    def aux_data(cls, data, kind, extra_shape=(), dtype="float32"):
        """
        Use this method to create auxiliary data. Auxiliary data should
        have a shape like `(dataset.shape.nav, extra_shape)` and on access,
        an appropriate view will be created. For example, if you access
        aux data in `process_frame`, you will get the auxiliary data for
        the current frame you are processing.

        Example
        -------

        We create a UDF to demonstrate the behavior:

        >>> class MyUDF(UDF):
        ...     def get_result_buffers(self):
        ...         # Result buffer for debug output
        ...         return {'aux_dump': self.buffer(kind='nav', dtype='object')}
        ...
        ...     def process_frame(self, frame):
        ...         # Extract value of aux data for demonstration
        ...         self.results.aux_dump[:] = str(self.params.aux_data[:])
        ...
        >>> # for each frame, provide three values from a sequential series:
        >>> aux1 = MyUDF.aux_data(
        ...     data=np.arange(np.prod(dataset.shape.nav) * 3, dtype=np.float32),
        ...     kind="nav", extra_shape=(3,), dtype="float32"
        ... )
        >>> udf = MyUDF(aux_data=aux1)
        >>> res = ctx.run_udf(dataset=dataset, udf=udf)

        process_frame for frame (0, 7) received a view of aux_data with values [21., 22., 23.]:

        >>> res['aux_dump'].data[0, 7]
        '[21. 22. 23.]'
        """
        buf = AuxBufferWrapper(kind, extra_shape, dtype)
        buf.set_buffer(data)
        return buf


class NoOpUDF(UDF):
    '''
    A UDF that does nothing and returns nothing.

    This is useful for testing.

    Parameters
    ----------
    preferred_input_dtype : numpy.dtype
        Perform dtype conversion. By default, this is :attr:`UDF.USE_NATIVE_DTYPE`.
    '''
    def __init__(self, preferred_input_dtype=UDF.USE_NATIVE_DTYPE) -> None:
        super().__init__(preferred_input_dtype=preferred_input_dtype)

    def process_tile(self, tile):
        '''
        Do nothing.
        '''
        pass

    def get_result_buffers(self):
        '''
        No result buffers.
        '''
        return {}

    def get_preferred_input_dtype(self):
        '''
        Return the value passed in the constructor.
        '''
        return self.params.preferred_input_dtype


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError(f"Unsafe automatic casting from {fromvar.dtype} to {tovar.dtype}")


class UDFParams:
    def __init__(
        self,
        kwargs: List[dict],
        roi: Optional[np.ndarray],
        corrections: Optional[CorrectionSet],
        tiling_scheme: TilingScheme,
    ):
        """
        Container class for UDF parameters for multiple UDFs

        Parameters
        ----------
        kwargs : List[dict]
            List of kwargs which can be used to re-create the UDF instances
            that were passed to `run_for_dataset`
        roi
            Boolean array to select parts of the navigation space
        corrections
            Corrections to apply
        """
        self._kwargs = kwargs
        self._roi = roi
        self._corrections = corrections
        self._tiling_scheme = tiling_scheme

    @classmethod
    def from_udfs(
        cls,
        udfs: Iterable[UDF],
        roi: Optional[np.ndarray],
        corrections: Optional[CorrectionSet],
        tiling_scheme: TilingScheme,
    ):
        kwargs = [udf._kwargs for udf in udfs]
        return cls(kwargs, roi, corrections, tiling_scheme)

    @property
    def roi(self):
        return self._roi

    @property
    def corrections(self):
        return self._corrections

    @property
    def kwargs(self):
        return self._kwargs

    @property
    def tiling_scheme(self):
        return self._tiling_scheme


class Task:
    """
    A computation on a partition. Inherit from this class and implement ``__call__``
    for your specific computation.

    .. versionchanged:: 0.4.0
        Moved from libertem.job.base to libertem.udf.base as part of Job API deprecation
    """

    def __init__(self, partition, idx):
        self.partition = partition
        self.idx = idx

    def get_locations(self):
        return self.partition.get_locations()

    def get_resources(self) -> ResourceDef:
        '''
        Specify the resources that a Task will use.

        The resources are designed to work with resource tags in Dask clusters:
        See https://distributed.dask.org/en/latest/resources.html

        The resources allow scheduling of CPU-only compute, CUDA-only compute,
        (CPU or CUDA) hybrid compute, and service tasks in such a way that all
        resources are used without oversubscription. Furthermore, they
        distinguish if the given resource can be accessed with a transparent
        NumPy ndarray interface -- namely if CuPy is installed to access CUDA
        resources.

        Each CPU worker gets one CPU, one compute and one ndarray resource
        assigned. Each CUDA worker gets one CUDA and one compute resource
        assigned. If CuPy is installed, it additionally gets an ndarray resource
        assigned. A service worker doesn't get any resources assigned.

        A CPU-only task consumes one CPU, one ndarray and one compute resource,
        i.e. it will be scheduled only on CPU workers. A CUDA-only task consumes
        one CUDA, one compute and possibly an ndarray resource, i.e. it will
        only be scheduled on CUDA workers. A hybrid task that can run on both
        CPU or CUDA using a transparent ndarray interface only consumes a
        compute and an ndarray resource, i.e. it can be scheduled on CPU workers
        or CUDA workers with CuPy. A service task doesn't request any resources
        and can therefore run on CPU, CUDA and service workers.

        Which device a hybrid task uses is only decided at runtime by the
        environment variables that are set on each worker process.

        That way, CPU-only and CUDA-only tasks can run in parallel without
        oversubscription, and hybrid tasks use whatever compute workers are free
        at the time. Furthermore, it allows using CUDA without installing CuPy.
        See https://github.com/LiberTEM/LiberTEM/pull/760 for detailed
        discussion.

        Compute tasks will not run on service workers, so that these can serve
        shorter service tasks with lower latency.

        At the moment, workers only get a single "item" of each resource
        assigned since we run one process per CPU. Requesting more of a resource
        than any of the workers has causes a :class:`RuntimeError`, including
        requesting a resource that is not available at all.

        For that reason one has to make sure that the workers are set up with
        the correct resources and matching environment.
        :meth:`libertem.utils.devices.detect`,
        :meth:`libertem.executor.dask.cluster_spec` and
        :meth:`libertem.executor.dask.DaskJobExecutor.make_local` can be used to
        set up a local cluster correctly.

        .. versionadded:: 0.6.0
        '''
        # default: run only on CPU and do computation
        # No CUDA support for the deprecated Job interface
        return {'CPU': 1, 'compute': 1, 'ndarray': 1}

    def __call__(self, params: UDFParams, env: Environment):
        raise NotImplementedError()


def _get_canonical_backends(backends: Optional[BackendSpec]) -> Set[Backend]:
    """
    Convert from either an iterable of backends or a simple string form into a
    canonical form of Set[str]
    """
    if backends is None:
        return set()
    if isinstance(backends, str):
        backends = (backends,)
    return set(backends)


def get_resources_for_backends(
    udf_backends: List[BackendSpec],
    user_backends: Optional[BackendSpec]
) -> ResourceDef:
    """
    Find the resource definition that is appropriate for the backends specified
    by the UDFs and the user. This is the intersection between the backend specified
    in each UDF, and the backend that was sepected by the user.

    Parameters
    ----------
    udf_backends
        The backends specified by each UDF that we want to run

    user_backends
        The backends specified by the user

    See :meth:`Task.get_resources` for details.
    """
    udf_backends_canonical = [
        _get_canonical_backends(bs)
        for bs in udf_backends
    ]
    user_backends_canonical = _get_canonical_backends(user_backends)

    needs_cuda = 0
    needs_cpu = 0
    needs_ndarray = 0

    # Limit to externally specified backends
    for backend_set in udf_backends_canonical:
        if user_backends_canonical:
            backends = set(user_backends_canonical).intersection(backend_set)
        else:
            backends = backend_set
        needs_cuda += 'numpy' not in backends
        needs_cpu += ('cuda' not in backends) and ('cupy' not in backends)
        needs_ndarray += 'cuda' not in backends
    if needs_cuda and needs_cpu:
        raise ValueError(
            "There is no common supported UDF backend (have: %r, limited to %r)"
            % (udf_backends, user_backends_canonical)
        )
    result: ResourceDef = {'compute': 1}
    if needs_cpu:
        result['CPU'] = 1
    if needs_cuda:
        result['CUDA'] = 1
    if needs_ndarray:
        result['ndarray'] = 1
    return result


class UDFTask(Task):
    def __init__(
        self,
        partition: Partition,
        idx: int,
        udf_classes: List[Type[UDF]],
        udf_backends: List[BackendSpec],
        backends: Optional[BackendSpec],
        runner_cls: Type['UDFRunner'],
    ):
        """
        A computation for a single partition. The parameters that stay the same
        for the whole dataset are excluded here and supplied by the executor in
        __call__.

        Parameters
        ----------
        partition : Partition
            The partition to work on
        idx : int
            the index of the task, used to identify results (?)
        udf_classes : List[Type[UDF]]
            The UDFs to run
        backends : List[str]
            The specified backends we want to run on
        """
        super().__init__(partition=partition, idx=idx)
        self._backends = backends
        self._udf_classes = udf_classes
        self._udf_backends = udf_backends
        self._runner_cls = runner_cls

    def __call__(self, params: UDFParams, env: Environment) -> Tuple[UDFData, ...]:
        udfs = [
            cls.new_for_partition(kwargs, self.partition, params.roi)
            for cls, kwargs in zip(self._udf_classes, params.kwargs)
        ]
        return self._runner_cls(udfs).run_for_partition(
            self.partition, params, env,
        )

    def get_resources(self) -> ResourceDef:
        """
        Intersection of resources of all UDFs, throws if empty.

        See docstring of super class for details.
        """
        return get_resources_for_backends(self._udf_backends, user_backends=self._backends)

    def __repr__(self):
        return f"<UDFTask {self._udf_classes!r}>"


class UDFRunner:
    @staticmethod
    def _apply_part_result(udfs, damage, part_results, task):
        for results, udf in zip(part_results, udfs):
            udf.set_views_for_partition(task.partition)
            udf.merge(
                dest=udf.results.get_proxy(),
                src=results.get_proxy()
            )
        v = damage.get_view_for_partition(task.partition)
        v[:] = True

    @staticmethod
    def _make_udf_result(udfs: Iterable[UDF], damage: BufferWrapper) -> "UDFResults":
        for udf in udfs:
            udf.clear_views()
        return UDFResults(
            buffers=tuple(
                udf._do_get_results()
                for udf in udfs
            ),
            damage=damage
        )

    def __init__(self, udfs: List[UDF], debug: bool = False):
        self._udfs = udfs
        self._debug = debug
        self._pool = ThreadPoolExecutor(max_workers=4)

    @classmethod
    def inspect_udf(
        cls,
        udf: UDF,
        dataset: DataSet,
        roi: Optional[np.ndarray] = None,
    ) -> Dict[str, BufferWrapper]:
        """
        Return result buffer declarations for a given UDF/DataSet/roi combination
        """
        runner = cls([udf])
        meta = UDFMeta(
            partition_slice=None,
            dataset_shape=dataset.shape,
            roi=None,
            dataset_dtype=dataset.dtype,
            input_dtype=runner._get_dtype(
                dataset.dtype,
                corrections=None,
            ),
            corrections=None,
        )

        udf = udf.copy()
        udf.set_meta(meta)
        buffers = udf.get_result_buffers()
        for buf in buffers.values():
            buf.set_shape_ds(dataset.shape, roi)
        return buffers

    @classmethod
    def dry_run(cls, udfs, dataset, roi=None):
        """
        Return result buffers for a given UDF/DataSet/roi combination
        exactly as running the UDFs would, just skipping execution and
        merging of the processing tasks.

        This can be used to create an empty result to initialize live plots
        before running an UDF.
        """
        runner = cls(udfs)
        executor = InlineJobExecutor()
        res = runner.run_for_dataset(
            dataset=dataset,
            executor=executor,
            roi=roi,
            dry=True
        )
        return res

    def _get_dtype(
        self,
        dtype: "nt.DTypeLike",
        corrections: Optional[CorrectionSet]
    ) -> "nt.DTypeLike":
        if corrections is not None and corrections.have_corrections():
            tmp_dtype = np.result_type(np.float32, dtype)
        else:
            tmp_dtype = np.dtype(dtype)
        for udf in self._udfs:
            tmp_dtype = np.result_type(
                udf.get_preferred_input_dtype(),
                tmp_dtype
            )
        return tmp_dtype

    def _init_udfs(
        self,
        numpy_udfs: List[UDF],
        cupy_udfs: List[UDF],
        partition: Partition,
        roi: Optional[np.ndarray],
        corrections: CorrectionSet,
        device_class,
        env: Environment,
        tiling_scheme: TilingScheme,
    ) -> Tuple[UDFMeta, "nt.DTypeLike"]:
        dtype = self._get_dtype(partition.dtype, corrections)
        meta = UDFMeta(
            partition_slice=partition.slice.adjust_for_roi(roi),
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=partition.dtype,
            input_dtype=dtype,
            tiling_scheme=tiling_scheme,
            corrections=corrections,
            device_class=device_class,
            threads_per_worker=env.threads_per_worker,
        )
        for udf in numpy_udfs:
            backends = udf.get_backends()
            if device_class == 'cuda':
                # Warnings etc handled in _udf_lists()
                if 'cuda' in backends:
                    udf.set_backend('cuda')
                elif 'numpy' in backends:  # fallback
                    udf.set_backend('numpy')
                else:
                    # Should be covered in _udf_lists(), but just to be sure.
                    raise ValueError("Can't run {udf} with backends {backends} on a CUDA worker.")
            else:
                udf.set_backend('numpy')
        if device_class == 'cpu':
            # Should be covered in _udf_lists(), but just to be sure.
            assert not cupy_udfs
        for udf in cupy_udfs:
            udf.set_backend('cupy')
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.get_method()  # validate that one of the `process_*` methods is implemented
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_part(partition, roi)
            udf.init_task_data()
            # TODO: preprocess doesn't have access to the tiling scheme - is this ok?
            if isinstance(udf, UDFPreprocessMixin):
                udf.clear_views()
                udf.preprocess()

        for udf in udfs:
            udf.set_meta(meta)
        return (meta, dtype)

    def _run_tile(
        self,
        udfs: List[UDF],
        partition: Partition,
        tile: DataTile,
        device_tile: DataTile,
        roi: Optional[np.ndarray],
    ) -> None:
        for udf in udfs:
            if isinstance(udf, UDFTileMixin):
                udf.set_contiguous_views_for_tile(partition, tile)
                udf.set_slice(tile.tile_slice)
                udf.set_tile_idx(tile.scheme_idx)
                udf.process_tile(device_tile)
            elif isinstance(udf, UDFFrameMixin):
                tile_slice = tile.tile_slice
                for frame_idx, frame in enumerate(device_tile):
                    frame_slice = Slice(
                        origin=(tile_slice.origin[0] + frame_idx,) + tile_slice.origin[1:],
                        shape=Shape((1,) + tuple(tile_slice.shape)[1:],
                                    sig_dims=tile_slice.shape.sig.dims),
                    )
                    # Internal checks for dataset consistency
                    assert frame.shape == tuple(partition.shape.sig)
                    udf.set_slice(frame_slice)
                    udf.set_views_for_frame(partition, tile, frame_idx)
                    udf.process_frame(frame)
            elif isinstance(udf, UDFPartitionMixin):
                # Internal checks for dataset consistency
                assert partition.slice.adjust_for_roi(roi) == tile.tile_slice
                udf.set_views_for_tile(partition, tile)
                udf.set_slice(tile.tile_slice)
                udf.process_partition(device_tile)

    def _run_udfs(
        self,
        numpy_udfs: List[UDF],
        cupy_udfs: List[UDF],
        partition: Partition,
        tiling_scheme: TilingScheme,
        roi: Optional[np.ndarray],
        dtype,
    ) -> None:
        # FIXME pass information on target location (numpy or cupy)
        # to dataset so that is can already move it there.
        # In the future, it might even decode data on the device instead of CPU
        tiles = partition.get_tiles(
            tiling_scheme=tiling_scheme,
            roi=roi, dest_dtype=dtype,
        )

        if cupy_udfs:
            xp = cupy_udfs[0].xp

        for tile in tiles:
            self._run_tile(numpy_udfs, partition, tile, tile, roi=roi)
            if cupy_udfs:
                # Work-around, should come from dataset later
                device_tile = xp.asanyarray(tile)
                self._run_tile(cupy_udfs, partition, tile, device_tile, roi=roi)

    def _wrapup_udfs(
        self,
        numpy_udfs: List[UDF],
        cupy_udfs: List[UDF],
        partition: Partition
    ) -> None:
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.flush(self._debug)
            if isinstance(udf, UDFPostprocessMixin):
                udf.clear_views()
                udf.postprocess()

            udf.cleanup()
            udf.clear_views()
            udf.export_results()

        if self._debug:
            try:
                cloudpickle.loads(cloudpickle.dumps(partition))
            except TypeError:
                raise TypeError("could not pickle partition")
            try:
                cloudpickle.loads(cloudpickle.dumps(
                    [u._do_get_results() for u in udfs]
                ))
            except TypeError:
                raise TypeError("could not pickle results")

    def _udf_lists(self, device_class: DeviceClass) -> Tuple[List[UDF], List[UDF]]:
        numpy_udfs = []
        cupy_udfs = []
        if device_class == 'cuda':
            for udf in self._udfs:
                backends = udf.get_backends()
                if 'cuda' in backends:
                    numpy_udfs.append(udf)
                elif 'cupy' in backends:
                    cupy_udfs.append(udf)
                elif 'numpy' in backends:
                    warnings.warn(f"UDF {udf} backends are {backends}, recommended on CUDA are "
                            "'cuda' and 'cupy'.", RuntimeWarning)
                    numpy_udfs.append(udf)
                else:
                    raise RuntimeError(
                        f"UDF {udf} backends are {backends}, supported on CUDA are "
                        "'cuda' and 'cupy', as well as 'numpy' as a compatibility fallback."
                    )
        elif device_class == 'cpu':
            for udf in self._udfs:
                backends = udf.get_backends()
                if 'numpy' not in backends:
                    raise RuntimeError(
                        f"UDF {udf} backends are {backends}, supported on CPU is 'numpy'."
                    )
            numpy_udfs = self._udfs
        else:
            raise ValueError(f"Unknown device class {device_class}, "
                "supported are 'cpu' and 'cuda'")
        return (numpy_udfs, cupy_udfs)

    def run_for_partition(
        self,
        partition: Partition,
        params: UDFParams,
        env: Environment
    ) -> Tuple[UDFData, ...]:
        roi = params.roi
        corrections = params.corrections
        with env.enter():
            try:
                previous_id = None
                device_class = get_device_class()
                # numpy_udfs and cupy_udfs contain references to the objects in
                # self._udfs
                numpy_udfs, cupy_udfs = self._udf_lists(device_class)
                # Will only be populated if actually on CUDA worker
                # and any UDF supports 'cupy' (and not 'cuda')
                if cupy_udfs:
                    # Avoid importing if not used
                    import cupy
                    device = get_use_cuda()
                    previous_id = cupy.cuda.Device().id
                    cupy.cuda.Device(device).use()
                (meta, dtype) = self._init_udfs(
                    numpy_udfs, cupy_udfs, partition, roi, corrections, device_class, env,
                    params.tiling_scheme,
                )
                partition.set_corrections(corrections)
                self._run_udfs(numpy_udfs, cupy_udfs, partition, params.tiling_scheme, roi, dtype)
                self._wrapup_udfs(numpy_udfs, cupy_udfs, partition)
            finally:
                if previous_id is not None:
                    cupy.cuda.Device(previous_id).use()
            # Make sure results are in the same order as the UDFs
            return tuple(udf.results for udf in self._udfs)

    def _debug_task_pickling(self, tasks: List[UDFTask]) -> None:
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps(tasks))

    def _check_preconditions(self, dataset: DataSet, roi: Optional[np.ndarray]) -> None:
        if roi is not None and prod(roi.shape) != prod(dataset.shape.nav):
            raise ValueError(
                "roi: incompatible shapes: {} (roi) vs {} (dataset)".format(
                    roi.shape, dataset.shape.nav
                )
            )

    def _prepare_run_for_dataset(
        self,
        dataset: DataSet,
        executor: JobExecutor,
        roi: Optional[np.ndarray],
        corrections: Optional[CorrectionSet],
        backends: Optional[BackendSpec],
        dry: bool,
    ) -> Tuple[List[UDFTask], UDFParams]:
        self._check_preconditions(dataset, roi)
        meta = UDFMeta(
            partition_slice=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=dataset.dtype,
            input_dtype=self._get_dtype(dataset.dtype, corrections),
            corrections=corrections,
        )
        for udf in self._udfs:
            udf.set_meta(meta)
            udf.init_result_buffers(executor=executor)
            udf.allocate_for_full(dataset, roi)

            if isinstance(udf, UDFPreprocessMixin):
                udf.set_views_for_dataset(dataset)
                udf.preprocess()
        neg = Negotiator()
        # FIXME take compute backend into consideration as well
        # Other boundary conditions when moving input data to device
        # FIXME: approximate partition shape here
        partition = next(dataset.get_partitions())
        tiling_scheme = neg.get_scheme(
            udfs=self._udfs,
            approx_partition_shape=partition.shape,
            dataset=dataset,
            read_dtype=meta.input_dtype,
            roi=roi,
            corrections=corrections,
        )
        params = UDFParams.from_udfs(
            udfs=self._udfs,
            roi=roi,
            corrections=corrections,
            tiling_scheme=tiling_scheme,
        )
        if dry:
            tasks = []
        else:
            tasks = list(self._make_udf_tasks(dataset, roi, backends))
        return (tasks, params)

    def run_for_dataset(
        self,
        dataset: DataSet,
        executor: JobExecutor,
        roi: Optional[np.ndarray] = None,
        progress: bool = False,
        corrections: Optional[CorrectionSet] = None,
        backends: Optional[BackendSpec] = None,
        dry: bool = False
    ) -> "UDFResults":
        for res in self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry,
            iterate=False
        ):
            pass
        return res

    def results_for_dataset_sync(
        self,
        dataset: DataSet,
        executor: JobExecutor,
        roi: Optional[np.ndarray] = None,
        progress: bool = False,
        corrections: Optional[CorrectionSet] = None,
        backends: Optional[BackendSpec] = None,
        dry: bool = False,
    ) -> Iterable[Tuple[Tuple[UDFData, ...], TaskProtocol]]:
        tasks, params = self._prepare_run_for_dataset(
            dataset, executor, roi, corrections, backends, dry
        )
        cancel_id = str(uuid.uuid4())
        self._debug_task_pickling(tasks)

        executor = executor.ensure_sync()

        try:
            if progress:
                from tqdm import tqdm
                t = tqdm(total=len(tasks))
            with executor.scatter(params) as params_handle:
                if tasks:
                    for res in executor.run_tasks(
                        tasks,
                        params_handle,
                        cancel_id,
                    ):
                        if progress:
                            t.update(1)
                        yield res
        finally:
            if progress:
                t.close()

    def run_for_dataset_sync(
        self,
        dataset: DataSet,
        executor: JobExecutor,
        roi: Optional[np.ndarray] = None,
        progress: bool = False,
        corrections: Optional[CorrectionSet] = None,
        backends: Optional[BackendSpec] = None,
        dry: bool = False,
        iterate: bool = True
    ) -> Generator["UDFResults", None, None]:
        executor = executor.ensure_sync()
        result_iter = self.results_for_dataset_sync(
            dataset=dataset,
            executor=executor,
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry,
        )
        damage = BufferWrapper(kind='nav', dtype=bool)
        damage.set_shape_ds(dataset.shape, roi)
        damage.allocate()
        any_result = False
        for part_results, task in result_iter:
            any_result = True
            self._apply_part_result(
                udfs=self._udfs,
                damage=damage,
                part_results=part_results,
                task=task
            )
            if iterate:
                yield self._make_udf_result(
                    udfs=self._udfs,
                    damage=damage
                )
        if not any_result or not iterate:
            yield self._make_udf_result(
                udfs=self._udfs,
                damage=damage
            )

    async def run_for_dataset_async(
        self,
        dataset: DataSet,
        executor: JobExecutor,
        cancel_id,
        roi: Optional[np.ndarray] = None,
        corrections: Optional[CorrectionSet] = None,
        backends: Optional[BackendSpec] = None,
        progress: bool = False,
        dry: bool = False,
        iterate: bool = True,
    ) -> AsyncGenerator["UDFResults", None]:
        gen = self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry,
            iterate=iterate,
        )

        async for res in async_generator_eager(gen, pool=self._pool):
            yield res

    def _roi_for_partition(self, roi, partition: Partition) -> np.ndarray:
        return roi.reshape(-1)[partition.slice.get(nav_only=True)]

    def _make_udf_tasks(
        self,
        dataset: DataSet,
        roi: Optional[np.ndarray],
        backends: Optional[BackendSpec]
    ) -> Generator[UDFTask, None, None]:
        udf_backends = [
            udf.get_backends()
            for udf in self._udfs
        ]
        for idx, partition in enumerate(dataset.get_partitions()):
            if roi is not None:
                roi_for_part = self._roi_for_partition(roi, partition)
                if np.count_nonzero(roi_for_part) == 0:
                    # roi is empty for this partition, ignore
                    continue
            udf_classes = [
                udf.__class__
                for udf in self._udfs
            ]
            tasks = UDFTask(
                partition=partition, idx=idx, udf_classes=udf_classes,
                udf_backends=udf_backends,
                backends=backends,
                runner_cls=self.__class__,
            )
            yield tasks


UDFResultDict = Mapping[str, BufferWrapper]


class UDFResults:
    '''
    Container class to combine UDF results with additional information.

    This class allows to return additional information from UDF execution
    together with UDF result buffers. This is currently used to pass
    "damage" information when running UDFs as an iterator using
    :meth:`libertem.api.Context.run_udf_iter`. "Damage" is
    a map of the nav space that is set to :code:`True`
    for all positions that have already been processed.

    .. versionadded:: 0.7.0

    Parameters
    ----------

    buffers
        Iterable containing the result buffer dictionaries for each of the UDFs being executed

    damage : BufferWrapper
        :class:`libertem.common.buffers.BufferWrapper` of :code:`kind='nav'`, :code:`dtype=bool`.
        It is set to :code:`True` for all positions in nav space that have been processed already.
    '''
    def __init__(self, buffers: Iterable[UDFResultDict], damage: BufferWrapper):
        self.buffers = tuple(buffers)
        self.damage = damage
