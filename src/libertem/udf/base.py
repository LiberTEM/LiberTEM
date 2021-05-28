from typing import Dict, Optional
import warnings
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor

import cloudpickle
import numpy as np

from libertem.warnings import UseDiscouragedWarning
from libertem.exceptions import UDFException
from libertem.common.buffers import (
    BufferWrapper, AuxBufferWrapper, PlaceholderBufferWrapper, PreallocBufferWrapper,
)
from libertem.common import Shape, Slice
from libertem.io.dataset.base import (
    TilingScheme, Negotiator, Partition, DataSet, get_coordinates
)
from libertem.corrections import CorrectionSet
from libertem.common.backend import get_use_cuda, get_device_class
from libertem.utils.async_utils import async_generator_eager
from libertem.executor.inline import InlineJobExecutor


log = logging.getLogger(__name__)


class UDFMeta:
    """
    UDF metadata. Makes all relevant metadata accessible to the UDF. Can be different
    for each task/partition.

    .. versionchanged:: 0.4.0
        Added distinction of dataset_dtype and input_dtype

    .. versionchanged:: 0.6.0
        Information on compute backend, corrections, coordinates and tiling scheme added
    """

    def __init__(self, partition_shape: Shape, dataset_shape: Shape, roi: np.ndarray,
                 dataset_dtype: np.dtype, input_dtype: np.dtype, tiling_scheme: TilingScheme = None,
                 tiling_index: int = 0, corrections=None, device_class: str = None,
                 threads_per_worker: Optional[int] = None):
        self._partition_shape = partition_shape
        self._dataset_shape = dataset_shape
        self._dataset_dtype = dataset_dtype
        self._input_dtype = input_dtype
        self._tiling_scheme = tiling_scheme
        self._tiling_index = tiling_index
        if device_class is None:
            device_class = 'cpu'
        self._device_class = device_class
        if roi is not None:
            roi = roi.reshape(dataset_shape.nav)
        self._roi = roi
        self._slice = None
        self._coordinates = None
        self._cached_coordinates = {}
        if corrections is None:
            corrections = CorrectionSet()
        self._corrections = corrections
        self._threads_per_worker = threads_per_worker

    @property
    def slice(self) -> Slice:
        """
        Slice : A :class:`~libertem.common.slice.Slice` instance that describes the location
                within the dataset with navigation dimension flattened and reduced to the ROI.
        """
        return self._slice

    @slice.setter
    def slice(self, new_slice: Slice):
        self._slice = new_slice

    @property
    def partition_shape(self) -> Shape:
        """
        Shape : The shape of the partition this UDF currently works on.
                If a ROI was applied, the shape will be modified accordingly.
        """
        return self._partition_shape

    @property
    def dataset_shape(self) -> Shape:
        """
        Shape : The original shape of the whole dataset, not influenced by the ROI
        """
        return self._dataset_shape

    @property
    def tiling_scheme(self) -> Shape:
        """
        TilingScheme : the tiling scheme that was negotiated

        .. versionadded:: 0.6.0
        """
        return self._tiling_scheme

    @property
    def roi(self) -> np.ndarray:
        """
        numpy.ndarray : Boolean array which limits the elements the UDF is working on.
                     Has a shape of :attr:`dataset_shape.nav`.
        """
        return self._roi

    @property
    def dataset_dtype(self) -> np.dtype:
        """
        numpy.dtype : Native dtype of the dataset
        """
        return self._dataset_dtype

    @property
    def input_dtype(self) -> np.dtype:
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
        # Check if key is present in the cached_coordinates, generate the coords otherwise
        roi_key = "None" if self._roi is None else tuple(map(tuple, self._roi))
        key = (self._slice, tuple(self._dataset_shape), roi_key)
        coords = self._cached_coordinates.get(key)
        if coords is None:
            coords = get_coordinates(self._slice, self._dataset_shape, self._roi)
            self._cached_coordinates.update({key: coords})
            self._coordinates = coords
        return coords

    @property
    def threads_per_worker(self) -> Optional[int]:
        """
        int or None : number of threads that a UDF is allowed to use in the `process_*` method.
                      For numba, pyfftw, OMP, MKL, OpenBLAS, this limit is set automatically;
                      this property can be used for other cases, like manually creating
                      thread pools.
                      None means no limit is set, and the UDF can use any number of threads
                      it deems necessary (should be limited to system limits, of course).

        See also: :func:`libertem.utils.threading.set_num_threads`

        .. versionadded:: 0.7.0
        """
        return self._threads_per_worker


class MergeAttrMapping:
    def __init__(self, dict_input):
        self._dict = dict_input

    def __iter__(self):
        return iter(self._dict)

    def __contains__(self, k):
        return k in self._dict

    def __getattr__(self, k):
        return self._dict[k]

    def __setattr__(self, k, v):
        if k in ['_dict']:
            super().__setattr__(k, v)
        else:
            self._dict[k][:] = v

    def __getitem__(self, k):
        warnings.warn(
            "dict-like access is discouraged, as it can be "
            "confusing vs. using attribute access",
            UseDiscouragedWarning,
            stacklevel=2,
        )
        return self._dict[k]


class UDFData:
    '''
    Container for result buffers, return value from running UDFs
    '''

    def __init__(self, data: Dict[str, BufferWrapper]):
        self._data = data
        self._views = {}

    def __repr__(self) -> str:
        return "<UDFData: %r>" % (
            self._data
        )

    def __getattr__(self, k: str):
        if k.startswith("_"):
            raise AttributeError("no such attribute: %s" % k)
        try:
            return self._get_view_or_data(k)
        except KeyError as e:
            raise AttributeError(str(e))

    def get_buffer(self, name):
        """
        Return the `BufferWrapper` for buffer `name`

        .. versionadded:: 0.7.0
        """
        return self._data[name]

    def get(self, k, default=None):
        try:
            return self.__getattr__(k)
        except KeyError:
            return default

    def __setattr__(self, k, v):
        if not k.startswith("_"):
            # convert UDFData.some_attr = something to array slice assignment
            getattr(self, k)[:] = v
        else:
            super().__setattr__(k, v)

    def _get_view_or_data(self, k):
        if k in self._views:
            return self._views[k]
        res = self._data[k]
        if hasattr(res, 'raw_data'):
            return res.raw_data
        return res

    def __getitem__(self, k):
        warnings.warn(
            "dict-like access is discouraged, as it can be "
            "confusing vs. using attribute access. Please use `get_buffer` instead, "
            "if you really need the `BufferWrapper` and not the current view",
            UseDiscouragedWarning,
            stacklevel=2,
        )
        return self._data[k]

    def __contains__(self, k):
        return k in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def as_dict(self) -> Dict[str, BufferWrapper]:
        return dict(self.items())

    def get_proxy(self):
        return MergeAttrMapping({
            k: (self._views[k] if k in self._views else self._data[k].raw_data)
            for k, v in self.items()
            if v and v.has_data()
        })

    def _get_buffers(self, filter_allocated: bool = False):
        for k, buf in self._data.items():
            if not hasattr(buf, 'has_data') or (buf.has_data() and filter_allocated):
                continue
            yield k, buf

    def allocate_for_part(self, partition: Shape, roi: np.ndarray, lib=None):
        """
        allocate all BufferWrapper instances in this namespace.
        for pre-allocated buffers (i.e. aux data), only set shape and roi
        """
        for k, buf in self._get_buffers():
            buf.set_shape_partition(partition, roi)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate(lib=lib)

    def allocate_for_full(self, dataset, roi: np.ndarray):
        for k, buf in self._get_buffers():
            buf.set_shape_ds(dataset.shape, roi)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate()

    def set_view_for_dataset(self, dataset):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_dataset(dataset)

    def set_view_for_partition(self, partition: Shape):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_partition(partition)

    def set_view_for_tile(self, partition, tile):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_tile(partition, tile)

    def set_contiguous_view_for_tile(self, partition, tile):
        # .. versionadded:: 0.5.0
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_contiguous_view_for_tile(partition, tile)

    def flush(self, debug=False):
        # .. versionadded:: 0.5.0
        for k, buf in self._get_buffers():
            buf.flush(debug=debug)

    def export(self):
        # .. versionadded:: 0.6.0
        for k, buf in self._get_buffers():
            buf.export()

    def set_view_for_frame(self, partition, tile, frame_idx):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_frame(partition, tile, frame_idx)

    def new_for_partition(self, partition, roi: np.ndarray):
        for k, buf in self._get_buffers():
            self._data[k] = buf.new_for_partition(partition, roi)

    def clear_views(self):
        self._views = {}


class UDFFrameMixin:
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


class UDFTileMixin:
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


class UDFPartitionMixin:
    '''
    Implement :code:`process_partition` for per-partition processing.
    '''

    def process_partition(self, partition: np.ndarray):
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


class UDFPreprocessMixin:
    '''
    Implement :code:`preprocess` to initialize the result buffers of a partition on the worker
    before the partition data is processed.

    .. versionadded:: 0.3.0
    '''

    def preprocess(self):
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


class UDFPostprocessMixin:
    '''
    Implement :code:`postprocess` to modify the resulf buffers of a partition on the worker
    after the partition data has been completely processed, but before it is returned to the
    main node for the final merging step.
    '''

    def postprocess(self):
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


class UDFBase:
    '''
    Base class for UDFs with helper functions.
    '''

    def allocate_for_part(self, partition, roi):
        for ns in [self.results]:
            ns.allocate_for_part(partition, roi, lib=self.xp)

    def allocate_for_full(self, dataset, roi):
        for ns in [self.params, self.results]:
            ns.allocate_for_full(dataset, roi)

    def set_views_for_dataset(self, dataset):
        for ns in [self.params]:
            ns.set_view_for_dataset(dataset)

    def set_views_for_partition(self, partition):
        for ns in [self.params, self.results]:
            ns.set_view_for_partition(partition)

    def set_views_for_tile(self, partition, tile):
        for ns in [self.params, self.results]:
            ns.set_view_for_tile(partition, tile)

    def set_contiguous_views_for_tile(self, partition, tile):
        # .. versionadded:: 0.5.0
        for ns in [self.params, self.results]:
            ns.set_contiguous_view_for_tile(partition, tile)

    def flush(self, debug=False):
        # .. versionadded:: 0.5.0
        for ns in [self.params, self.results]:
            ns.flush(debug=debug)

    def set_views_for_frame(self, partition, tile, frame_idx):
        for ns in [self.params, self.results]:
            ns.set_view_for_frame(partition, tile, frame_idx)

    def clear_views(self):
        for ns in [self.params, self.results]:
            ns.clear_views()

    def init_task_data(self):
        self.task_data = UDFData(self.get_task_data())

    def init_result_buffers(self):
        self.results = UDFData(self.get_result_buffers())

    def export_results(self):
        # .. versionadded:: 0.6.0.dev0
        self.results.export()

    def set_meta(self, meta):
        self.meta = meta

    def set_slice(self, slice_):
        self.meta.slice = slice_

    def set_backend(self, backend):
        assert backend in self.get_backends()
        self._backend = backend

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

    def get_method(self):
        if hasattr(self, 'process_tile'):
            method = 'tile'
        elif hasattr(self, 'process_frame'):
            method = 'frame'
        elif hasattr(self, 'process_partition'):
            method = 'partition'
        else:
            raise TypeError("UDF should implement one of the `process_*` methods")
        return method

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

    def _do_get_results(self):
        results_tmp = self.get_results()
        decl = self.get_result_buffers()

        # include any results that were not explicitly included, but have non-private `use`:
        results_tmp.update({
            k: getattr(self.results, k)
            for k, v in decl.items()
            if k not in results_tmp and v.use is None
        })

        # wrap numpy results into `ResultBuffer`s:
        results = {}
        for name, arr in results_tmp.items():
            self._check_results(decl, arr, name)
            buf_decl = decl[name]
            buf = PreallocBufferWrapper(
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
    TILE_SIZE_BEST_FIT = object()
    TILE_SIZE_MAX = np.inf
    TILE_DEPTH_DEFAULT = object()
    TILE_DEPTH_MAX = np.inf

    def __init__(self, **kwargs):
        self._backend = 'numpy'  # default so that self.xp can always be used
        self._kwargs = kwargs
        self.params = UDFData(kwargs)
        self.task_data = None
        self.results = None
        self._requires_custom_merge = None

    def copy(self):
        return self.__class__(**self._kwargs)

    def copy_for_partition(self, partition: np.ndarray, roi: np.ndarray):
        """
        create a copy of the UDF, specifically slicing aux data to the
        specified pratition and roi
        """
        new_instance = self.__class__(**self._kwargs)
        new_instance.params.new_for_partition(partition, roi)
        return new_instance

    def get_task_data(self):
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

    def get_result_buffers(self):
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

    def get_results(self):
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

    def get_preferred_input_dtype(self):
        '''
        Override this method to specify the preferred input dtype of the UDF.

        The default is :code:`float32` since most numerical processing tasks
        perform best with this dtype, namely dot products.

        The back-end uses this preferred input dtype in combination with the
        dataset`s native dtype to determine the input dtype using
        :meth:`numpy.result_type`. That means :code:`float` data in a dataset
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

    def get_tiling_preferences(self):
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

    def get_backends(self):
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

    def cleanup(self):  # FIXME: name? implement cleanup as context manager somehow?
        pass

    def buffer(self, kind, extra_shape=(), dtype="float32", where=None, use=None):
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
    def __init__(self, preferred_input_dtype=UDF.USE_NATIVE_DTYPE):
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
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


class Task(object):
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

    def get_resources(self):
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

    def __call__(self):
        raise NotImplementedError()


class UDFTask(Task):
    def __init__(self, partition: Partition, idx, udfs, roi, backends, corrections=None):
        super().__init__(partition=partition, idx=idx)
        self._roi = roi
        self._udfs = udfs
        self._backends = backends
        self._corrections = corrections

    def __call__(self, env=None):
        return UDFRunner(self._udfs).run_for_partition(
            self.partition, self._roi, self._corrections, env,
        )

    def get_resources(self):
        """
        Intersection of resources of all UDFs, throws if empty.

        See docstring of super class for details.
        """
        needs_cuda = 0
        needs_cpu = 0
        needs_ndarray = 0
        backends_for_udfs = []
        for udf in self._udfs:
            b = udf.get_backends()
            if isinstance(b, str):
                b = (b, )
            backends_for_udfs.append(set(b))

        # Limit to externally specified backends
        for backend_set in backends_for_udfs:
            if self._backends is not None:
                b = self._backends
                if isinstance(b, str):
                    b = (b, )
                backends = set(b).intersection(backend_set)
            else:
                backends = backend_set
            needs_cuda += 'numpy' not in backends
            needs_cpu += ('cuda' not in backends) and ('cupy' not in backends)
            needs_ndarray += 'cuda' not in backends
        if needs_cuda and needs_cpu:
            raise ValueError(
                "There is no common supported UDF backend (have: %r, limited to %r)"
                % (backends_for_udfs, self._backends)
            )
        result = {'compute': 1}
        if needs_cpu:
            result['CPU'] = 1
        if needs_cuda:
            result['CUDA'] = 1
        if needs_ndarray:
            result['ndarray'] = 1
        return result

    def __repr__(self):
        return "<UDFTask %r>" % (self._udfs,)


class UDFRunner:
    def __init__(self, udfs, debug=False):
        self._udfs = udfs
        self._debug = debug
        self._pool = ThreadPoolExecutor(max_workers=4)

    @classmethod
    def inspect_udf(cls, udf, dataset, roi=None):
        """
        Return result buffer declarations for a given UDF/DataSet/roi combination
        """
        runner = UDFRunner([udf])
        meta = UDFMeta(
            partition_shape=None,
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
        runner = UDFRunner(udfs)
        executor = InlineJobExecutor()
        res = runner.run_for_dataset(
            dataset=dataset,
            executor=executor,
            roi=roi,
            dry=True
        )
        return res

    def _get_dtype(self, dtype, corrections):
        if corrections is not None and corrections.have_corrections():
            tmp_dtype = np.result_type(np.float32, dtype)
        else:
            tmp_dtype = dtype
        for udf in self._udfs:
            tmp_dtype = np.result_type(
                udf.get_preferred_input_dtype(),
                tmp_dtype
            )
        return tmp_dtype

    def _init_udfs(self, numpy_udfs, cupy_udfs, partition, roi, corrections, device_class, env):
        dtype = self._get_dtype(partition.dtype, corrections)
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=partition.dtype,
            input_dtype=dtype,
            tiling_scheme=None,
            corrections=corrections,
            device_class=device_class,
            threads_per_worker=env.threads_per_worker,
        )
        for udf in numpy_udfs:
            if device_class == 'cuda':
                udf.set_backend('cuda')
            else:
                udf.set_backend('numpy')
        if device_class == 'cpu':
            assert not cupy_udfs
        for udf in cupy_udfs:
            udf.set_backend('cupy')
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_part(partition, roi)
            udf.init_task_data()
            # TODO: preprocess doesn't have access to the tiling scheme - is this ok?
            if hasattr(udf, 'preprocess'):
                udf.clear_views()
                udf.preprocess()
        neg = Negotiator()
        # FIXME take compute backend into consideration as well
        # Other boundary conditions when moving input data to device
        tiling_scheme = neg.get_scheme(
            udfs=udfs,
            partition=partition,
            read_dtype=dtype,
            roi=roi,
            corrections=corrections,
        )

        # print(tiling_scheme)

        # FIXME: don't fully re-create?
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=partition.dtype,
            input_dtype=dtype,
            tiling_scheme=tiling_scheme,
            corrections=corrections,
            device_class=device_class,
            threads_per_worker=env.threads_per_worker,
        )
        for udf in udfs:
            udf.set_meta(meta)
        return (meta, tiling_scheme, dtype)

    def _run_tile(self, udfs, partition, tile, device_tile):
        for udf in udfs:
            method = udf.get_method()
            if method == 'tile':
                udf.set_contiguous_views_for_tile(partition, tile)
                udf.set_slice(tile.tile_slice)
                udf.process_tile(device_tile)
            elif method == 'frame':
                tile_slice = tile.tile_slice
                for frame_idx, frame in enumerate(device_tile):
                    frame_slice = Slice(
                        origin=(tile_slice.origin[0] + frame_idx,) + tile_slice.origin[1:],
                        shape=Shape((1,) + tuple(tile_slice.shape)[1:],
                                    sig_dims=tile_slice.shape.sig.dims),
                    )
                    udf.set_slice(frame_slice)
                    udf.set_views_for_frame(partition, tile, frame_idx)
                    udf.process_frame(frame)
            elif method == 'partition':
                udf.set_views_for_tile(partition, tile)
                udf.set_slice(partition.slice)
                udf.process_partition(device_tile)

    def _run_udfs(self, numpy_udfs, cupy_udfs, partition, tiling_scheme, roi, dtype):
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
            self._run_tile(numpy_udfs, partition, tile, tile)
            if cupy_udfs:
                # Work-around, should come from dataset later
                device_tile = xp.asanyarray(tile)
                self._run_tile(cupy_udfs, partition, tile, device_tile)

    def _wrapup_udfs(self, numpy_udfs, cupy_udfs, partition):
        udfs = numpy_udfs + cupy_udfs
        for udf in udfs:
            udf.flush(self._debug)
            if hasattr(udf, 'postprocess'):
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

    def _udf_lists(self, device_class):
        numpy_udfs = []
        cupy_udfs = []
        if device_class == 'cuda':
            for udf in self._udfs:
                backends = udf.get_backends()
                if 'cuda' in backends:
                    numpy_udfs.append(udf)
                elif 'cupy' in backends:
                    cupy_udfs.append(udf)
                else:
                    raise ValueError(f"UDF backends are {backends}, supported on CUDA are "
                            "'cuda' and 'cupy'")
        elif device_class == 'cpu':
            assert all('numpy' in udf.get_backends() for udf in self._udfs)
            numpy_udfs = self._udfs
        else:
            raise ValueError(f"Unknown device class {device_class}, "
                "supported are 'cpu' and 'cuda'")
        return (numpy_udfs, cupy_udfs)

    def run_for_partition(self, partition: Partition, roi, corrections, env):
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
                (meta, tiling_scheme, dtype) = self._init_udfs(
                    numpy_udfs, cupy_udfs, partition, roi, corrections, device_class, env,
                )
                # print("UDF TilingScheme: %r" % tiling_scheme.shape)
                partition.set_corrections(corrections)
                self._run_udfs(numpy_udfs, cupy_udfs, partition, tiling_scheme, roi, dtype)
                self._wrapup_udfs(numpy_udfs, cupy_udfs, partition)
            finally:
                if previous_id is not None:
                    cupy.cuda.Device(previous_id).use()
            # Make sure results are in the same order as the UDFs
            return tuple(udf.results for udf in self._udfs)

    def _debug_task_pickling(self, tasks):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps(tasks))

    def _check_preconditions(self, dataset: DataSet, roi):
        if roi is not None and np.product(roi.shape) != np.product(dataset.shape.nav):
            raise ValueError(
                "roi: incompatible shapes: %s (roi) vs %s (dataset)" % (
                    roi.shape, dataset.shape.nav
                )
            )

    def _prepare_run_for_dataset(
        self, dataset: DataSet, executor, roi, corrections, backends, dry
    ):
        self._check_preconditions(dataset, roi)
        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=dataset.dtype,
            input_dtype=self._get_dtype(dataset.dtype, corrections),
            corrections=corrections,
        )
        for udf in self._udfs:
            udf.set_meta(meta)
            udf.init_result_buffers()
            udf.allocate_for_full(dataset, roi)

            if hasattr(udf, 'preprocess'):
                udf.set_views_for_dataset(dataset)
                udf.preprocess()
        if dry:
            tasks = []
        else:
            tasks = list(self._make_udf_tasks(dataset, roi, corrections, backends))
        return tasks

    def run_for_dataset(self, dataset: DataSet, executor,
                        roi=None, progress=False, corrections=None, backends=None, dry=False):
        for res in self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry
        ):
            pass
        return res

    def run_for_dataset_sync(self, dataset: DataSet, executor,
                        roi=None, progress=False, corrections=None, backends=None, dry=False):
        tasks = self._prepare_run_for_dataset(
            dataset, executor, roi, corrections, backends, dry
        )
        cancel_id = str(uuid.uuid4())
        self._debug_task_pickling(tasks)

        if progress:
            from tqdm import tqdm
            t = tqdm(total=len(tasks))

        executor = executor.ensure_sync()

        damage = BufferWrapper(kind='nav', dtype=bool)
        damage.set_shape_ds(dataset.shape, roi)
        damage.allocate()
        if tasks:
            for part_results, task in executor.run_tasks(tasks, cancel_id):
                if progress:
                    t.update(1)
                for results, udf in zip(part_results, self._udfs):
                    udf.set_views_for_partition(task.partition)
                    udf.merge(
                        dest=udf.results.get_proxy(),
                        src=results.get_proxy()
                    )
                    udf.clear_views()
                v = damage.get_view_for_partition(task.partition)
                v[:] = True
                yield UDFResults(
                    buffers=tuple(
                        udf._do_get_results()
                        for udf in self._udfs
                    ),
                    damage=damage
                )
        else:
            # yield at least one result (which should be empty):
            for udf in self._udfs:
                udf.clear_views()
            yield UDFResults(
                buffers=tuple(
                    udf._do_get_results()
                    for udf in self._udfs
                ),
                damage=damage
            )

        if progress:
            t.close()

    async def run_for_dataset_async(
        self, dataset: DataSet, executor, cancel_id, roi=None, corrections=None, backends=None,
        progress=False, dry=False
    ):
        gen = self.run_for_dataset_sync(
            dataset=dataset,
            executor=executor.ensure_sync(),
            roi=roi,
            progress=progress,
            corrections=corrections,
            backends=backends,
            dry=dry
        )

        async for res in async_generator_eager(gen, pool=self._pool):
            yield res

    def _roi_for_partition(self, roi, partition: Partition):
        return roi.reshape(-1)[partition.slice.get(nav_only=True)]

    def _make_udf_tasks(self, dataset: DataSet, roi, corrections, backends):
        for idx, partition in enumerate(dataset.get_partitions()):
            if roi is not None:
                roi_for_part = self._roi_for_partition(roi, partition)
                if np.count_nonzero(roi_for_part) == 0:
                    # roi is empty for this partition, ignore
                    continue
            udfs = [
                udf.copy_for_partition(partition, roi)
                for udf in self._udfs
            ]
            yield UDFTask(
                partition=partition, idx=idx, udfs=udfs, roi=roi, corrections=corrections,
                backends=backends,
            )


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

    buffers : Iterable[dict]
        Iterable containing the result buffer dictionaries for each of the UDFs being executed

    damage : BufferWrapper
        :class:`libertem.common.buffers.BufferWrapper` of :code:`kind='nav'`, :code:`dtype=bool`.
        It is set to :code:`True` for all positions in nav space that have been processed already.
    '''
    def __init__(self, buffers, damage):
        self.buffers = buffers
        self.damage = damage
