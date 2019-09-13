from types import MappingProxyType
from typing import Dict
import uuid

import cloudpickle
import numpy as np

from libertem.job.base import Task
from libertem.common.buffers import BufferWrapper, AuxBufferWrapper
from libertem.common import Shape, Slice


class UDFMeta:
    """
    UDF metadata. Makes all relevant metadata accessible to the UDF. Can be different
    for each task/partition.
    """
    def __init__(self, partition_shape, dataset_shape, roi, dataset_dtype):
        self._partition_shape = partition_shape
        self._dataset_shape = dataset_shape
        self._dataset_dtype = dataset_dtype
        if roi is not None:
            roi = roi.reshape(dataset_shape.nav)
        self._roi = roi
        self._slice = None

    @property
    def slice(self):
        """
        Slice : A :class:`~libertem.common.slice.Slice` instance that describes the location
                within the dataset with navigation dimension flattened and reduced to the ROI.
        """
        return self._slice

    @slice.setter
    def slice(self, new_slice):
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


class UDFData:
    '''
    Container for result buffers, return value from running UDFs
    '''
    def __init__(self, data):
        self._data = data
        self._views = {}

    def __repr__(self):
        return "<UDFData: %r>" % (
            self._data
        )

    def __getattr__(self, k):
        if k.startswith("_"):
            raise AttributeError("no such attribute: %s" % k)
        try:
            return self._get_view_or_data(k)
        except KeyError as e:
            raise AttributeError(str(e))

    def get(self, k, default=None):
        try:
            return self.__getattr__(k)
        except KeyError:
            return default

    def __setattr__(self, k, v):
        if not k.startswith("_"):
            raise AttributeError(
                "cannot re-assign attribute %s, did you mean `.%s[:] = ...`?" % (
                    k, k
                )
            )
        super().__setattr__(k, v)

    def _get_view_or_data(self, k):
        if k in self._views:
            return self._views[k]
        res = self._data[k]
        if hasattr(res, 'raw_data'):
            return res.raw_data
        return res

    def __getitem__(self, k):
        return self._data[k]

    def __contains__(self, k):
        return k in self._data

    def items(self):
        return self._data.items()

    def keys(self):
        return self._data.keys()

    def as_dict(self):
        return dict(self.items())

    def get_proxy(self):
        return MappingProxyType({
            k: (self._views[k] if k in self._views else self._data[k].raw_data)
            for k, v in self._data.items()
        })

    def _get_buffers(self, filter_allocated=False):
        for k, buf in self._data.items():
            if not hasattr(buf, 'has_data') or (buf.has_data() and filter_allocated):
                continue
            yield k, buf

    def allocate_for_part(self, partition, roi):
        """
        allocate all BufferWrapper instances in this namespace.
        for pre-allocated buffers (i.e. aux data), only set shape and roi
        """
        for k, buf in self._get_buffers():
            buf.set_shape_partition(partition, roi)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate()

    def allocate_for_full(self, dataset, roi):
        for k, buf in self._get_buffers():
            buf.set_shape_ds(dataset, roi)
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.allocate()

    def set_view_for_partition(self, partition):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_partition(partition)

    def set_view_for_tile(self, partition, tile):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_tile(partition, tile)

    def set_view_for_frame(self, partition, tile, frame_idx):
        for k, buf in self._get_buffers():
            if buf.roi_is_zero:
                raise ValueError("should not happen")
            else:
                self._views[k] = buf.get_view_for_frame(partition, tile, frame_idx)

    def new_for_partition(self, partition, roi):
        for k, buf in self._get_buffers():
            self._data[k] = buf.new_for_partition(partition, roi)

    def clear_views(self):
        self._views = {}


class UDFFrameMixin:
    '''
    Implement :code:`process_frame` for per-frame processing.
    '''
    def process_frame(self, frame):
        """
        Implement this method to process the data on a frame-by-frame manner.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        - `self.meta`      - meta data about the current operation and data set

        Parameters
        ----------
        frame : numpy.ndarray
            A single frame or signal element from the dataset.
            The shape is the same as `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 2D, for spectra 1D etc.
        """
        raise NotImplementedError()


class UDFTileMixin:
    '''
    Implement :code:`process_tile` for per-tile processing.
    '''
    def process_tile(self, tile):
        """
        Implement this method to process the data in a tiled manner.

        Data available in this method:

        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances
        - `self.meta`      - meta data about the current operation and data set

        Parameters
        ----------
        tile : numpy.ndarray
            A small number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()


class UDFPartitionMixin:
    '''
    Implement :code:`process_partition` for per-partition processing.
    '''
    def process_partition(self, partition):
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
        partition : numpy.ndarray
            A large number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()


class UDFPostprocessMixin:
    '''
    Implement :code:`postprocess` to modify the resulf buffers of a partition on the worker
    after the partition data has been completely processed, but before it is returned to the
    master node for the final merging step.
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
            ns.allocate_for_part(partition, roi)

    def allocate_for_full(self, dataset, roi):
        for ns in [self.results]:
            ns.allocate_for_full(dataset, roi)

    def set_views_for_partition(self, partition):
        for ns in [self.params, self.results]:
            ns.set_view_for_partition(partition)

    def set_views_for_tile(self, partition, tile):
        for ns in [self.params, self.results]:
            ns.set_view_for_tile(partition, tile)

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

    def set_meta(self, meta):
        self.meta = meta

    def set_slice(self, slice_):
        self.meta.slice = slice_

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


class UDF(UDFBase):
    """
    The main user-defined functions interface. You can implement your functionality
    by overriding methods on this class.
    """
    def __init__(self, **kwargs):
        """
        Create a new UDF instance. If you override `__init__`, please take care,
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
        self._kwargs = kwargs
        self.params = UDFData(kwargs)
        self.task_data = None
        self.results = None

    def copy(self):
        return self.__class__(**self._kwargs)

    def copy_for_partition(self, partition, roi):
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

        Values of the returned dict should be `BufferWrapper`
        instances, which, when accessed via `self.results.key`,
        will automatically return a view corresponding to the
        current unit of data (frame, tile, partition).

        The values also need to be serializable via pickle.

        Data available in this method:

        - `self.params` - the parameters of this UDF
        - `self.meta` - relevant metadata, see :class:`UDFMeta` documentation.
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

    def merge(self, dest: Dict[str, np.array], src: Dict[str, np.array]):
        """
        Merge a partial result `src` into the current global result `dest`.

        Data available in this method:

        - `self.params` - the parameters of this UDF

        Parameters
        ----------

        dest
            global results; dictionary mapping the buffer name (from `get_result_buffers`)
            to a numpy array

        src
            results for a partition; dictionary mapping the buffer name (from `get_result_buffers`)
            to a numpy array

        Note
        ----
        This function is running on the leader node, which means `self.results`
        and `self.task_data` are not available.
        """
        for k in dest:
            check_cast(dest[k], src[k])
            dest[k][:] = src[k]

    def cleanup(self):  # FIXME: name? implement cleanup as context manager somehow?
        pass

    def buffer(self, kind, extra_shape=(), dtype="float32"):
        return BufferWrapper(kind, extra_shape, dtype)

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


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


class UDFTask(Task):
    def __init__(self, partition, idx, udf, roi):
        super().__init__(partition=partition, idx=idx)
        self._roi = roi
        self._udf = udf

    def __call__(self):
        return UDFRunner(self._udf).run_for_partition(self.partition, self._roi)


class UDFRunner:
    def __init__(self, udf, debug=False):
        self._udf = udf
        self._debug = debug

    def _get_dtype(self, dtype):
        '''
        FIXME replace with dtype switching logic and user control similar to MaskJob
        '''
        dtype = np.dtype(dtype)
        # simple dtype logic for now: if data is complex or float, keep data dtype,
        # otherwise convert to float32
        if dtype.kind not in ('c', 'f'):
            dtype = np.dtype("float32")
        return dtype

    def run_for_partition(self, partition, roi):
        dtype = self._get_dtype(partition.dtype)
        meta = UDFMeta(
            partition_shape=partition.slice.adjust_for_roi(roi).shape,
            dataset_shape=partition.meta.shape,
            roi=roi,
            dataset_dtype=dtype
        )
        self._udf.set_meta(meta)
        self._udf.init_result_buffers()
        self._udf.allocate_for_part(partition, roi)
        self._udf.init_task_data()
        method = self._udf.get_method()
        if method == 'tile':
            tiles = partition.get_tiles(full_frames=False, roi=roi, dest_dtype=dtype)
        elif method == 'frame':
            tiles = partition.get_tiles(full_frames=True, roi=roi, dest_dtype=dtype)
        elif method == 'partition':
            tiles = [partition.get_macrotile(roi=roi, dest_dtype=dtype)]

        for tile in tiles:
            if method == 'tile':
                self._udf.set_views_for_tile(partition, tile)
                self._udf.set_slice(tile.tile_slice)
                self._udf.process_tile(tile.data)
            elif method == 'frame':
                tile_slice = tile.tile_slice
                for frame_idx, frame in enumerate(tile.data):
                    frame_slice = Slice(
                        origin=(tile_slice.origin[0] + frame_idx,) + tile_slice.origin[1:],
                        shape=Shape((1,) + tuple(tile_slice.shape)[1:],
                                    sig_dims=tile_slice.shape.sig.dims),
                    )
                    self._udf.set_slice(frame_slice)
                    self._udf.set_views_for_frame(partition, tile, frame_idx)
                    self._udf.process_frame(frame)
            elif method == 'partition':
                self._udf.set_views_for_tile(partition, tile)
                self._udf.set_slice(partition.slice)
                self._udf.process_partition(tile.data)

        if hasattr(self._udf, 'postprocess'):
            self._udf.clear_views()
            self._udf.postprocess()

        self._udf.cleanup()
        self._udf.clear_views()

        if self._debug:
            try:
                cloudpickle.loads(cloudpickle.dumps(partition))
            except TypeError:
                raise TypeError("could not pickle partition")
            try:
                cloudpickle.loads(cloudpickle.dumps(self._udf.results))
            except TypeError:
                raise TypeError("could not pickle results")

        return self._udf.results, partition

    def run_for_dataset(self, dataset, executor, roi=None):
        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=self._get_dtype(dataset.dtype)
        )
        self._udf.set_meta(meta)
        self._udf.init_result_buffers()
        self._udf.allocate_for_full(dataset, roi)

        tasks = self._make_udf_tasks(dataset, roi)
        cancel_id = str(uuid.uuid4())

        for part_results, partition in executor.run_tasks(tasks, cancel_id):
            self._udf.set_views_for_partition(partition)
            self._udf.merge(
                dest=self._udf.results.get_proxy(),
                src=part_results.get_proxy()
            )

        self._udf.clear_views()

        return self._udf.results.as_dict()

    async def run_for_dataset_async(self, dataset, executor, cancel_id, roi=None):
        meta = UDFMeta(
            partition_shape=None,
            dataset_shape=dataset.shape,
            roi=roi,
            dataset_dtype=dataset.dtype
        )
        self._udf.set_meta(meta)
        # FIXME: code duplication?
        self._udf.init_result_buffers()
        self._udf.allocate_for_full(dataset, roi)

        tasks = self._make_udf_tasks(dataset, roi)

        async for part_results, partition in executor.run_tasks(tasks, cancel_id):
            self._udf.set_views_for_partition(partition)
            self._udf.merge(
                dest=self._udf.results.get_proxy(),
                src=part_results.get_proxy()
            )
            self._udf.clear_views()
            yield self._udf.results.as_dict()
        else:
            # yield at least one result (which should be empty):
            self._udf.clear_views()
            yield self._udf.results.as_dict()

    def _roi_for_partition(self, roi, partition):
        return roi.reshape(-1)[partition.slice.get(nav_only=True)]

    def _make_udf_tasks(self, dataset, roi):
        for idx, partition in enumerate(dataset.get_partitions()):
            if roi is not None:
                roi_for_part = self._roi_for_partition(roi, partition)
                if np.count_nonzero(roi_for_part) == 0:
                    # roi is empty for this partition, ignore
                    continue
            udf = self._udf.copy_for_partition(partition, roi)
            yield UDFTask(partition=partition, idx=idx, udf=udf, roi=roi)
