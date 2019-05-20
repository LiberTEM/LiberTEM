from types import MappingProxyType
import uuid

import numpy as np

from libertem.job.base import Task


class UDFNamespace:
    def __init__(self, data):
        self._data = data
        self._views = {}

    def __getattr__(self, k):
        assert not k.startswith("_")
        # TODO: return a view into the current unit of data
        # (partition, tile, frame) instead, if it is a BufferWrapper instance
        if k in self._views:
            return self._views[k]
        return self._data[k]

    def __getitem__(self, k):
        return self._data[k]

    def __contains__(self, k):
        return k in self._data

    def _get_buffers(self, filter_allocated=False):
        for k, buf in self._data.items():
            if not hasattr(buf, 'has_data') or (buf.has_data() and filter_allocated):
                continue
            yield k, buf

    def allocate_for_part(self, slice_, roi):
        """
        allocate all BufferWrapper instances in this namespace
        """
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.set_shape_part(slice_, roi)
            buf.allocate()

    def allocate_for_full(self, shape, roi):
        for k, buf in self._get_buffers(filter_allocated=True):
            buf.set_shape_full(shape, roi)
            buf.allocate()

    def set_view_for_partition(self, partition):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_into_part(partition.slice)

    def set_view_for_tile(self, partition, tile):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_into_part(tile.tile_slice)

    def set_view_for_frame(self, partition, tile, frame_idx):
        for k, buf in self._get_buffers():
            self._views[k] = buf.get_view_for_frame(partition, tile, frame_idx)

    def clear_views(self):
        self._views = {}

    def items(self):
        return self._data.items()

    def get_proxy(self):
        return MappingProxyType({
            k: (self._views[k] if k in self._views else self._data[k])
            for k, v in self._data.items()
        })


class UDF:
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs
            Input parameters. They are scattered to the worker processes and
            available as `self.params` from here on.

            Values can be `BufferWrapper` instances, which, when accessed via
            `self.params.the_key_here`, will automatically return a view corresponding
            to the current unit of data (frame, tile, partition).
        """
        self.params = UDFNamespace(kwargs)

    def get_task_data(self):
        """
        Initialize per-task data.

        Per-task data can be mutable. Use this function
        to allocate temporary buffers, or to initialize
        system resources.

        If you want to distribute static data, use
        parameters instead.

        Data available in this method:
        - `self.params` - the input parameters of this UDF

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

        Returns
        -------
        dict
            Flat dict with string keys. Keys should
            be valid python identifiers, which allows
            access via `self.results.the_key_here`.
        """
        raise NotImplementedError()

    def process_partition(self, partition):
        """
        Implement this method to process the data partitioned into large
        (100s of MiB) partitions.

        Data available in this method:
        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances

        Note
        ----
        Only use this method if you know what you are doing; especially if
        you are running a processing pipeline with multiple steps, or multiple
        processing pipelines at the same time, performance may be adversely
        impacted.

        Parameters
        ----------
        partition : ndarray
            A large number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()

    def process_tile(self, tile):
        """
        Implement this method to process the data in a tiled manner.

        Data available in this method:
        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances

        Parameters
        ----------
        tile : ndarray
            A small number N of frames or signal elements from the dataset.
            The shape is (N,) + `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 3D, for spectra 2D etc.
        """
        raise NotImplementedError()

    def process_frame(self, frame):
        """
        Implement this method to process the data on a frame-by-frame manner.

        Data available in this method:
        - `self.params`    - the parameters of this UDF
        - `self.task_data` - task data created by `get_task_data`
        - `self.results`   - the result buffer instances

        Parameters
        ----------
        frame : ndarray
            A single frame or signal element from the dataset.
            The shape is the same as `dataset.shape.sig`. In case of pixelated
            STEM / scanning diffraction data this is 2D, for spectra 1D etc.
        """
        raise NotImplementedError()

    def merge(self, dest, src):
        """
        Merge a partial result `src` into the current global result `dest`.

        Data available in this method:
        - `self.params` - the parameters of this UDF

        Note
        ----
        This function is running on the leader node, which means `self.results`
        and `self.task_data` are not available.
        """
        for k in dest:
            check_cast(dest[k], src[k])
            dest[k][:] = src[k]

    def cleanup(self):  # FIXME: name?
        pass

    def allocate_for_part(self, slice_, roi):
        for ns in [self.results]:
            ns.allocate_for_part(slice_, roi)

    def allocate_for_full(self, shape, roi):
        for ns in [self.results]:
            ns.allocate_for_full(shape, roi)

    def set_views_for_partition(self, partition):
        for ns in [self.params, self.results]:
            ns.set_view_for_partition(partition)

    def set_views_for_tile(self, partition, tile):
        for ns in [self.params, self.results]:
            ns.set_view_for_tile(partition, tile)

    def set_views_for_frame(self, partition, tile, frame_idx):
        for ns in [self.params, self.results]:
            ns.set_view_for_frame(partition, tile, frame_idx)

    def init_task_data(self):
        self.task_data = UDFNamespace(self.get_task_data())

    def init_result_buffers(self):
        self.results = UDFNamespace(self.get_result_buffers())


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


def merge_assign(dest, src):
    for k in dest:
        check_cast(dest[k], src[k])
        dest[k][:] = src[k]


class UDFTask(Task):
    def __init__(self, partition, idx, udf, roi):
        super().__init__(partition=partition, idx=idx)
        self._roi = roi
        self._udf = udf

    def __call__(self):
        return UDFRunner(self._udf).run_for_partition(self.partition, self._roi)


class UDFRunner:
    def __init__(self, udf):
        self._udf = udf

    def run_for_partition(self, partition, roi):
        self._udf.init_result_buffers()
        self._udf.allocate_for_part(partition.slice, roi)
        self._udf.init_task_data()
        for tile in partition.get_tiles(full_frames=True, roi=roi):
            for frame_idx, frame in enumerate(tile.data):
                self._udf.set_views_for_frame(partition, tile, frame_idx)
                self._udf.process_frame(frame)
        self._udf.cleanup()
        return self._udf.results, partition

    def run_for_dataset(self, dataset, executor, roi):
        self._udf.init_result_buffers()
        self._udf.allocate_for_full(dataset.shape, roi)
        for k, v in self._udf.results.items():
            assert self._udf.results[k]._full_shape is not None

        tasks = self._make_udf_tasks(dataset, roi)
        cancel_id = str(uuid.uuid4())

        for part_results, partition in executor.run_tasks(tasks, cancel_id):
            self._udf.set_views_for_partition(partition)
            self._udf.merge(
                dest=self._udf.results.get_proxy(),
                src=part_results.get_proxy()
            )
        return self._udf.results

    def _make_udf_tasks(self, dataset, roi):
        return (
            UDFTask(partition=partition, idx=idx, udf=self._udf, roi=roi)
            for idx, partition in enumerate(dataset.get_partitions())
        )
