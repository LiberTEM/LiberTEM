from functools import partial
from typing import Any, Optional
from collections.abc import Iterable
import contextlib
from collections import defaultdict, OrderedDict

import numpy as np
import dask
from dask import delayed
import dask.array as da

from libertem.io.corrections import CorrectionSet
from libertem.io.dataset.base import DataSet
from libertem.utils.devices import detect

from .base import BaseJobExecutor
from libertem.common.executor import Environment, TaskCommHandler, TaskProtocol
from libertem.common.scheduler import Worker, WorkerSet

from ..common.buffers import BufferWrapper
from ..common.math import prod
from ..udf.base import (
    UDFMergeAllMixin, UDFRunner, UDF, UDFData, MergeAttrMapping,
    get_resources_for_backends, UDFResults, BackendSpec
)

from .utils.dask_buffer import DaskBufferWrapper, DaskPreallocBufferWrapper, DaskResultBufferWrapper
from .utils import delayed_unpack


class DelayedUDFRunner(UDFRunner):
    def __init__(self, udfs: list[UDF], debug: bool = False, progress_reporter: Any = None):
        self._part_results = defaultdict(dict)
        super().__init__(udfs, debug=debug)

    @staticmethod
    def _make_udf_result(udfs: Iterable[UDF], damage: BufferWrapper) -> "UDFResults":
        udf_results = UDFRunner._make_udf_result(udfs, damage)
        buffers = udf_results.buffers
        damage = udf_results.damage
        new_buffers = tuple(
            {
                k: DaskResultBufferWrapper.from_buffer_wrapper(v)
                for k, v in bufs.items()
            }
            for bufs in buffers
        )
        return UDFResults(
            buffers=new_buffers,
            damage=damage,
        )

    def _apply_part_result(self, udfs: Iterable[UDF], damage, part_results, task):
        for part_results_udf, udf in zip(part_results, udfs):
            # Allow user to define an alternative merge strategy
            # using dask-compatible functions. In the Delayed case we
            # won't be getting partial results with damage anyway.
            # Currently there is no interface to provide all of the results
            # to udf.merge at once and in the correct order, so I am accumulating
            # results in self._part_results[udf] = {partition_slice_roi: part_results, ...}
            if (isinstance(udf, UDFMergeAllMixin)
                    or (type(udf).merge is UDF.merge and not udf.requires_custom_merge_all)):
                if self._accumulate_part_results(udf, part_results_udf, task):
                    try:
                        parts = {
                            key: value.get_proxy()
                            for key, value in self._part_results[udf].items()
                        }
                        udf._do_merge_all(parts)
                    finally:
                        self._part_results.pop(udf, None)
                continue

            # In principle we can requrire that sig merges are done sequentially
            # by using dask.graph_manipulation.bind, this could improve thread
            # safety on the sig buffer. In the current implem it's not necessary as
            # we are calling delayed_apply_part_result sequentially on results
            # and replacing all buffers with their partially merged versions
            # on each call, so we are implicitly sequential when updating
            # try:
            #     bind_to = udf.prior_merge
            # except AttributeError:
            #     bind_to = None

            structure = structure_from_task([udf], task)[0]
            flat_structure = delayed_unpack.flatten_nested(structure)
            flat_mapping = delayed_unpack.build_mapping(structure)

            src = part_results_udf.get_proxy()
            src_dict = {k: b for k, b in src._dict.items()}

            dest_dict = {}
            for k, b_decl in udf.get_result_buffers().items():
                # We skip result_only buffers since they are not expected
                # in the dest_dict by the default merge function
                if b_decl.use == 'result_only':
                    continue
                b = udf.results.get_buffer(k)
                view = b.get_view_for_partition(task.partition)
                # Handle result-only buffers
                if view is not None:
                    try:
                        dest_dict[k] = view.unwrap_sliced()
                    except AttributeError:
                        # Handle kind='single' buffers
                        # Could handle sig buffers this way too if we assert no chunking
                        dest_dict[k] = view

            # Run the udf.merge function in a wrapper which flattens
            # the results to a flat list which can be unpacked from the Delayed
            # This step makes a copy of the buffer view inside the delayed
            # call because this default Dask behaviour is to give a read-only
            # view of the arguments to the delayed function
            merged_del = delayed(merge_wrap, nout=len(flat_mapping))
            # need to copy the damage, as it will be overwritten in the next loop iteration,
            # which is independent from actually evaluating the delayed object:
            merged_del_result = merged_del(udf, dest_dict, src_dict, damage.raw_data.copy())

            # The following is needed when binding sequential updates on sig buffers
            # if bind_to is not None:
            #     merged_del = delayed_bind([*merged_del_result],
            #                               [*bind_to])
            # if any([b.kind != 'nav' for _, b in udf.results.items()]):
            #     # have sig result, gotta do a bind merge next time round
            #     udf.prior_merge = merged_del_result

            # We now build the result back into dask arrays and assign them
            # into the appropriate slices of the full result buffers
            wrapped_res = delayed_to_buffer_wrappers(merged_del_result, flat_structure,
                                                     task.partition, as_buffer=False)
            renested = delayed_unpack.rebuild_nested(wrapped_res, flat_mapping)

            # Assign into the result buffers with the partially merged version
            # This is OK because we are calling this part merge function sequentially
            # so each new call gets the most recent version of the buffer in the
            # dask task graph
            udf.set_views_for_partition(task.partition)
            dest = udf.results.get_proxy()
            merged = MergeAttrMapping(renested)
            for k in dest:
                getattr(dest, k)[:] = getattr(merged, k)

        v = damage.get_view_for_partition(task.partition)
        v[:] = True

    def _accumulate_part_results(self, udf, part_results, task):
        """
        If the udf has a merge_all method, this function is used
        to accumulate dask array-backed partial results on the executor
        so that the merge_all can be called on all of them at once

        Ensures that the results are correctly ordered and complete
        before allowing merge_all to be called
        """
        buf = next(iter(part_results.values()))  # get the first buffer
        slice_with_roi = buf._slice_for_partition(task.partition)
        self._part_results[udf][slice_with_roi] = part_results

        # number of frames in dataset
        if udf.meta.roi is not None:
            target_coverage = np.count_nonzero(udf.meta.roi)
        else:
            target_coverage = prod(task.partition.meta.shape.nav)
        # number of frames we have results for
        current_coverage = sum(prod(k.shape.nav) for k in self._part_results[udf].keys())
        if target_coverage == current_coverage:
            ordered_results = sorted(self._part_results[udf].items(),
                                     key=lambda kv: kv[0].origin[0])
            self._part_results[udf] = OrderedDict(ordered_results)
            return True
        elif current_coverage > target_coverage:
            raise RuntimeError('More frames accumulated than ROI specifies - '
                              f'target {target_coverage} - processed {current_coverage}')
        return False

    def results_for_dataset_sync(self, dataset: DataSet, executor: 'DelayedJobExecutor',
            roi: Optional[np.ndarray] = None, progress: bool = False,
            corrections: Optional[CorrectionSet] = None, backends: Optional[BackendSpec] = None,
            dry: bool = False) -> Iterable[tuple]:

        executor.register_master_udfs(self._udfs)

        return super().results_for_dataset_sync(
            dataset, executor, roi=roi, progress=progress,
            corrections=corrections, backends=backends, dry=dry
        )


class DelayedJobExecutor(BaseJobExecutor):
    """
    :class:`~libertem.common.executor.JobExecutor` that uses dask.delayed to execute tasks.

    .. versionadded:: 0.9.0

    Highly experimental at this time!
    """
    def __init__(self):
        # Only import if actually instantiated, i.e. will likely be used
        import libertem.preload  # noqa: 401
        self._udfs = None

    @contextlib.contextmanager
    def scatter(self, obj):
        yield delayed(obj)

    def cancel(self, cancel_id: Any):
        pass

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: TaskCommHandler,
    ):
        """
        Wraps the call task() such that it returns a flat list
        of results, then unpacks the Delayed return value into
        the normal

            :code:`tuple(udf.results for udf in self._udfs)`

        where the buffers inside udf.results are dask arrays instead
        of normal np.arrays

        Needs a reference to the udfs on the master node so
        that the results structure can be inferred. This reference is
        found in self._udfs, which is set with the method:

            :code:`executor.register_master_udfs(udfs)`

        called from :meth:`DelayedUDFRunner.results_for_dataset_sync`
        """
        env = Environment(threads_per_worker=1, threaded_executor=True)
        for task in tasks:
            structure = structure_from_task(self._udfs, task)
            flat_structure = delayed_unpack.flatten_nested(structure)
            flat_mapping = delayed_unpack.build_mapping(structure)
            flat_result_task = partial(task_wrap, task)
            result = delayed(flat_result_task, nout=len(flat_structure))(env=env,
                                                                         params=params_handle)
            wrapped_res = delayed_to_buffer_wrappers(result, flat_structure, task.partition,
                                                     roi=self._udfs[0].meta.roi)
            renested = delayed_unpack.rebuild_nested(wrapped_res, flat_mapping)
            result = tuple(UDFData(data=res) for res in renested)
            yield result, task

    def run_function(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return result

    def run_delayed(self, fn, *args, _delayed_kwargs=None, **kwargs):
        if _delayed_kwargs is None:
            _delayed_kwargs = {}
        result = delayed(fn, **_delayed_kwargs)(*args, **kwargs)
        return result

    def map(self, fn, iterable):
        return [fn(item)
                for item in iterable]

    def run_each_host(self, fn, *args, **kwargs):
        return {"localhost": fn(*args, **kwargs)}

    def run_each_worker(self, fn, *args, **kwargs):
        return {"delayed": fn(*args, **kwargs)}

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}
        # We don't know at this time,
        # but assume one worker per CPU
        devices = detect()
        return WorkerSet([
            Worker(
                name='delayed', host='localhost',
                resources=resources,
                nthreads=len(devices['cpus']),
            )
        ])

    def modify_buffer_type(self, buf):
        """
        Convert existing buffers from BufferWrapper to DaskBufferWrapper

        A refactoring of the UDF backend would remove the need for this method.

        :meta private:
        """
        return DaskBufferWrapper.from_buffer(buf)

    def register_master_udfs(self, udfs):
        """
        Give the executor a reference to the udfs instantiated
        on the main node, for introspection purposes

        :meta private:
        """
        self._udfs = udfs

    def _compute(self, *args, udfs=None, user_backends=None, traverse=True, **kwargs):
        """
        Acts as dask.compute(*args, **kwargs) but with knowledge
        of Libertem data structures and compute resources
        """
        if 'resources' in kwargs:
            if udfs is not None:
                raise ValueError('Cannot specify both udfs for resources and resources to use')
            resources = kwargs.get('resources')
        elif udfs is not None:
            resources = self.get_resources(udfs, user_backends=user_backends)
        else:
            resources = None
        kwargs['resources'] = resources

        to_unpack = tuple(a for a in args)
        unwrapped_args = tuple(self.unwrap_results(a) for a in to_unpack)
        results = dask.compute(*unwrapped_args, traverse=traverse, **kwargs)
        if len(args) == 1:
            if len(results) > 1:
                raise RuntimeWarning(f'Unexpected number of results {len(results)} '
                                     'from dask.compute, dropping extras')
            results = results[0]
        return results

    @staticmethod
    def get_resources_from_udfs(udfs, user_backends=None):
        """
        Returns the resources required by the udfs passed as
        argument, excluding those not in the tuple user_backends
        """
        if user_backends is None:
            user_backends = tuple()
        if isinstance(udfs, UDF):
            udfs = [udfs]
        backends = [udf.get_backends() for udf in udfs]
        return get_resources_for_backends(backends, user_backends)

    @staticmethod
    def unwrap_results(results):
        unpackable = {**delayed_unpack.default_unpackable(),
                      UDFData: lambda x: x._data.items(),
                      DaskPreallocBufferWrapper: lambda x: [(0, x.data)],
                      DaskBufferWrapper: lambda x: [(0, x.data)],
                      }

        res_unpack = delayed_unpack.flatten_nested(results, unpackable_types=unpackable)
        flat_mapping = delayed_unpack.build_mapping(results, unpackable_types=unpackable)
        flat_mapping_reduced = [el[:-1] if issubclass(el[-1][0], BufferWrapper) else el
                                for el in flat_mapping]
        return delayed_unpack.rebuild_nested(res_unpack, flat_mapping_reduced)

    def get_udf_runner(self) -> type['UDFRunner']:
        return DelayedUDFRunner


def make_copy(array_dict):
    for k, v in array_dict.items():
        if not v.flags['WRITEABLE']:
            array_dict[k] = v.copy()
    return array_dict


def merge_wrap(udf, dest_dict, src_dict, raw_damage):
    """
    The function called as delayed, acting as a wrapper
    to return a flat list of results rather than a structure
    of UDFData or MergeAttrMapping

    :meta private:
    """
    # Have to make a copy of dest buffers because Dask brings
    # data into the delayed function as read-only np arrays
    # I experimented with setting WRITEABLE to True but this
    # resulted in errors in the final array
    dest_dict = make_copy(dest_dict)

    dest = MergeAttrMapping(dest_dict)
    src = MergeAttrMapping(src_dict)

    # In place merge into the copy of dest
    udf.meta.set_valid_nav_mask(raw_damage)
    udf.merge(dest=dest, src=src)
    # Return flat list of results so they can be unpacked later
    return delayed_unpack.flatten_nested(dest._dict)


def task_wrap(task, *args, **kwargs):
    """
    Flatten the structure tuple(udf.results for udf in self._udfs)
    where udf.results is an instance of UDFData(data={'name':BufferWrapper,...})
    into a simple list [np.ndarray, np.ndarray, ...]

    :meta private:
    """
    res = task(*args, **kwargs)
    res = tuple(r._data for r in res)
    flat_res = delayed_unpack.flatten_nested(res)
    return [r._data for r in flat_res]


def structure_from_task(udfs, task):
    """
    Based on the instantiated whole dataset UDFs and the task
    information, build a description of the expected UDF results
    for the task's partition like:

    :code:`({'buffer_name': StructDescriptor(shape, dtype, extra_shape, buffer_kind), ...}, ...)`

    :meta private:
    """
    structure = []
    for udf in udfs:
        res_data = {}
        for buffer_name, buffer in udf.results.items():
            part_buf_extra_shape = buffer.extra_shape
            buffer.set_shape_partition(task.partition, roi=buffer._roi)
            part_buf_shape = buffer.shape
            part_buf_dtype = buffer.dtype
            res_data[buffer_name] = \
                delayed_unpack.StructDescriptor(np.ndarray,
                                                shape=part_buf_shape,
                                                dtype=part_buf_dtype,
                                                kind=buffer.kind,
                                                extra_shape=part_buf_extra_shape)
        results_container = res_data
        structure.append(results_container)
    return tuple(structure)


def delayed_to_buffer_wrappers(flat_delayed, flat_structure, partition, roi=None, as_buffer=True):
    """
    Take the iterable Delayed results object, and re-wrap each Delayed object
    back into a BufferWrapper wrapping a dask.array of the correct shape and dtype

    :meta private:
    """
    wrapped_res = []
    for el, descriptor in zip(flat_delayed, flat_structure):
        buffer_kind = descriptor.kwargs.pop('kind')
        extra_shape = descriptor.kwargs.pop('extra_shape')
        buffer_dask = da.from_delayed(el, *descriptor.args, **descriptor.kwargs)
        if as_buffer:
            buffer = DaskBufferWrapper(buffer_kind,
                                       extra_shape=extra_shape,
                                       dtype=descriptor.kwargs['dtype'])
            # Need to test whether roi=None here is a problem
            buffer.set_shape_partition(partition, roi=roi)
            buffer.replace_array(buffer_dask)
            wrapped_res.append(buffer)
        else:
            wrapped_res.append(buffer_dask)
    return wrapped_res
