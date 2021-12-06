import dask.array as da
from dask import delayed
import numpy as np
from functools import partial

import libertem.common.buffers
import libertem.io.dataset.raw
import libertem.udf.base
import libertem.executor.delayed
from libertem.common.math import prod
from libertem.common.slice import Slice
from libertem.executor.base import Environment

import delayed_unpack
from dask_inplace import DaskInplaceBufferWrapper

global_ds_shape = None
n_nav_chunks = -1  # number of nav chunks in 0th nav dimension == num_partitions
n_sig_chunk = 1  # number of sig chunks in 0th sig dimension


def build_increasing_ds(array, axis, mode='arange'):
    """
    Applies either range(len(axis)) or linspace(0, 1) to axis of array
    Used to make a dummy dataset more interesting than just np.ones!
    """
    ds_shape = array.shape
    multishape = tuple(v if idx == axis else 1 for idx, v in enumerate(ds_shape))
    if mode == 'arange':
        multi = np.arange(ds_shape[axis])
    elif mode == 'linspace':
        multi = np.linspace(0., 1., num=ds_shape[axis], endpoint=True)
    else:
        raise
    return array * multi.reshape(multishape)


def get_chunks(dimension, n_chunks):
    """
    Util function to split a dimension into n_chunks
    n_chunks == -1 will return a single chunk
    """
    div = dimension // n_chunks
    if div <= 0:
        chunks = (dimension,)
    else:
        chunks = (div,) * n_chunks
        if div * n_chunks < dimension:
            chunks += (dimension % n_chunks,)
    return chunks


def get_num_partitions(self):
    """
    Necessary as we don't have a num_partitions argument to RawFileDataset
    """
    return n_nav_chunks

# do no evil
libertem.io.dataset.raw.RawFileDataSet.get_num_partitions = get_num_partitions


def init_result_buffers(self) -> None:
    """
    Use the monkeypatched attribute udf._allocate_dask_buffers to
    flag udf.results (UDFData) to allocate with allocate_dask
    and not the normal allocate function

    This monkeypatched attribute is present on the main node
    but not on children UDFs created on workers, so we only create
    numpy buffers on worker processes, as normal.
    """
    self.results = libertem.udf.base.UDFData(self.get_result_buffers())
    try:
        if self._allocate_dask_buffers:
            self.results._allocate_as_dask = True
    except AttributeError:
        self.results._allocate_as_dask = False

libertem.udf.base.UDFBase.init_result_buffers = init_result_buffers


def allocate_for_full(self, dataset, roi) -> None:
    """
    Read the monkeypatched attribute UDFData._allocate_as_dask
    set in udf.init_result_buffers() to either create Dask-backed
    buffers or normal numpy-backed buffers.

    This should only create Dask buffers on the main node and never
    on a worker node/UDFTask.
    """
    for k, buf in self._get_buffers():
        buf.set_shape_ds(dataset.shape, roi)
    for k, buf in self._get_buffers(filter_allocated=True):
        if self._allocate_as_dask:
            buf.allocate_dask()
        else:
            buf.allocate()

libertem.udf.base.UDFData.allocate_for_full = allocate_for_full

def allocate_dask(self, lib=None):
    """
    Special BufferWrapper allocate for dask array buffers
    The slightly ugly handling of nav/sig and chunking via
    globals is simply because the interface is not there,
    a real implementation would do this better!
    """
    if self._kind == 'nav':
        flat_nav_shape = prod(global_ds_shape.nav)
        _full_chunking = get_chunks(flat_nav_shape, n_nav_chunks)
        if self._shape == (flat_nav_shape,):
            _buf_chunking = _full_chunking
        elif self._shape[0] in _full_chunking:
            # This branch should never be taken if we are only allocating
            # with dask on the main node and not inside UDFTasks
            _buf_chunking = self._shape
        else:
            raise RuntimeError('Unrecognized buffer size relative to ds/chunk globals')
    elif self._kind == 'sig':
        _buf_chunking = (get_chunks(global_ds_shape.sig[0], n_sig_chunk), (global_ds_shape.sig[1],))
    elif self._kind == 'single':
        _buf_chunking = self._shape
    else:
        raise NotImplementedError('Unrecognized buffer kind')
    _z = partial(da.zeros, chunks=_buf_chunking)
    self._data = _z(self._shape, dtype=self._dtype)

# see no evil
libertem.common.buffers.BufferWrapper.allocate_dask = allocate_dask

def dask_get_slice(self, slice: Slice):
    """
    Insert a wrapper around the view of the buffer
    such that view[:] = array is done inplace
    even when self._data is a Dask array

    If the buffer is not a dask array, just return the normal view
    """
    if isinstance(self._data, da.Array):
        real_slice = slice.get()
        inplace_wrapper = DaskInplaceBufferWrapper(self._data)
        inplace_wrapper.set_slice(real_slice)
        return inplace_wrapper
    else:
        real_slice = slice.get()
        result = self._data[real_slice]
        # Defend against #1026 (internal bugs), allow deactivating in
        # optimized builds for performance
        assert result.shape == tuple(slice.shape) + self.extra_shape
        return result

# hear no evil
libertem.common.buffers.BufferWrapper._get_slice = dask_get_slice

def dask_export(self):
    """
    Current to_numpy doesn't know how to handle da.array's
    """
    self._data = self._data

# say no evil
libertem.common.buffers.BufferWrapper.export = dask_export


def flat_udf_results(self, params, env):
    """
    Flatten the structure tuple(udf.results for udf in self._udfs)
    where udf.results is an instance of UDFData(data={'name':BufferWrapper,...})
    into a simple list [np.ndarray, np.ndarray, ...]
    """
    udfs = [
        cls.new_for_partition(kwargs, self.partition, params.roi)
        for cls, kwargs in zip(self._udf_classes, params.kwargs)
    ]

    res = libertem.udf.base.UDFRunner(udfs).run_for_partition(self.partition, params, env)
    res = tuple(r._data for r in res)
    flat_res = delayed_unpack.flatten_nested(res)
    return [r._data for r in flat_res]

libertem.udf.base.UDFTask.__call__ = flat_udf_results


def structure_from_task(udfs, task):
    """
    Based on the instantiated whole dataset UDFs and the task
    information, build a description of the expected UDF results
    for the task's partition like:
       ({'buffer_name': StructDescriptor(shape, dtype, extra_shape, buffer_kind), ...}, ...)
    """
    structure = []
    partition_shape = task.partition.shape
    for udf in udfs:
        res_data = {}
        for buffer_name, buffer in udf.results.items():
            if buffer.kind == 'sig':
                part_buf_shape = partition_shape.sig
            elif buffer.kind == 'nav':
                part_buf_shape = partition_shape.nav
            elif buffer.kind == 'single':
                part_buf_shape = buffer.shape[:1]
            else:
                raise NotImplementedError
            part_buf_dtype = buffer.dtype
            part_buf_extra_shape = buffer.extra_shape
            part_buf_shape = part_buf_shape + part_buf_extra_shape
            res_data[buffer_name] = \
                delayed_unpack.StructDescriptor(np.ndarray,
                                                shape=part_buf_shape,
                                                dtype=part_buf_dtype,
                                                kind=buffer.kind,
                                                extra_shape=part_buf_extra_shape)
        results_container = res_data
        structure.append(results_container)
    return tuple(structure)


def delayed_to_buffer_wrappers(flat_delayed, flat_structure, partition, as_buffer=True):
    """
    Take the iterable Delayed results object, and re-wrap each Delayed object
    back into a BufferWrapper wrapping a dask.array of the correct shape and dtype
    """
    wrapped_res = []
    for el, descriptor in zip(flat_delayed, flat_structure):
        buffer_kind = descriptor.kwargs.pop('kind')
        extra_shape = descriptor.kwargs.pop('extra_shape')
        buffer_dask = da.from_delayed(el, *descriptor.args, **descriptor.kwargs)
        if as_buffer:
            buffer = libertem.common.buffers.BufferWrapper(buffer_kind,
                                                           extra_shape=extra_shape,
                                                           dtype=descriptor.kwargs['dtype'])
            buffer.set_shape_partition(partition, roi=None)
            buffer._data = buffer_dask
            wrapped_res.append(buffer)
        else:
            wrapped_res.append(buffer_dask)
    return wrapped_res


udfs = None  # No choice but to global here to pass instantiated udfs to executor.run_tasks


def run_tasks(
    self,
    tasks,
    params_handle,
    cancel_id,
):
    """
    Intercept the call to

        results = delayed(task)()

    to infer the expected results structure, then re-build the normal

        tuple(udf.results for udf in self._udfs)

    return value from the Delayed result, but with each buffer
    backed by Dask arrays instead of normal np.arrays
    """
    global udfs
    env = Environment(threads_per_worker=1)
    for task in tasks:
        structure = structure_from_task(udfs, task)
        flat_structure = delayed_unpack.flatten_nested(structure)
        flat_mapping = delayed_unpack.build_mapping(structure)
        result = delayed(task, nout=len(flat_structure))(env=env, params=params_handle)
        wrapped_res = delayed_to_buffer_wrappers(result, flat_structure, task.partition)
        renested = delayed_unpack.rebuild_nested(wrapped_res, flat_mapping)
        result = tuple(libertem.udf.base.UDFData(data=res) for res in renested)
        yield result, task

libertem.executor.delayed.DelayedJobExecutor.run_tasks = run_tasks

"""
Currently run_wrap (delayed) is only used for merging and finalising results
With the unpacking in run_tasks this is no longer necessary, so we can run
the merge/finalise results as a regular function call without using delayed
"""
libertem.executor.delayed.DelayedJobExecutor.run_wrap =\
    libertem.executor.delayed.DelayedJobExecutor.run_function


def do_copy(merge_mapping):
    merge_mapping._dict = {k: v.copy() for k, v in merge_mapping._dict.items()}


def merge_wrap(udf, dest_dict, src_dict):
    dest = libertem.udf.base.MergeAttrMapping(dest_dict)
    do_copy(dest)
    src = libertem.udf.base.MergeAttrMapping(src_dict)
    do_copy(src)

    udf.merge(
        dest=dest,
        src=src
    )
    return delayed_unpack.flatten_nested(dest._dict)


# Not a method of UDFRunner to avoid potentially including self in dask.delayed
def delayed_apply_part_result(udfs, damage, part_results, task):
    for results, udf in zip(part_results, udfs):
        udf.set_views_for_partition(task.partition)

        dest = udf.results.get_proxy()
        src = results.get_proxy()
        dest_dict = {k: v.unwrap_sliced() for k, v in dest._dict.items()}
        src_dict = {k: v for k, v in src._dict.items()}

        structure = structure_from_task([udf], task)[0]
        flat_structure = delayed_unpack.flatten_nested(structure)
        flat_mapping = delayed_unpack.build_mapping(structure)
        merged_del = delayed(merge_wrap, nout=len(flat_mapping))(udf,
                                                                 dest_dict,
                                                                 src_dict)
        wrapped_res = delayed_to_buffer_wrappers(merged_del, flat_structure,
                                                 task.partition, as_buffer=False)
        renested = delayed_unpack.rebuild_nested(wrapped_res, flat_mapping)
        
        src = libertem.udf.base.MergeAttrMapping(renested)

        for k in dest:
            getattr(dest, k)[:] = getattr(src, k)

    v = damage.get_view_for_partition(task.partition)
    v[:] = True
    return (udfs, damage)

libertem.udf.base._apply_part_result = delayed_apply_part_result

if __name__ == '__main__':
    import pathlib
    import libertem.api as lt
    from libertem.executor.delayed import DelayedJobExecutor
    from libertem.executor.inline import InlineJobExecutor
    from libertem.udf.sumsigudf import SumSigUDF
    from libertem.udf.stddev import StdDevUDF
    from libertem.udf.sum import SumUDF
    from libertem.common.shape import Shape
    import matplotlib.pyplot as plt

    dtype = np.float32
    global_ds_shape = Shape((5, 10, 64, 64), sig_dims=2)
    data = np.ones(tuple(global_ds_shape), dtype=dtype)
    for i, mode in enumerate(['arange'] * 2 + ['linspace'] * 2):
        data = build_increasing_ds(data, i, mode=mode)
    data = data.astype(dtype)

    # Write dataset to file so we can load via 'raw'
    rawpath = pathlib.Path('.') / 'test.raw'
    rawfile = rawpath.open(mode='wb').write(data.data)

    executor = DelayedJobExecutor()
    ctx = lt.Context(executor=executor)

    # Load ds and force partitioning only on nav[0] via monkeypatched get_num_parititions
    n_nav_chunks = global_ds_shape.nav[0]  # used in BufferWrapper.allocate
    n_sig_chunk = -1  # this is a global used only in BufferWrapper.allocate
    ds = ctx.load('raw', rawpath, dtype=dtype,
                  nav_shape=global_ds_shape.nav,
                  sig_shape=global_ds_shape.sig)
    sigsum_udf = SumSigUDF()
    sigsum_udf._allocate_dask_buffers = True
    navsum_udf = SumUDF()
    navsum_udf._allocate_dask_buffers = True
    stddev_udf = StdDevUDF()
    stddev_udf._allocate_dask_buffers = True
    udfs = [sigsum_udf, navsum_udf, stddev_udf]

    res = ctx.run_udf(dataset=ds, udf=udfs)
    sigsum_intensity = res[0]['intensity'].data
    navsum_intensity = res[1]['intensity'].data

    try:
        sigsum_intensity.visualize('sigsum_direct.png')
        navsum_intensity.visualize('navsum_direct.png')
    except Exception:
        print('Failed to create task graph PNGs')

    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(sigsum_intensity.compute())
    axs[0].set_title('SigSum over Nav')
    axs[1].imshow(navsum_intensity.compute())
    axs[1].set_title('NavSum over Sig')
    plt.show()

    try:
        rawpath.unlink()
    except OSError:
        pass
