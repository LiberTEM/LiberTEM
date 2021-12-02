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


def dask_allocate(self, lib=None):
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
            _buf_chunking = self._shape
        else:
            raise RuntimeError('Unrecognized buffer size relative to ds/chunk globals')
    elif self._kind == 'sig':
        _buf_chunking = (get_chunks(global_ds_shape.sig[0], n_sig_chunk), (global_ds_shape.sig[1],))
    _z = partial(da.zeros, chunks=_buf_chunking)
    self._data = _z(self._shape, dtype=self._dtype)

# see no evil
libertem.common.buffers.BufferWrapper.allocate = dask_allocate

def dask_get_slice(self, slice: Slice):
    """
    Insert a wrapper around the view of the buffer
    such that view[:] = array is done inplace
    even when self._data is a Dask array
    """
    real_slice = slice.get()
    inplace_wrapper = DaskInplaceBufferWrapper(self._data)
    inplace_wrapper.set_slice(real_slice)
    return inplace_wrapper

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
    udfs = [
        cls.new_for_partition(kwargs, self.partition, params.roi)
        for cls, kwargs in zip(self._udf_classes, params.kwargs)
    ]

    res = libertem.udf.base.UDFRunner(udfs).run_for_partition(self.partition, params, env)
    res = tuple(r._data for r in res)
    flat_res = delayed_unpack.flatten_nested(res)
    return [r._data for r in flat_res]

libertem.udf.base.UDFTask.__call__ = flat_udf_results


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


def structure_from_task(udfs, task):
    structure = []
    partition_shape = task.partition.shape
    for udf in udfs:
        res_data = {}
        for buffer_name, buffer in udf.results.items():
            if buffer.kind == 'sig':
                part_buf_shape = partition_shape.sig
            elif buffer.kind == 'nav':
                part_buf_shape = partition_shape.nav
            else:
                raise NotImplementedError
            part_buf_dtype = buffer.dtype
            part_buf_extra_shape = buffer.extra_shape
            part_buf_shape = part_buf_shape + part_buf_extra_shape
            res_data[buffer_name] = delayed_unpack.StructDescriptor(np.ndarray,
                                                                    shape=part_buf_shape,
                                                                    dtype=part_buf_dtype,
                                                                    kind=buffer.kind,
                                                                    extra_shape=part_buf_extra_shape)
        results_container = res_data
        structure.append(results_container)
    return tuple(structure)


def delayed_to_buffer_wrappers(flat_delayed, flat_structure, partition):
    wrapped_res = []
    for el, descriptor in zip(flat_delayed, flat_structure):
        buffer_kind = descriptor.kwargs.pop('kind')
        extra_shape = descriptor.kwargs.pop('extra_shape')
        buffer_dask = da.from_delayed(el, *descriptor.args, **descriptor.kwargs)
        buffer = libertem.common.buffers.BufferWrapper(buffer_kind,
                                                       extra_shape=extra_shape,
                                                       dtype=descriptor.kwargs['dtype'])
        buffer.set_shape_partition(partition, roi=None)
        buffer._data = buffer_dask
        wrapped_res.append(buffer)
    return wrapped_res


udfs = None  # No choice but to global here to pass instantiated udfs to run_tasks


def run_tasks(
    self,
    tasks,
    params_handle,
    cancel_id,
):
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
libertem.executor.delayed.DelayedJobExecutor.run_wrap =\
             libertem.executor.delayed.DelayedJobExecutor.run_function

if __name__ == '__main__':
    import pathlib
    import libertem.api as lt
    from libertem.executor.delayed import DelayedJobExecutor
    from libertem.executor.inline import InlineJobExecutor
    from libertem.udf.sumsigudf import SumSigUDF
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
    n_sig_chunk = 4  # this is a global used only in BufferWrapper.allocate
    ds = ctx.load('raw', rawpath, dtype=dtype,
                  nav_shape=global_ds_shape.nav,
                  sig_shape=global_ds_shape.sig)
    sigsum_udf = SumSigUDF()
    navsum_udf = SumUDF()
    udfs = [sigsum_udf, navsum_udf]

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
