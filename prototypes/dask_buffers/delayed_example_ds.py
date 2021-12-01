import dask.array as da
import numpy as np
from functools import partial

import libertem.common.buffers
import libertem.io.dataset.raw
from libertem.common.math import prod
from libertem.common.slice import Slice


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

    res = ctx.run_udf(dataset=ds, udf=[sigsum_udf, navsum_udf])
    sigsum_intensity = res[0]['intensity'].data
    navsum_intensity = res[1]['intensity'].data

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
