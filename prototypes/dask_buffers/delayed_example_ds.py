import numpy as np
import dask
# dask.config.set(scheduler='synchronous')
import dask.array as da

from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.stddev import StdDevUDF
from libertem.udf.sum import SumUDF


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


def dask_simple_nav_merge(self, ordered_results):
    intensity = da.concatenate([b.intensity for b in ordered_results.values()])
    self.results.get_buffer('intensity').reset_buffer(intensity)

SumSigUDF.dask_merge = dask_simple_nav_merge

def dask_sig_sum_merge(self, ordered_results):
    intensity_chunks = [b.intensity for b in ordered_results.values()]
    intensity_sum = da.stack(intensity_chunks, axis=0).sum(axis=0)
    self.results.get_buffer('intensity').reset_buffer(intensity_sum)

SumUDF.dask_merge = dask_sig_sum_merge

def dask_stddev_merge(self, ordered_results):
    n_frames = da.concatenate([[b.num_frames[0] for b in ordered_results.values()]])
    pixel_sums = da.concatenate([[b.sum for b in ordered_results.values()]])
    pixel_varsums = da.concatenate([[b.varsum for b in ordered_results.values()]])

    n_frames = da.rechunk(n_frames, (-1,) * n_frames.ndim)
    pixel_sums = da.rechunk(pixel_sums, (-1,) * pixel_sums.ndim)
    pixel_varsums = da.rechunk(pixel_varsums, (-1,) * pixel_varsums.ndim)

    # Expand n_frames to be broadcastable
    extra_dims = pixel_sums.ndim - n_frames.ndim
    n_frames = n_frames.reshape(n_frames.shape + (1,) * extra_dims)

    cumulative_frames = da.cumsum(n_frames, axis=0)
    cumulative_sum = da.cumsum(pixel_sums, axis=0)
    sumsum = cumulative_sum[-1, ...]
    total_frames = cumulative_frames[-1, 0]

    mean_0 = cumulative_sum / cumulative_frames
    # Handle the fact that mean_0 is indexed to results from
    # up-to the partition before. We shift everything one to
    # the right, and we don't care about result 0 because it
    # is by definiition replaced with varsum[0, ...]
    mean_0 = da.roll(mean_0, 1, axis=0)

    mean_1 = pixel_sums / n_frames
    delta = mean_1 - mean_0
    mean = mean_0 + (n_frames * delta) / cumulative_frames
    partial_delta = mean_1 - mean
    varsum = pixel_varsums + (n_frames * delta * partial_delta)
    varsum[0, ...] = pixel_varsums[0, ...]
    varsum_cumulative = da.cumsum(varsum, axis=0)
    varsum_total = varsum_cumulative[-1, ...]

    self.results.get_buffer('sum').reset_buffer(sumsum)
    self.results.get_buffer('varsum').reset_buffer(varsum_total)
    self.results.get_buffer('num_frames').reset_buffer(total_frames)

StdDevUDF.dask_merge = dask_stddev_merge

def get_results(self):
    '''
    From StdDevUDF

    Corrects an int() conversion which causes
    a double compute when result buffers are dask arrays
    '''
    # num_frames = int(self.results.num_frames[0])
    num_frames = self.results.num_frames[0]

    var = self.results.varsum / num_frames

    return {
        'var': var,
        'std': np.sqrt(var),
        'mean': self.results.sum / num_frames,
    }

StdDevUDF.get_results = get_results


if __name__ == '__main__':
    import pathlib
    import libertem.api as lt
    from libertem.executor.delayed import DelayedJobExecutor
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
    ds = ctx.load('raw', rawpath, dtype=dtype,
                  nav_shape=global_ds_shape.nav,
                  sig_shape=global_ds_shape.sig)

    sigsum_udf = SumSigUDF()
    navsum_udf = SumUDF()
    stddev_udf = StdDevUDF()
    udfs = [sigsum_udf, navsum_udf, stddev_udf]

    res = ctx.run_udf(dataset=ds, udf=udfs)

    sigsum_dask = res[0]['intensity'].data
    navsum_dask = res[1]['intensity'].data
    stddev_dask = {k: v.data for k, v in res[2].items()}

    sigsum_intensity, navsum_intensity, std_dev_results = dask.compute(sigsum_dask,
                                                                       navsum_dask,
                                                                       stddev_dask)

    try:
        sigsum_dask.visualize('sigsum_direct.png')
        navsum_dask.visualize('navsum_direct.png')
        stddev_dask['var'].visualize('var_direct.png')
        stddev_dask['std'].visualize('std_direct.png')
        stddev_dask['varsum'].visualize('varsum_direct.png')
    except Exception:
        print('Failed to create task graph PNGs')

    fig, axs = plt.subplots(2, 4)
    _axs = axs[0, :]
    _axs[0].imshow(sigsum_dask)
    _axs[0].set_title('SigSum over Nav')
    _axs[1].imshow(navsum_dask)
    _axs[1].set_title('NavSum over Sig')
    _axs[2].imshow(std_dev_results['std'])
    _axs[2].set_title('Std')
    _axs[3].imshow(std_dev_results['sum'])
    _axs[3].set_title('Sum from StdDevUDF')
    _axs = axs[1, :]
    _axs[0].imshow(data.sum(axis=(2, 3)))
    _axs[0].set_title('Numpy sigsum')
    _axs[1].imshow(data.sum(axis=(0, 1)))
    _axs[1].set_title('Numpy navsum')
    _axs[2].imshow(np.std(data, axis=(0, 1)))
    _axs[2].set_title('Numpy std')
    _axs[3].imshow(np.std(data, axis=(0, 1)) / std_dev_results['std'])
    _axs[3].set_title('np.std / StdDevUDF["std"] via dask')
    plt.show()

    try:
        rawpath.unlink()
    except OSError:
        pass
