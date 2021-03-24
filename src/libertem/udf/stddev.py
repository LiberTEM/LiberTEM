from collections import defaultdict

import numpy as np
import numba

from libertem.udf import UDF
from libertem.common.buffers import reshaped_view


@numba.njit(fastmath=True, cache=True)
def merge_single(n, n_0, sum_0, varsum_0, n_1, sum_1, varsum_1, mean_1):
    '''
    Basic function to perform numerically stable merge.

    This function is designed to be inlined in a loop over all pixels in a frame
    to merge an individual pixel, with some parts pre-calculated.

    Based on :cite:`Schubert2018`.

    Parameters
    ----------
    n : int
        Total number of frames, assumed n == n_0 + n_1
        Pre-calculated since equal for all pixels.
    n_0 : int
        Number of frames aggregated in sum_0 and varsum_0, n_0 > 0.
        The case n_0 == 0 is handled separately in the calling function.
    sum_0, varsum_0 : float
        Aggregate sum and sum of variances
    n_1 : int
        Number of frames to merge from minibatch, n_1 > 0
    sum_1, varsum_1, mean_1 : float
        Minibatch sum, sum of variances and mean. The mean is
        available from the previous minibatch calculation in the innermost
        aggregation loop as a "hot" value.

    Returns
    -------
    sum, varsum : float
        New aggregate sum and sum of variances
    '''
    # compute mean for each partitions
    mean_0 = sum_0 / n_0

    # compute mean for joint samples
    delta = mean_1 - mean_0
    mean = mean_0 + (n_1 * delta) / n

    # compute sum of images for joint samples
    sumsum = sum_0 + sum_1

    # compute sum of variances for joint samples
    partial_delta = mean_1 - mean
    varsum = varsum_0 + varsum_1 + (n_1 * delta * partial_delta)
    return sumsum, varsum


@numba.njit
def merge(dest_n, dest_sum, dest_varsum, src_n, src_sum, src_varsum):
    """
    Given two sets of buffers, with sum of frames
    and sum of variances, aggregate joint sum of frames
    and sum of variances in destination buffers using one pass
    algorithm.

    This is ther numerical workhorse for :meth:`StdDevUDF.merge`.

    Based on :cite:`Schubert2018`.

    Parameters
    ----------
    dest_n : int
        Number of frames aggregated in dest_sum and dest_varsum
        The case :code:`dest_N == 0` is handled correctly.
    dest_sum, dest_varsum : one-dimensional numpy.ndarray
        Aggregation buffers, will be updated: sum of pixels, sum of variances
    src_n : int
        Number of frames aggregated in src_sum and src_varsum
    src_sum, src_varsum : one-dimensional numpy.ndarray
        Source buffers to merge: sum of pixels, sum of variances


    Returns
    -------
    N:
        New number of frames aggregated in aggregation buffer
    """
    if dest_n == 0:
        dest_sum[:] = src_sum
        dest_varsum[:] = src_varsum
        return src_n
    else:
        n = dest_n + src_n

        for pixel in range(len(dest_sum)):
            dest_sum[pixel], dest_varsum[pixel] = merge_single(
                n,
                dest_n, dest_sum[pixel], dest_varsum[pixel],
                src_n, src_sum[pixel], src_varsum[pixel], src_sum[pixel] / src_n
            )
    return n


@numba.njit(fastmath=True)
def process_tile(tile, n_0, sum_inout, varsum_inout):
    '''
    Compute sum and variance of :code:`tile` along navigation axis
    and merge into aggregation buffers. Numerical "workhorse" for
    :meth:`StdDevUDF.process_tile`.

    Parameters
    ----------
    tile : 2-dimensional numpy.ndarray
        Tile with flattened signal dimension
    n_0 : int
        Number of frames already aggegated in sum_inout, varsum_inout
        Cannot be 0 -- the initial case is handled in :meth:`StdDevUDF.process_tile`
        For unknown reasons, handling the case N0 == 0 in this function degraded performance
        significantly. Re-check with a newer compiler version!
    sum_inout, varsum_inout : numpy.ndarray
        Aggregation buffers (1D) with length matching the flattened signal dimension
        of the tile. The tile will be merged into these buffers (call by reference)

    Returns
    -------
    n : int
        New number of frames in aggregation buffer
    '''
    n_frames = tile.shape[0]
    n_pixels = tile.shape[1]
    n = n_0 + n_frames

    BLOCKSIZE = 256
    n_blocks = n_pixels // BLOCKSIZE

    sumsum = np.zeros(BLOCKSIZE, dtype=sum_inout.dtype)
    varsum = np.zeros(BLOCKSIZE, dtype=varsum_inout.dtype)

    for block in range(n_blocks):
        pixel_offset = block*BLOCKSIZE
        sumsum[:] = 0
        for frame in range(n_frames):
            for i in range(BLOCKSIZE):
                sumsum[i] += tile[frame, pixel_offset + i]
        mean = sumsum / n_frames
        varsum[:] = 0
        for frame in range(n_frames):
            for i in range(BLOCKSIZE):
                varsum[i] += (tile[frame, pixel_offset + i] - mean[i])**2
        for i in range(BLOCKSIZE):
            sum_inout[pixel_offset + i], varsum_inout[pixel_offset + i] = merge_single(
                n,
                n_0, sum_inout[pixel_offset + i], varsum_inout[pixel_offset + i],
                n_frames, sumsum[i], varsum[i], mean[i]
            )
    for pixel in range(n_blocks*BLOCKSIZE, n_pixels):
        sumsum_rest = 0.
        for frame in range(n_frames):
            sumsum_rest += tile[frame, pixel]
        mean_rest = sumsum_rest / n_frames
        varsum_rest = 0.
        for frame in range(n_frames):
            varsum_rest += (tile[frame, pixel] - mean_rest)**2
        sum_inout[pixel], varsum_inout[pixel] = merge_single(
            n,
            n_0, sum_inout[pixel], varsum_inout[pixel],
            n_frames, sumsum_rest, varsum_rest, mean_rest
        )
    return n


# Helper function to make sure the frame count
# is consistent at the merge stage
def _validate_n(num_frame):
    if len(num_frame) == 0:
        return 0
    else:
        values = tuple(num_frame.values())
        assert np.all(np.equal(values, values[0]))
        return values[0]


class StdDevUDF(UDF):
    """
    Compute sum of variances and sum of pixels from the given dataset

    The one-pass algorithm used in this code is taken from the following
    paper: :cite:`Schubert2018`.

    ..versionchanged:: 0.5.0
        Result buffers have been renamed

    ..versionchanged:: 0.7.0
        :code:`var`, :code:`mean`, and :code:`std` are now returned directly
        from the UDF via :code:`get_results`.

    Examples
    --------

    >>> udf = StdDevUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> np.array(result["varsum"])        # variance times number of frames
    array(...)
    >>> np.array(result["num_frames"])  # number of frames for each tile
    array(...)
    >>> np.array(result["sum"])  # sum of all frames
    array(...)
    >>> np.array(result["var"])
    array(...)
    >>> np.array(result["mean"])
    array(...)
    >>> np.array(result["std"])
    array(...)
    """
    def get_result_buffers(self):
        """
        Initializes BufferWrapper objects for sum of variances,
        sum of frames, and the number of frames

        Returns
        -------
        dict
            A dictionary that maps 'varsum',  'num_frames', 'sum' to
            the corresponding BufferWrapper objects
        """
        dtype = np.result_type(self.meta.input_dtype, np.float64)
        return {
            'varsum': self.buffer(
                kind='sig', dtype=dtype
            ),
            'num_frames': self.buffer(
                kind='single', dtype='int'
            ),
            'sum': self.buffer(
                kind='sig', dtype=dtype
            ),
            'var': self.buffer(
                kind='sig', dtype=dtype, use='result_only',
            ),
            'std': self.buffer(
                kind='sig', dtype=dtype, use='result_only',
            ),
            'mean': self.buffer(
                kind='sig', dtype=dtype, use='result_only',
            ),
        }

    def get_task_data(self):
        return {
            'num_frames': defaultdict(lambda: 0)
        }

    def postprocess(self):
        self.results.num_frames[:] = _validate_n(self.task_data.num_frames)

    def merge(self, dest, src):
        """
        Given destination and source buffers that contain sum of variances, sum of frames,
        and the number of frames used in each of the buffers, merge the source
        buffers into the destination buffers by computing the joint sum of variances and
        sum of frames over all frames used

        Parameters
        ----------
        dest
            Aggregation bufer that contains sum of variances, sum of frames, and the
            number of frames
        src
            Partial results that contains sum of variances, sum of frames, and the
            number of frames of a partition to be merged into the aggregation buffers
        """
        dest_n = dest.num_frames[0]
        src_n = src.num_frames[0]

        n = merge(
            dest_n=dest_n,
            dest_sum=reshaped_view(dest.sum, (-1,)),
            dest_varsum=reshaped_view(dest.varsum, (-1,)),
            src_n=src_n,
            src_sum=reshaped_view(src.sum, (-1,)),
            src_varsum=reshaped_view(src.varsum, (-1,)),
        )
        dest.num_frames[:] = n

    def process_tile(self, tile):
        """
        Calculate a sum and variance minibatch for the tile and update partition buffers
        with it.

        Parameters
        ----------
        tile
            tile of the data
        """

        key = tile.scheme_idx
        n_0 = self.task_data.num_frames[key]
        n_1 = tile.shape[0]

        if n_0 == 0:
            self.results.sum[:] = tile.sum(axis=0)
            # ddof changes the number the sum of variances is divided by.
            # Setting it like here avoids multiplying by n_1 to get the sum
            # of variances
            # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html
            self.results.varsum[:] = np.var(tile, axis=0, ddof=n_1 - 1)
            self.task_data.num_frames[key] = n_1
        else:
            self.task_data.num_frames[key] = process_tile(
                tile=reshaped_view(tile, (n_1, -1)),
                n_0=n_0,
                sum_inout=reshaped_view(self.results.sum, (-1, )),
                varsum_inout=reshaped_view(self.results.varsum, (-1, )),
            )

    def get_results(self):
        '''
        Calculate variance, mean and standard deviation from raw UDF results

        Returns
        -------
        pass_results : Dict[str, BufferWrapper]
            Result dictionary with keys
            :code:`'sum', 'varsum', 'num_frames', 'var', 'std', and 'mean'`
            as :code:`BufferWrapper`
        '''
        num_frames = int(self.results.num_frames[0])

        var = self.results.varsum / num_frames

        return {
            'var': var,
            'std': np.sqrt(var),
            'mean': self.results.sum / num_frames,
        }


def consolidate_result(udf_result):
    '''
    Calculate variance, mean and standard deviation
    from raw UDF results and consolidate the per-tile frame counter
    into a single value. Convert all result arrays to `ndarray`.

    Note
    ----
    This is mostly here for backwards-compatability - nowadays, 'var', 'std',
    and 'mean' are already calculated in :meth:`StdDevUDF.get_results`.

    Parameters
    ----------
    udf_result : Dict[str, BufferWrapper]
        UDF result with keys 'sum', 'varsum', 'num_frames', 'var', 'std', 'mean'

    Returns
    -------
    pass_results : Dict[str, Union[numpy.ndarray, int]]
        Result dictionary with keys :code:`'sum', 'varsum', 'var', 'std', 'mean'` as
        :class:`numpy.ndarray`, and :code:`'num_frames'` as :code:`int`
    '''
    return {
        'num_frames': udf_result['num_frames'].data[0],
        'varsum': udf_result['varsum'].data,
        'sum': udf_result['sum'].data,
        'var': udf_result['var'].data,
        'std': udf_result['std'].data,
        'mean': udf_result['mean'].data,
    }


def run_stddev(ctx, dataset, roi=None, progress=False):
    """
    Compute sum of variances and sum of pixels from the given dataset

    One-pass algorithm used in this code is taken from the following paper:
    :cite:`Schubert2018`.

    ..versionchanged:: 0.5.0
        Result buffers have been renamed

    ..versionchanged:: 0.5.0
        Added :code:`progress` parameter for progress bar

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
        dataset to work on
    roi : numpy.ndarray
        Region of interest, see :ref:`udf roi` for more information.
    progress : bool, optional
        Show progress bar. Default is :code:`False`.

    Returns
    -------
    pass_results
        A dictionary of narrays that contains sum of variances, sum of pixels,
        and number of frames used to compute the above statistic

    To retrieve statistic, using the following commands:
    variance : :code:`pass_results['var']`
    standard deviation : :code:`pass_results['std']`
    sum of pixels : :code:`pass_results['sum']`
    mean : :code:`pass_results['mean']`
    number of frames : :code:`pass_results['num_frames']`
    """
    stddev_udf = StdDevUDF()
    pass_results = ctx.run_udf(
        dataset=dataset, udf=stddev_udf, roi=roi, progress=progress
    )

    return consolidate_result(pass_results)
