import numpy as np
import numba

from libertem.udf import UDF


@numba.njit(fastmath=True)
def merge_single(n, n_0, sum_0, varsum_0, n_1, sum_1, varsum_1, mean_1):
    # FIXME manual citation due to issues in CI.
    # Check if :cite:`Schubert2018` works in a future release
    '''
    Basic function to perform numerically stable merge.

    This function is designed to be inlined in a loop over all pixels in a frame
    to merge an individual pixel, with some parts pre-calculated.

    Erich Schubert and Michael Gertz. Numerically stable parallel computation of
    (co-)variance. In `Proceedings of the 30th International Conference on
    Scientific and Statistical Database Management - SSDBM 18`. ACM Press, 2018.
    `doi:10.1145/3221269.3223036 <https://doi.org/10.1145/3221269.3223036>`_

    cite:Schubert2018

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
    # FIXME manual citation due to issues in CI.
    # Check if :cite:`Schubert2018` works in a future release
    """
    Given two sets of buffers, with sum of frames
    and sum of variances, aggregate joint sum of frames
    and sum of variances in destination buffers using one pass
    algorithm.

    This is ther numerical workhorse for :meth:`StdDevUDF.merge`.

    Erich Schubert and Michael Gertz. Numerically stable parallel computation of
    (co-)variance. In `Proceedings of the 30th International Conference on
    Scientific and Statistical Database Management - SSDBM 18`. ACM Press, 2018.
    `doi:10.1145/3221269.3223036 <https://doi.org/10.1145/3221269.3223036>`_

    cite:Schubert2018

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

    BLOCKSIZE = 1024
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
        sumsum2 = 0.
        for frame in range(n_frames):
            sumsum2 += tile[frame, pixel]
        mean2 = sumsum2 / n_frames
        varsum2 = 0.
        for frame in range(n_frames):
            varsum2 += (tile[frame, pixel] - mean2)**2
        sum_inout[pixel], varsum_inout[pixel] = merge_single(
            n,
            n_0, sum_inout[pixel], varsum_inout[pixel],
            n_frames, sumsum2, varsum2, mean2
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
    # FIXME manual citation due to issues in CI.
    # Check if :cite:`Schubert2018` works in a future release
    """
    Compute sum of variances and sum of pixels from the given dataset

    The one-pass algorithm used in this code is taken from the following paper:

    Erich Schubert and Michael Gertz. Numerically stable parallel computation of
    (co-)variance. In `Proceedings of the 30th International Conference on
    Scientific and Statistical Database Management - SSDBM 18`. ACM Press, 2018.
    `doi:10.1145/3221269.3223036 <https://doi.org/10.1145/3221269.3223036>`_

    cite:Schubert2018

    ..versionchanged:: 0.5.0.dev0
        Result buffers have been renamed

    Examples
    --------

    >>> udf = StdDevUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> # Note: These are raw results. Use run_stddev() instead of
    >>> # using the UDF directly to obtain
    >>> # variance, standard deviation and mean
    >>> np.array(result["varsum"])        # variance times number of frames
    array(...)
    >>> np.array(result["num_frames"])  # number of frames for each tile
    array(...)
    >>> np.array(result["sum"])  # sum of all frames
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
        return {
            'varsum': self.buffer(
                kind='sig', dtype='float32'
            ),
            'num_frames': self.buffer(
                kind='single', dtype='object'
            ),
            'sum': self.buffer(
                kind='sig', dtype='float32'
            )
        }

    def preprocess(self):
        self.results.num_frames[:] = dict()

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
        dest_n = _validate_n(dest['num_frames'][0])
        src_n = _validate_n(src['num_frames'][0])

        n = merge(
            dest_n=dest_n,
            dest_sum=dest['sum'].reshape((-1,)),
            dest_varsum=dest['varsum'].reshape((-1,)),
            src_n=src_n,
            src_sum=src['sum'].reshape((-1,)),
            src_varsum=src['varsum'].reshape((-1,)),
        )
        for key in src['num_frames'][0]:
            dest['num_frames'][0][key] = n

    def process_tile(self, tile):
        """
        Calculate a sum and variance minibatch for the tile and update partition buffers
        with it.

        Parameters
        ----------
        tile
            tile of the data
        """

        key = self.meta.slice.discard_nav()

        if key not in self.results.num_frames[0]:
            self.results.num_frames[0][key] = 0

        n_0 = self.results.num_frames[0][key]
        n_1 = tile.shape[0]

        if n_0 == 0:
            self.results.sum[:] = tile.sum(axis=0)
            # ddof changes the number the sum of variances is divided by.
            # Setting it like here avoids multiplying by n_1 to get the sum
            # of variances
            # See https://docs.scipy.org/doc/numpy/reference/generated/numpy.var.html
            self.results.varsum[:] = np.var(tile, axis=0, ddof=n_1 - 1)
            self.results.num_frames[0][key] = n_1
        else:
            self.results.num_frames[0][key] = process_tile(
                tile=tile.reshape((n_1, -1)),
                n_0=n_0,
                sum_inout=self.results.sum.reshape((-1, )),
                varsum_inout=self.results.varsum.reshape((-1, )),
            )


def consolidate_result(udf_result):
    '''
    Calculate variance, mean and standard deviation
    from raw UDF results and consolidate the per-tile frame counter
    into a single value.

    Parameters
    ----------
    udf_result : Dict[str, BufferWrapper]
        UDF result with keys 'sum', 'varsum', 'num_frames'

    Returns
    -------
    pass_results : Dict[str, Union[numpy.ndarray, int]]
        Result dictionary with keys :code:`'sum', 'varsum', 'var', 'std', 'mean'` as
        :class:`numpy.ndarray`, and :code:`'num_frames'` as :code:`int`
    '''
    udf_result = dict(udf_result.items())
    num_frames = _validate_n(udf_result['num_frames'].data[0])

    udf_result['num_frames'] = num_frames
    udf_result['varsum'] = udf_result['varsum'].data
    udf_result['sum'] = udf_result['sum'].data

    udf_result['var'] = udf_result['varsum'] / num_frames
    udf_result['std'] = np.sqrt(udf_result['var'])
    udf_result['mean'] = udf_result['sum'] / num_frames

    return udf_result


def run_stddev(ctx, dataset, roi=None):
    """
    Compute sum of variances and sum of pixels from the given dataset

    One-pass algorithm used in this code is taken from the following paper:
    "Numerically Stable Parallel Computation of (Co-) Variance"
    DOI : https://doi.org/10.1145/3221269.3223036

    ..versionchanged:: 0.5.0.dev0
        Result buffers have been renamed

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
        dataset to work on
    roi : numpy.ndarray
        Region of interest, see :ref:`udf roi` for more information.


    Returns
    -------
    pass_results
        A dictionary of narrays that contains sum of variances, sum of pixels,
        and number of frames used to compute the above statistic

    To retrieve statistic, using the following commands:
    variance : :code:`pass_results['var']`
    standard deviation : pass_results['std']
    sum of pixels : pass_results['sum']
    mean : pass_results['mean']
    number of frames : pass_results['num_frames']
    """
    stddev_udf = StdDevUDF()
    pass_results = ctx.run_udf(dataset=dataset, udf=stddev_udf, roi=roi)

    return consolidate_result(pass_results)
