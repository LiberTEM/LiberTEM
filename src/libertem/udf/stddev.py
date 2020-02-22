import numpy as np
import numba

from libertem.udf import UDF


@numba.njit(fastmath=True)
def _merge(N, N0, sum_im0, var0, N1, sum_im1, var1):
    '''
    Basic function to perform numerically stable merge :cite:`Schubert2018`
    '''
    # compute mean for each partitions
    mean_A = sum_im0 / N0
    mean_B = sum_im1 / N1

    # compute mean for joint samples
    delta = mean_B - mean_A
    mean = mean_A + (N1 * delta) / N

    # compute sum of images for joint samples
    sum_im_AB = sum_im0 + sum_im1

    # compute sum of variances for joint samples
    delta_P = mean_B - mean
    var_AB = var0 + var1 + (N1 * delta * delta_P)
    return sum_im_AB, var_AB


@numba.njit(fastmath=True)
def merge(dest_N, dest_sum, dest_varsum, src_N, src_sum, src_varsum):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, aggregate joint sum of frames
    and sum of variances in destination partition using one pass
    algorithm :cite:`Schubert2018`

    Parameters
    ----------
    dest_N : int
        Number of frames aggregated in dest_sum and dest_varsum
        The case :code:`dest_N == 0` is handled correctly.
    dest_sum, dest_varsum
        Aggregation buffers, will be updated: sum of pixels, sum of variances
    src_N, src_sum, src_varsum
        Information about the second partition


    Returns
    -------
    N:
        New number of frames aggregated in aggregation buffer
    """
    if dest_N == 0:
        dest_sum[:] = src_sum
        dest_varsum[:] = src_varsum
        return src_N
    else:
        N = dest_N + src_N

        le = len(dest_sum)
        for i in range(le):
            dest_sum[i], dest_varsum[i] = _merge(
                N,
                dest_N, dest_sum[i], dest_varsum[i],
                src_N, src_sum[i], src_varsum[i]
            )
    return N


@numba.njit(fastmath=True)
def process_tile(tile, N0, sum_inout, var_inout):
    '''
    Compute sum and variance of :code:`tile` along navigation axis
    and merge into aggregation buffers. Numerical "workhorse" for
    :meth:`StdDevUDF.process_tile`.

    Parameters
    ----------
    tile : numpy.ndarray
        Tile with flattened signal dimension
    N0 : int
        Number of frames already aggegated in partition buffer
        Cannot be 0 -- the initial case is handled in :meth:`StdDevUDF.process_tile`
        For unknown reasons, handling the case N0 == 0 in this function degraded performance
        significantly. Re-check with a newer compiler version!
    sum_inout, var_inout : numpy.ndarray
        Aggregation buffers with flattened signal dimension. The tile
        will be merged into these buffers (call by reference)

    Returns
    -------
    N : int
        New number of frames in aggregation buffer
    '''
    N1 = tile.shape[0]
    le = tile.shape[1]
    N = N0 + N1
    for i in range(le):
        s = 0
        for j in range(N1):
            s += tile[j, i]
        mean = s / N1
        varsum = 0
        for j in range(N1):
            varsum += (tile[j, i] - mean)**2

        sum_inout[i], var_inout[i] = _merge(
            N,
            N0, sum_inout[i], var_inout[i],
            N1, s, varsum
        )
    return N


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

    The one-pass algorithm used in this code is taken from the following paper:
    :cite:`Schubert2018`

    Examples
    --------

    >>> udf = StdDevUDF()
    >>> result = ctx.run_udf(dataset=dataset, udf=udf)
    >>> # Note: These are raw results. Use run_stddev() instead of
    >>> # using the UDF directly to obtain
    >>> # variance, standard deviation and mean
    >>> np.array(result["var"])        # variance times number of frames
    array(...)
    >>> np.array(result["num_frame"])  # number of frames
    array(...)
    >>> np.array(result["sum_frame"])  # sum of all frames
    array(...)
    """

    def get_result_buffers(self):
        """
        Initializes BufferWrapper objects for sum of variances,
        sum of frames, and the number of frames

        Returns
        -------
        A dictionary that maps 'var',  'num_frame', 'sum_frame' to
        the corresponding BufferWrapper objects
        """
        return {
            'var': self.buffer(
                kind='sig', dtype='float32'
            ),
            'num_frame': self.buffer(
                kind='single', dtype='object'
            ),
            'sum_frame': self.buffer(
                kind='sig', dtype='float32'
            )
        }

    def preprocess(self):
        self.results.num_frame[:] = dict()

    def merge(self, dest, src):
        """
        Given two buffers that contain sum of variances, sum of frames,
        and the number of frames used in each of the partitions, merge the
        partitions and compute the joint sum of variances and sum of frames
        over all frames used

        Parameters
        ----------
        dest
            Partial results that contains sum of variances, sum of frames, and the
            number of frames used over all the frames used

        src
            Partial results that contains sum of variances, sum of frames, and the
            number of frames used over current iteration of partition
        """
        dest_N = _validate_n(dest['num_frame'][0])
        src_N = _validate_n(src['num_frame'][0])

        oldshape = dest['sum_frame'][:].shape

        dest['sum_frame'][:].shape = (-1,)
        dest['var'][:].shape = (-1, )
        src['sum_frame'][:].shape = (-1,)
        src['var'][:].shape = (-1, )

        N = merge(
            dest_N=dest_N,
            dest_sum=dest['sum_frame'][:],
            dest_varsum=dest['var'][:],
            src_N=src_N,
            src_sum=src['sum_frame'][:],
            src_varsum=src['var'][:],
        )
        for key in src['num_frame'][0]:
            dest['num_frame'][0][key] = N

        dest['sum_frame'][:].shape = oldshape
        dest['var'][:].shape = oldshape
        src['sum_frame'][:].shape = oldshape
        src['var'][:].shape = oldshape

    def process_tile(self, tile):
        """
        Given a frame, update sum of variances, sum of frames,
        and the number of total frames

        Parameters
        ----------
        tile
            tile of the data
        """

        key = self.meta.slice.discard_nav()

        if key not in self.results.num_frame[0]:
            self.results.num_frame[0][key] = 0

        N0 = self.results.num_frame[0][key]
        N1 = tile.shape[0]

        oldshape = self.results.sum_frame.shape

        tile.shape = (N1, -1)

        self.results.sum_frame.shape = (-1, )
        self.results.var.shape = (-1, )

        if N0 == 0:
            self.results.sum_frame[:] = tile.sum(axis=0)
            self.results.var[:] = np.var(tile, axis=0, ddof=N1-1)
            self.results.num_frame[0][key] = N1
        else:
            self.results.num_frame[0][key] = process_tile(
                tile=tile,
                N0=N0,
                sum_inout=self.results.sum_frame,
                var_inout=self.results.var
            )
        self.results.sum_frame.shape = oldshape
        self.results.var.shape = oldshape
        tile.shape = (N1, *oldshape)


def consolidate_result(udf_result):
    udf_result = dict(udf_result.items())
    num_frame = _validate_n(udf_result['num_frame'].data[0])

    udf_result['var'] = udf_result['var'].data/num_frame
    udf_result['std'] = np.sqrt(udf_result['var'].data)

    udf_result['mean'] = udf_result['sum_frame'].data/num_frame
    udf_result['num_frame'] = num_frame
    udf_result['sum_frame'] = udf_result['sum_frame'].data
    return udf_result


def run_stddev(ctx, dataset, roi=None):
    """
    Compute sum of variances and sum of pixels from the given dataset

    One-pass algorithm used in this code is taken from the following paper:
    "Numerically Stable Parallel Computation of (Co-) Variance"
    DOI : https://doi.org/10.1145/3221269.3223036

    Parameters
    ----------
    ctx : libertem.api.Context

    dataset : libertem.io.dataset.base.DataSet
        dataset to work on

    Returns
    -------
    pass_results
        A dictionary of narrays that contains sum of variances, sum of pixels,
        and number of frames used to compute the above statistic

    To retrieve statistic, using the following commands:
    variance : pass_results['var']
    standard deviation : pass_results['std']
    sum of pixels : pass_results['sum_frame']
    mean : pass_results['mean']
    number of frames : pass_results['num_frame']
    """
    stddev_udf = StdDevUDF()
    pass_results = ctx.run_udf(dataset=dataset, udf=stddev_udf, roi=roi)

    return consolidate_result(pass_results)
