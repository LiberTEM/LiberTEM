import collections

import numpy as np
import numba

from libertem.udf import UDF


VariancePart = collections.namedtuple('VariancePart', ['var', 'sum_im', 'N'])


@numba.njit
def merge(N0, sum_im0, var0, N1, sum_im1, var1):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, compute joint sum of frames
    and sum of variances using one pass algorithm

    Parameters
    ----------
    N0, sum_im0, var0
        Contains information about the first partition:
        Number of frames used, sum of pixels, sum of variances

    N1, sum_im1, var1
        Contains information about the second partition:
        

    Returns
    -------
    N, sum_im, var:
        Number of frames used, sum of pixels, sum of variances of the merged partitions
    """
    if N0 == 0:
        return N1, sum_im1, var1
    if N1 == 0:
        return N0, sum_im0, var0
    N = N0 + N1

    shape = sum_im0.shape

    sum_im0_ = sum_im0.flatten()
    sum_im1_ = sum_im1.flatten()
    sum_im_AB = np.zeros_like(sum_im0_)
    var0_ = var0.flatten()
    var1_ = var1.flatten()
    var_AB = np.zeros_like(var0_)

    le = len(sum_im0_)
    BLOCKSIZE = 16
    n_blocks = le // BLOCKSIZE

    def _merge(sum_im0, var0, sum_im1, var1):
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

    for block in range(n_blocks):
        for i in range(block*BLOCKSIZE, (block+1)*BLOCKSIZE):
            sum_im_AB[i], var_AB[i] = _merge(
                sum_im0_[i], var0_[i], sum_im1_[i], var1_[i]
            )
    for i in range(n_blocks*BLOCKSIZE, le):
        sum_im_AB[i], var_AB[i] = _merge(
            sum_im0_[i], var0_[i], sum_im1_[i], var1_[i]
        )

    return N, sum_im_AB.reshape(shape), var_AB.reshape(shape)


@numba.njit
def tile_sum_var(tile):
    BLOCKSIZE = 16
    shape = tile.shape[1:]
    N = tile.shape[0]
    # Flatten signal dimension
    tile_ = tile.reshape((N, -1))
    le = tile_.shape[1]
    s = np.zeros(le, dtype=tile_.dtype)
    v = np.zeros(le, dtype=tile_.dtype)

    n_blocks = le // BLOCKSIZE

    means = np.zeros(BLOCKSIZE, dtype=tile_.dtype)

    for block in range(n_blocks):
        offset = block*BLOCKSIZE
        for j in range(N):
            for i in range(offset, offset + BLOCKSIZE):
                s[i] += tile_[j, i]
        for i in range(BLOCKSIZE):
            means[i] = s[i + offset] / N
        for j in range(N):
            for i in range(offset, offset + BLOCKSIZE):
                v[i] += (tile_[j, i] - means[i - offset])**2
    for i in range(n_blocks*BLOCKSIZE, le):
        for j in range(N):
            s[i] += tile_[j, i]
        mean = s[i] / N
        for j in range(N):
            v[i] += (tile_[j, i] - mean)**2

    return s.reshape(shape), v.reshape(shape)


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

    One-pass algorithm used in this code is taken from the following paper:
    "Numerically Stable Parallel Computation of (Co-) Variance"
    DOI : https://doi.org/10.1145/3221269.3223036

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
        A dictionary that maps 'var', 'std', 'mean', 'num_frame', 'sum_frame' to
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
        N0 = _validate_n(dest['num_frame'][0])
        sum_im0 = dest['sum_frame'][:]
        var0 = dest['var'][:]

        N1 = _validate_n(src['num_frame'][0])
        sum_im1 = src['sum_frame'][:]
        var1 = src['var'][:]

        N, sum_im, var = merge(N0, sum_im0, var0, N1, sum_im1, var1)

        dest['var'][:] = var
        dest['sum_frame'][:] = sum_im
        for key in src['num_frame'][0]:
            dest['num_frame'][0][key] = N

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

        tile_sum, tile_var = tile_sum_var(tile)

        N0 = self.results.num_frame[0][key]
        sum_im0 = self.results.sum_frame
        var0 = self.results.var

        N1 = tile.shape[0]
        sum_im1 = tile_sum
        var1 = tile_var

        N, sum_im, var = merge(N0, sum_im0, var0, N1, sum_im1, var1)

        self.results.var[:] = var

        self.results.sum_frame[:] = sum_im
        self.results.num_frame[0][key] = N


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
