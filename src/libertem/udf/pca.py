import collections

import numpy as np

from libertem.udf import UDF
from libertem.udf.stddev import merge


pca = collections.namedtuple('PCA', ['sum_im', 'var', 'N',  'n_components', 'singular_vals'])


def flip_svd(U, V):
    """
    Adjust the columns of u and the loadings of v such that the
    loadings in the columns in u that are largest in absolute value
    are always positive

    Parameters
    ----------
    U : numpy.array
        Left singular vectors matrix

    V : numpy.array
        Right singular vectors matrix

    Returns
    -------
    U_adjusted : numpy.array
        Adjusted left singular vectors matrix

    V_adjusted : numpy.array
        Adjusted right singular vectors matrix
    """
    max_abs_rows = np.argmax(np.abs(V), axis=1)
    signs = np.sign(V[range(V.shape[0]), max_abs_rows])
    U *= signs
    V *= signs[:, np.newaxis]

    return U, V


def IncrementalPCA(prev_result, data):
    """
    Given previous PCA results, characterized by,
    perform Incremental PCA by adding additional data

    Parameters
    ----------
    prev_result
        pca collections namedtuple object that contains
        information about pca performed on the data so far considered

    data
        pca collections namedtuple object that contains
        information about pca performed on the new data

    Returns
    -------
    pca
        pca collections namedtuple object that contains
        information about merged PCA
    """
    updated = merge(prev_result, data)

    total_mean, total_var, n_total = updated.sum_im/updated.N, updated.var, updated.N

    # whitening
    data_mean = data.sum_im / data.N

    if prev_result.N == 0:
        X = data_mean - total_mean

    else:
        X = data.sum_im - data_mean

        corrected_mean = (np.sqrt((prev_result.N * data.N) / n_total)
                        * (prev_result.mean - data.mean))
        X = np.vstack((prev_result.singular_vals.reshape((-1, 1))
                    * prev_result.n_components, X, corrected_mean))

    U, D, V = np.linalg.svd(X, full_matrices=False)
    U, V = flip_svd(U, V)

    n_components = prev_result.n_components
    singular_vals = D[:n_components]

    return pca(
            N=n_total,
            sum_im=total_mean * n_total,
            var=total_var,
            n_components=n_components,
            singular_vals=singular_vals
        )


class PCA(UDF):

    def get_result_buffers(self):
        """
        Initialize BufferWrapper object for covariance,

        Returns
        -------
        A dictionary that maps number of components, mean, variance,
        number of frames, and singular values matrix to the corresponding
        BufferWrapper objects
        """
        return {
            'n_components': self.buffer(
                kind='single', dtype='int32'
                ),
            'mean': self.buffer(
                kind='sig', dtype='float32'
                ),
            'var': self.buffer(
                kind='sig', dtype='float32'
                ),
            'num_frame': self.buffer(
                kind='single', dtype='float32'
                ),
            'singular_vals': self.buffer(
                kind='sig', dtype='float32'
                )
            }

    def merge(self, dest, src):
        """
        Given two sets of partitions, with number of components,
        mean, variance, number of frames used, singular values, and
        explained variance (by singular values), update the joint
        mean, variance, number of frames used, singular values, and
        explained variance

        Parameters
        ----------
        dest
            Contains information about the first partition, including
            sum of variances, sum of pixels, and number of frames used

        src
            Contains information about the second partition, including
            sum of variances, sum of pixels, and number of frames used

        pca
            colletions.namedtuple object that contains information about
            the merged partitions, including mean, variances,
            number of frames used, singular values for partitions, and
            the number of components
        """
        prev = pca(
                    var=dest['var'][:],
                    mean=dest['mean'][:],
                    num_frame=dest['num_frame'][:],
                    singular_val=dest['singular_vals'][:],
                    n_components=dest['n_compoennts'][:]
                    )
        new = pca(
                    var=src['var'][:],
                    mean=src['mean'][:],
                    num_frame=src['num_frame'][:],
                    singular_val=src['singular_vals'][:],
                    n_components=src['n_compoennts'][:]
                    )

        compute_merge = IncrementalPCA(prev, new)

        dest['var'][:] = compute_merge.var
        dest['mean'][:] = compute_merge.sum_im / compute_merge.N
        dest['num_frame'][:] = compute_merge.N
        dest['singular_vals'][:] = compute_merge.singular_vals
        dest['n_components'][:] = compute_merge.n_components

    def process_tile(self, tile):
        """
        Given a tile, update parameters related to PCA

        Parameters
        ----------
        tile
            single tile of the data
        """
        prev = pca(
            sum_im=self.results.mean * self.results.N,
            var=self.results.var,
            N=self.results.num_frame,
            n_components=self.results.n_components,
            singular_values=self.results.singular_vals
            )

        sig_dim = self.results.singular_vals.shape
        num_frame = tile.shape[0]

        new = pca(
            sum_im=np.sum(tile),
            var=np.var(tile),
            N=num_frame,
            n_components=self.results.n_components,
            singular_values=np.zeros(sig_dim)
            )

        compute_merge = IncrementalPCA(prev, new)

        self.results.var[:] = compute_merge.var
        self.results.mean[:] = compute_merge.sum_im / compute_merge.N
        self.results.num_frame[:] = compute_merge.N
        self.results.singular_vals[:] = compute_merge.singular_vals


def run_pca(ctx, dataset, roi=None):
    pcajob = PCA()
    pass_results = ctx.run_udf(dataset=dataset,
                            udf=pcajob,
                            roi=roi)

    pass_results = dict(pass_results.items())

    return pass_results
