import collections

import numpy as np

from libertem.udf.base import UDF
from libertem.udf.stddev import merge as merge_stddev


PCA = collections.namedtuple('PCA', ['sum_im', 'var', 'N', 'singular_vals', 'components'])


def flip_svd(U, V):
    """
    Adjust the columns of u and the loadings of v such that the
    loadings in the columns in u that are largest in absolute value
    are always positive. This ensures deterministic output from SVD

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


def johnson_lindenstrauss(num_frames, epsilon=0.1):
    """
    Compute the lower bound on the dmension
    needed for the random projection to work within
    reasonable error bound
    
    Parameters
    ----------
    num_frames : int
        Number of diffraction frames
        (i.e., number of rows in the data)

    Returns
    -------
    lower_bound : int
        Lower bound on the needed number of features
    """
    return (4 * np.log(num_frames) /
        ((epsilon ** 2) / 2 - (epsilon ** 3) / 2)).astype(np.int)


def random_projection(X):
    """
    Perform random projection on the data matrix X
    to obtain a matrix X' in the smaller embedding dimension
    
    Parameters
    ----------
    X : numpy.array
        Data matrix 

    desired_dim : int
        Desired number of features (dimension of the columns of X)
        to which the dimension of X gets reduced

    Returns
    -------
    X : numpy.array
        Projected data matrix
    """
    n_frames, n_features = X.shape
    reduced_dim = johnson_lindenstrauss(n_frames) # minimum number of dimensions

    gaussian_projection = np.random.normal(size=(n_features, reduced_dim))

    return X.dot(gaussian_projection)


def incremental_pca(prev_result, data):
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
        information about merged PCAx
    """
    updated = merge_stddev(prev_result, data)

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
    components = V[:n_components]

    return pca(
            N=n_total,
            sum_im=total_mean * n_total,
            var=total_var,
            singular_vals=singular_vals,
            components=components
        )


def run_pca(ctx, dataset):
    """
    """
    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=)
class Pca(UDF, n_components):

    self.n_components = n_components

    def get_result_buffers(self):
        """
        Initialize BufferWrapper object for PCA,

        Returns
        -------
        A dictionary that maps mean, variance, number of frames, 
        singular values, and Principal Component matrix to the corresponding
        BufferWrapper objects
        """
        return {
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
                kind='sig', extra_shape=(n_components,), dtype='float32'
                ),
            'components': self.buffer(
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

        """
        prev = pca(
                    var=dest['var'][:],
                    mean=dest['mean'][:],
                    num_frame=dest['num_frame'][:],
                    singular_val=dest['singular_vals'][:],
                    components=dest['components'][:]
                    )

        new = pca(
                    var=src['var'][:],
                    mean=src['mean'][:],
                    num_frame=src['num_frame'][:],
                    singular_val=src['singular_vals'][:],
                    components=src['components'][:]
                    )

        compute_merge = incremental_pca(prev, new)

        dest['var'][:] = compute_merge.var
        dest['mean'][:] = compute_merge.sum_im / compute_merge.N
        dest['num_frame'][:] = compute_merge.N
        dest['singular_vals'][:] = compute_merge.singular_vals
        dest['components'][:][:self.n_components] = compute_merge.components

    def process_frame(self, frame):
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
            singular_values=self.results.singular_vals,
            components=self.results.components
            )

        sig_dim = self.results.singular_vals.shape
        num_frame = frame.shape[0]

        new = pca(
            sum_im=np.sum(frame, axis=0),
            var=np.var(frame, axis=0),
            N=num_frame,
            singular_values=np.zeros(sig_dim),
            components=np.zeros(sig_dim)
            )

        compute_merge = IncrementalPCA(prev, new)
        n_components = 
        self.results.var[:] = compute_merge.var
        self.results.mean[:] = compute_merge.sum_im / compute_merge.N
        self.results.num_frame[:] = compute_merge.N
        self.results.singular_vals[:] = compute_merge.singular_vals
        self.results.components[:][:self.n_components] = compute_merge.components

def run_pca(ctx, dataset, n_components, roi=None):
    pcajob = PCA(n_components)
    pass_results = ctx.run_udf(dataset=dataset,
                            udf=pcajob,
                            roi=roi)

    pass_results = dict(pass_results.items())

    return pass_results
