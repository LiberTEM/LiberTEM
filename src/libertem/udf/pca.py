import collections

import scipy
import numpy as np

from libertem.udf.stddev import merge as merge_stddev
from libertem.masks import _make_circular_mask, radial_bins


PCA = collections.namedtuple('PCA', ['n_components', 'sum_im', 'var', 'N', 'singular_vals', 'components', 'left_singular'])

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
        'n_components': self.buffer(
            kind='single', dtype='float32'
            ),
        'sum_im': self.buffer(
            kind='sig', dtype='float32'
            ),
        'var': self.buffer(
            kind='sig', dtype='float32'
            ),
        'num_frame': self.buffer(
            kind='single', dtype='float32'
            ),
        'singular_vals': self.buffer(
            kind='sig', extra_shape=(,), dtype='float32'
            ),
        'components': self.buffer(
            kind='sig', dtype='float32'
            ),
        'left_singular': self.buffer(
            kind='sig', dtype='float32'
            )
        }


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


def incremental_svd_frame(prev_result, frame):
    """
    Given previous SVD results, characterized by, sum of
    frames, number of frames, variance of frames, singular values,
    and right singular vector matrix, perform Incremental SVD 
    by adding additional frame

    Parameters
    ----------
    prev_result: 
        pca collections namedtuple object that contains
        information about pca performed on the data so far considered

    frame : numpy.array
        A diffraction pattern frame

    Returns
    -------
    pca
        pca collections namedtuple object that contains
        information about merged PCA
    """
    frame_flattened = frame.reshape(1, frame.size)

    if prev_result == None:

        U, D, V = scipy.linalg.svd(frame_flattened, full_matrices=False)

        return pca(
                N=1,
                sum_im=frame,
                var=0,
                singular_vals=D,
                components=V,
                left_singular=U,
                n_components=frame.size
                )

    stddev = collections.namedtuple('stddev', ['var', 'sum_im', 'N'])
    stddev_frame = stddev(
                        var=0,
                        sum_im=frame,
                        N=1
                        )
    stddev_prev = stddev(
                        var=prev_result.var,
                        sum_im=prev_result.sum_im,
                        N=prev_result.N
                        )

    updated = merge_stddev(stddev_prev, stddev_frame)
    total_sum, total_var, n_total = updated.sum_im, updated.var/updated.N, updated.N

    # whitening
    corrected_mean = (np.sqrt(prev_result.N / n_total)
                    * (prev_result.sum_im / prev_result.N))
    X = np.vstack((prev_result.singular_vals.reshape((-1, 1)) * prev_result.components, 
                    frame_flattened, 
                    corrected_mean))

    U, D, V = scipy.linalg.svd(X, full_matrices=False)
    U, V = flip_svd(U, V)

    n_components = prev_result.n_components
    singular_vals = D[:n_components]
    components = V[:n_components]

    return pca(
            N=n_total,
            sum_im=total_sum,
            var=total_var,
            singular_vals=singular_vals,
            components=components,
            left_singular=U,
            n_components=n_components
        )


def merge_svd(p0, p1):
    """
    Given two sets of svd results, merge them into
    a single SVD result

    Parameters
    ----------
    p0
        Contains information about the first partition, including
        sum of frames, number of frames, variance of frames,
        number of principal components, singular values, and
        right singular value matrix

    p1
        Contains information about the second partition, including
        sum of frames, number of frames, variance of frames,
        number of principal components, singular values, and
        right singular value matrix

    Returns
    -------
    pca
        colletions.namedtuple object that contains information about
        the merged partitions, including sum of frames, number of frames,
        variance of frames, number of principal components, singular 
        values, and right singular value matrix
    """
    U1, U2 = p0.left_singular, p1.left_singular
    assert p0.singular_vals.size == p1.singular_vals.size

    m, n1, n2 = U1.shape[0], U1.shape[1], U2.shape[1]

    c = np.dot(U1.T, U2)
    U2 -= np.dot(U1, c)

    q, r = scipy.linalg.qr(U2)

    def pad(mat, padrow, padcol):
        if padrow < 0:
            padrow = 0
        if padcol < 0:
            padcol = 0
        rows, cols = mat.shape
        return np.bmat([
            [mat, np.matrix(np.zeros((rows, padcol)))],
            [np.matrix(np.zeros((padrow, cols + padcol)))],
        ])

    k = np.bmat([
        [np.diag(p0.singular_vals), np.multiply(c, p1.singular_vals)],
        [pad(np.array([]).reshape(0, 0), min(m, n2), n1), np.multiply(r, p1.singular_vals)]
    ])

    U_updated, D_updated, V_updated = scipy.linalg.svd(k, full_matrices=False)
    U_updated, V_updated = flip_svd(U_updated, V_updated)

    stddev1 = stddev(
                        var=p0.var,
                        sum_im=p0.sum_im,
                        N=p0.N
                        )
    stddev2 = stddev(
                        var=p1.var,
                        sum_im=p1.sum_im,
                        N=p1.N
                        )

    updated = merge_stddev(stddev1, stddev2)

    return pca(
        components=V_updated,
        singular_vals=D_updated,
        left_singular=U_updated,
        n_components=p0.n_components,
        N=updated.N,
        var=updated.var,
        sum_im=updated.sum_im
        )

def run_pca(ctx, dataset, roi=None):
    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=incremental_svd_frame,
        make_buffers=get_result_buffers,
        merge=merge_svd
        )

    pass_results = dict(pass_results.items())

    return pass_results


class Pca(UDF):

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
            'n_components': self.buffer(
                kind='single', dtype='float32'
                ),
            'sum_im': self.buffer(
                kind='sig', dtype='float32'
                ),
            'var': self.buffer(
                kind='sig', dtype='float32'
                ),
            'num_frame': self.buffer(
                kind='single', dtype='float32'
                ),
            'singular_vals': self.buffer(
                kind='sig', extra_shape=(,), dtype='float32'
                ),
            'components': self.buffer(
                kind='sig', dtype='float32'
                ),
            'left_singular': self.buffer(
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
                    sum_im=dest['sum_im'][:],
                    num_frame=dest['num_frame'][:],
                    singular_val=dest['singular_vals'][:],
                    components=dest['components'][:],
                    n_components=dest['n_components'][:],
                    left_singular=dest['left_singular'][:]
                    )

        new = pca(
                    var=src['var'][:],
                    sum_im=src['sum_im'][:],
                    num_frame=src['num_frame'][:],
                    singular_val=src['singular_vals'][:],
                    components=src['components'][:],
                    n_components=src['n_components'][:],
                    left_singular=src['left_singular'][:]
                    )

        compute_merge = merge_svd(prev, new)

        dest['var'][:] = compute_merge.var
        dest['sum_im'][:] = compute_merge.sum_im
        dest['num_frame'][:] = compute_merge.N
        dest['singular_vals'][:] = compute_merge.singular_vals
        dest['components'][:] = compute_merge.components
        dest['left_singular'][:] = compute_merge.left_singular

    def process_frame(self, frame):
        """
        Given a tile, update parameters related to PCA

        Parameters
        ----------
        tile
            single tile of the data
        """
        if self.results.N == 0:
            compute_merge = incremental_svd_frame(None , frame)

        else:
            n_component = self.results.n_component

            prev = pca(
                n_component=self.results.n_component,
                sum_im=self.results.sum_im,
                var=self.results.var,
                N=self.results.num_frame,
                singular_values=self.results.singular_vals,
                components=self.results.components
                )

            compute_merge = incremental_svd_frame(prev, frame)

        self.results.var[:] = compute_merge.var
        self.results.sum_im[:] = compute_merge.sum_im
        self.results.num_frame[:] = compute_merge.N
        self.results.singular_vals[:] = compute_merge.singular_vals
        self.results.components[:] = compute_merge.components
        self.results.left_singular[:] = compute_merge.left_singular

# def johnson_lindenstrauss(num_frames, epsilon=0.1):
#     """
#     Compute the lower bound on the dmension
#     needed for the random projection to work within
#     reasonable error bound
    
#     Parameters
#     ----------
#     num_frames : int
#         Number of diffraction frames
#         (i.e., number of rows in the data)

#     Returns
#     -------
#     lower_bound : int
#         Lower bound on the needed number of features
#     """
#     return (4 * np.log(num_frames) /
#         ((epsilon ** 2) / 2 - (epsilon ** 3) / 2)).astype(np.int)


# def random_projection(X):
#     """
#     Perform random projection on the data matrix X
#     to obtain a matrix X' in the smaller embedding dimension
    
#     Parameters
#     ----------
#     X : numpy.array
#         Data matrix 

#     desired_dim : int
#         Desired number of features (dimension of the columns of X)
#         to which the dimension of X gets reduced

#     Returns
#     -------
#     X : numpy.array
#         Projected data matrix
#     """
#     n_frames, n_features = X.shape
#     reduced_dim = johnson_lindenstrauss(n_frames) # minimum number of dimensions

#     gaussian_projection = np.random.normal(size=(n_features, reduced_dim))

#     return X.dot(gaussian_projection)

# def svd_rank_one_update(U, D, V, x):
#     """
#     Given singular value decomposition matrices U (left singular),
#     D (singular vector), V (right singular), and rank-1 matrix x,
#     update the singular value decomposition matrices

#     Parameters
#     ----------
#     U : numpy.array
#         Left singular matrix

#     D : numpy.array
#         Singular vector

#     V : numpy.array
#         Right singular matrix

#     x : numpy.array
#         rank-1 matrix

#     Returns
#     -------
#     U_updated : numpy.array
#         Updated left singular matrix

#     D_updated : numpy.array
#         Updated singular vector

#     V_updated : numpy.array
#         Updated right singular matrix
#     """
#     b = np.zeros(len(D) + 1)
#     b[-1] = 1

#     m = np.transpose(U).dot(x)
#     p = x - U.dot(m)
#     Ra = np.transpose(p).dot(p) ** 0.5
#     P = p / np.transpose(p).dot(p) ** 0.5

#     V = np.vstack([V, np.zeros(len(V))])
#     n = np.transpose(V).dot(b)
#     q = b - V.dot(n)
#     Rb = np.transpose(q).dot(q) ** 0.5
#     Q = q / Rb

#     K = np.diag(np.zeros(len(D) + 1))
#     K[:-1, -1] = m 
#     K[:-1, :-1] = np.diag(D)
#     K[-1, -1] = Ra

#     U_P = np.transpose(np.vstack([np.transpose(U).dot(P)]))
#     V_Q = np.transpose(np.vstack([np.transpose(V).dot(Q)]))

#     D_updated, eig_vec = np.eig(K)

#     U_updated = U_P.dot(eig_vec)
#     V_updated = np.inv(eig_vec).dot(V_Q)

#     return U_updated, 

