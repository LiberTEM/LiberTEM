import collections
import functools 

import scipy.linalg as la
import numpy as np

from libertem.udf.stddev import merge as merge_stddev
from libertem.masks import _make_circular_mask, radial_bins
from libertem.common.buffers import BufferWrapper

VariancePart = collections.namedtuple('VariancePart', ['var', 'sum_im', 'N'])

def batch_buffer():
    """
    Initializes BufferWrapper objects for sum of variances,
    sum of frames, and the number of frames

    Returns
    -------
    A dictionary that maps 'var', 'std', 'mean', 'num_frame', 'sum_frame' to
    the corresponding BufferWrapper objects
    """
    return {
        'var': BufferWrapper(
            kind='sig', dtype='float32'
            ),
        'num_frame': BufferWrapper(
            kind='single', dtype='float32'
            ),
        'sum_frame': BufferWrapper(
            kind='sig', dtype='float32'
            )
    }


def compute_batch(frame, var, sum_frame, num_frame):
    """
    Given a frame, update sum of variances, sum of frames,
    and the number of total frames

    Parameters
    ----------
    frame
        single frame of the data

    var
        Buffer that stores sum of variances of the previous set of frames

    sum_frame
        Buffer that sores sum of frames of the previous set of frames

    num_frame
        Buffer that stores the number of frames used for computation

    """
    if num_frame == 0:
        var[:] = 0

    else:
        p0 = VariancePart(var=var, sum_im=sum_frame, N=num_frame)
        p1 = VariancePart(var=0, sum_im=frame, N=1)
        compute_merge = merge(p0, p1)

        var[:] = compute_merge.var

    sum_frame[:] += frame
    num_frame[:] += 1


def batch_merge(dest, src):
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
    p0 = VariancePart(var=dest['var'][:],
                    sum_im=dest['sum_frame'][:],
                    N=dest['num_frame'][:])
    p1 = VariancePart(var=src['var'][:],
                    sum_im=src['sum_frame'][:],
                    N=src['num_frame'][:])
    compute_merge = merge(p0, p1)

    dest['var'][:] = compute_merge.var
    dest['sum_frame'][:] = compute_merge.sum_im
    dest['num_frame'][:] = compute_merge.N


def merge(p0, p1):
    """
    Given two sets of partitions, with sum of frames
    and sum of variances, compute joint sum of frames
    and sum of variances using one pass algorithm

    Parameters
    ----------
    p0
        Contains information about the first partition, including
        sum of variances, sum of pixels, and number of frames used

    p1
        Contains information about the second partition, including
        sum of variances, sum of pixels, and number of frames used

    Returns
    -------
    VariancePart
        colletions.namedtuple object that contains information about
        the merged partitions, including sum of variances,
        sum of pixels, and number of frames used
    """
    if p0.N == 0:
        return p1
    N = p0.N + p1.N

    # compute mean for each partitions
    mean_A = (p0.sum_im / p0.N)
    mean_B = (p1.sum_im / p1.N)

    # compute mean for joint samples
    delta = mean_B - mean_A
    mean = mean_A + (p1.N * delta) / (p0.N + p1.N)

    # compute sum of images for joint samples
    sum_im_AB = p0.sum_im + p1.sum_im

    # compute sum of variances for joint samples
    delta_P = mean_B - mean
    var_AB = p0.var + p1.var + (p1.N * delta * delta_P)

    return VariancePart(var=var_AB, sum_im=sum_im_AB, N=N)


def run_pca(ctx, dataset):
    """
    Compute sum of variances and sum of pixels from the given dataset

    One-pass algorithm used in this code is taken from the following paper:
    "Numerically Stable Parallel Computation of (Co-) Variance"
    DOI : https://doi.org/10.1145/3221269.3223036

    Parameters
    ----------
    ctx
        Context class that contains methods for loading datasets, creating jobs on them
        and running them

    dataset
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
    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=compute_batch,
        make_buffers=batch_buffer,
        merge=batch_merge,
    )

    cov = pass_results['var'].data/pass_results['num_frame'].data
    eigvals, eigvecs = la.eig(cov)
    singular_vals = np.sqrt(eigvals)

    return singular_vals, eigvecs

def merge_pca(p0, p1):
    """
    Given two sets of SVD results, merge them into
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
    D0, D1 = p0.singular_vals, p1.singular_vals
    V0, V1 = p0.components, p1.components

    S0 = reduce(np.dot, [V0.T, np.square(np.diag(D0)), V0])
    S1 = reduce(np.dot, [V1.T, np.square(np.diag(D1)), V1])

    frame_size = p0.components.shape[1]
    assert S0.shape == (frame_size, frame_size)
    assert S1.shape == (frame_size, frame_size)

    new_S = S0 + S1

    stddev1 = stddev(
                        var=p0.var,
                        sum_im=p0.sum_im,
                        N=p0.num_frame,
                        )
    stddev2 = stddev(
                        var=p1.var,
                        sum_im=p1.sum_im,
                        N=p1.num_frame,
                        )

    updated = merge_stddev(stddev1, stddev2)

    return pca(
        components=new_S,
        singular_vals=None,
        left_singular=None,
        num_frame=updated.N,
        var=updated.var,
        sum_im=updated.sum_im,
        )

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



def incremental_svd_frame(frame, num_frame, sum_im, var, singular_vals, components, left_singular, intermediate):
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
    n_components = singular_vals[:].size

    if num_frame[:] == 0:

        U, D, V = la.svd(frame_flattened, full_matrices=False)

        assert U.shape == (1, 1)
        assert D.shape == (1,)
        assert V.shape == (1, frame.size)

        var[:] = 0
        sum_im[:] = frame
        num_frame[:] = 1
        singular_vals[:] = D
        components[:] = V
        left_singular[:] = U

        # return pca(
        #         N=1,
        #         sum_im=frame,
        #         var=0,
        #         singular_vals=D,
        #         components=V,
        #         left_singular=U,
        #         n_components=frame.size
        #         )

    stddev_frame = stddev(
                        var=0,
                        sum_im=frame,
                        N=1
                        )
    stddev_prev = stddev(
                        var=var[:],
                        sum_im=sum_im[:],
                        N=num_frame[:]
                        )

    updated = merge_stddev(stddev_prev, stddev_frame)
    total_sum, total_var, n_total = updated.sum_im, updated.var/updated.N, updated.N

    # whitening
    corrected_mean = (np.sqrt(num_frame[:] / n_total)
                    * (sum_im[:] / num_frame[:]))

    corrected_mean = corrected_mean.reshape(1, frame.size)

    assert corrected_mean.reshape(1, frame.size).shape == frame_flattened.shape
    assert singular_vals[:].reshape((-1, 1)).shape == (n_components, 1)

    X = np.vstack((singular_vals[:].reshape((-1, 1)) * components[:], 
                    frame_flattened, 
                    corrected_mean))

    U, D, V = la.svd(X, full_matrices=False)
    U, V = flip_svd(U, V)

    # Update singular values vector D
    singular_vals[:][:D[:n_components].size] = D[:n_components]

    # Update right singular matrix V
    if V.shape[0] < n_components:
        components[:][:V.shape[0], :] = V[:n_components, :]
    else:
        components[:][:n_components, :] = V[:n_components, :]

    # Update left singular matrix U
    if U.shape[1] < n_components:
        left_singular[:][:U.shape[0], :U.shape[1]] = U[:, :n_components]
    else:
        left_singular[:][:U.shape[0], :] = U[:, :n_components]

    var[:] = total_var
    sum_im[:] = total_sum
    num_frame[:] = n_total

    # return pca(
    #         N=n_total,
    #         sum_im=total_sum,
    #         var=total_var,
    #         singular_vals=singular_vals,
    #         components=components,
    #         left_singular=U,
    #         n_components=n_components
    #     )

# class Pca(UDF, total_frames, frame_size):

#     self.total_frames = total_frames
#     self.frame_size = frame_size

#     def get_result_buffers(self):
#         """
#         Initialize BufferWrapper object for PCA,

#         Returns
#         -------
#         A dictionary that maps mean, variance, number of frames, 
#         singular values, and Principal Component matrix to the corresponding
#         BufferWrapper objects
#         """
#         return {
#             'n_components': self.buffer(
#                 kind='single', dtype='float32'
#                 ),
#             'sum_im': self.buffer(
#                 kind='sig', dtype='float32'
#                 ),
#             'var': self.buffer(
#                 kind='sig', dtype='float32'
#                 ),
#             'num_frame': self.buffer(
#                 kind='single', dtype='float32'
#                 ),
#             'singular_vals': self.buffer(
#                 kind='single', extra_shape=(self.frame_size,), dtype='float32'
#                 ),
#             'components': self.buffer(
#                 kind='single', extra_shape=(self.total_frames, self.frame_size), dtype='float32'
#                 ),
#             'left_singular': self.buffer(
#                 kind='single', extra_shape=(self.frame_size, self.frame_size), dtype='float32'
#                 )
#             }

#     def merge(self, dest, src):
#         """
#         Given two sets of partitions, with number of components,
#         mean, variance, number of frames used, singular values, and
#         explained variance (by singular values), update the joint
#         mean, variance, number of frames used, singular values, and
#         explained variance

#         Parameters
#         ----------
#         dest
#             Contains information about the first partition, including
#             sum of variances, sum of pixels, and number of frames used

#         src
#             Contains information about the second partition, including
#             sum of variances, sum of pixels, and number of frames used

#         """
#         prev = pca(
#                     var=dest['var'][:],
#                     sum_im=dest['sum_im'][:],
#                     num_frame=dest['num_frame'][:],
#                     singular_val=dest['singular_vals'][:],
#                     components=dest['components'][:],
#                     n_components=dest['n_components'][:],
#                     left_singular=dest['left_singular'][:]
#                     )

#         new = pca(
#                     var=src['var'][:],
#                     sum_im=src['sum_im'][:],
#                     num_frame=src['num_frame'][:],
#                     singular_val=src['singular_vals'][:],
#                     components=src['components'][:],
#                     n_components=src['n_components'][:],
#                     left_singular=src['left_singular'][:]
#                     )

#         compute_merge = merge_svd(prev, new)

#         dest['var'][:] = compute_merge.var
#         dest['sum_im'][:] = compute_merge.sum_im
#         dest['num_frame'][:] = compute_merge.N
#         dest['singular_vals'][:] = compute_merge.singular_vals
#         dest['components'][:] = compute_merge.components
#         dest['left_singular'][:] = compute_merge.left_singular

#     def process_frame(self, frame):
#         """
#         Given a tile, update parameters related to PCA

#         Parameters
#         ----------
#         tile
#             single tile of the data
#         """
#         if self.results.N == 0:
#             compute_merge = incremental_svd_frame(None , frame)

#         else:
#             n_component = self.results.n_component

#             prev = pca(
#                 n_component=self.results.n_component,
#                 sum_im=self.results.sum_im,
#                 var=self.results.var,
#                 N=self.results.num_frame,
#                 singular_values=self.results.singular_vals,
#                 components=self.results.components
#                 )

#             compute_merge = incremental_svd_frame(prev, frame)

#         self.results.var[:] = compute_merge.var
#         self.results.sum_im[:] = compute_merge.sum_im
#         self.results.num_frame[:] = compute_merge.N
#         self.results.singular_vals[:] = compute_merge.singular_vals
#         self.results.components[:] = compute_merge.components
#         self.results.left_singular[:] = compute_merge.left_singular


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

