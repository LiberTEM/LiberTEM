import collections
import functools

import scipy.linalg as la
import numpy as np

from libertem.common.buffers import BufferWrapper


pca = collections.namedtuple('pca', ['num_frame', 'singular_vals', 'components', 'left_singular'])


def get_result_buffers(n_components, frame_size, total_frames):
    """
    Initialize BufferWrapper object for PCA,

    Returns
    -------
    A dictionary that maps mean, variance, number of frames,
    singular values, and Principal Component matrix to the corresponding
    BufferWrapper objects
    """
    return {
        'num_frame': BufferWrapper(
            kind='single', dtype='float32'
            ),
        'singular_vals': BufferWrapper(
            kind='single', extra_shape=(n_components,), dtype='float32'
            ),
        'components': BufferWrapper(
            kind='single', extra_shape=(frame_size, n_components), dtype='float32'
            ),
        'left_singular': BufferWrapper(
            kind='single', extra_shape=(total_frames, n_components), dtype='float32'
            )
        }


def ccfipca(frame, num_frame, left_singular, singular_vals, components):
    """
    Implementation of Candid Covariance free Incremental PCA algorithm.
    As the name suggests, this algorithm does not explicitly computes
    the covariance matrix and thus, can lead to efficient use of memory
    compared to other algorithms that utilizes the covariance matrix,
    which can be arbitrarily large based on the dimension of the data

    Parameters
    """
    num_features, n_components = components[:].shape

    # initialize eigenvalues and eigenspace matrices, if needed
    if num_frame[:] == 0:
        U = np.random.normal(
                            loc=0,
                            scale=1/num_features,
                            size=(num_features, n_components)
                            ).T
        eigvals = np.abs(np.random.normal(
                                        loc=0,
                                        scale=1,
                                        size=(n_components,)
                                        ) / np.sqrt(n_components)
                        )

    amnesic = max(1, num_frame[:]-2) / (num_frame[:] + 1)

    frame_flattened = frame.reshape(frame.size,)

    for i in range(n_components):

        V =  amnesic * eigvals[i] * U[i, :] + (1 - amnesic) * np.dot(frame_flattened.reshape(-1, 1), U[i, :]) * frame_flattened

        # update eigenvalues and eigenspace matrices
        eigvals[i] = np.linalg.norm(V)
        U[i, :] = V / eigvals[i]

        frame_flattened -= np.dot(frame_flattened, U[i, :]) * U[i, :]

    num_frame[:] += 1
    left_singular[:] = U
    singular_vals[:] = np.sqrt(eigvals)


def incremental_pca(frame, num_frame, left_singular, singular_vals, components):
    """
    Given previous SVD results, characterized by, sum of
    frames, number of frames, variance of frames, singular values,
    and right singular vector matrix, perform Incremental SVD
    by adding additional frame

    Parameters
    ----------
    prev_result
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
    error_tolerance = 1e-7
    num_features, n_components = components[:].shape[0], components[:].shape[1]

    # initialize left singular vector matrix and eigenvalues
    if num_frame[:] == 0:
        U = np.random.normal(
                            loc=0,
                            scale=1/num_features,
                            size=(num_features, n_components),
                            )
        eigvals = np.abs(np.random.normal(0, 1, (n_components))) / np.sqrt(n_components)
    else:
        U = components[:]
        eigvals = np.square(singular_vals[:])  # assume positive definite data

    frame_flattened = frame.reshape(frame.size,)
    num_frame[:] += 1

    eigvals *= (1 - 1/num_frame[:])
    frame_flattened *= np.sqrt(1/num_frame[:])

    # project new frame into current estimate to check error
    estimate = U.T.dot(frame_flattened)
    error = frame_flattened - U.dot(estimate)
    error_norm = np.sqrt(error.dot(error))

    if error_norm >= error_tolerance:
        eigvals = np.concatenate((eigvals, [0]))
        estimate = np.concatenate((estimate, [error_norm]))
        U = np.concatenate((U, error[:, np.newaxis] / error_norm), 1)

    M = np.diag(eigvals) + np.outer(estimate, estimate.T)
    d, V = la.eig(M, overwrite_a=True)

    idx = np.argsort(d)[::-1]
    eigvals = d[idx][:n_components]
    V = V[:, idx]
    U = U.dot(V[:, :n_components])

    singular_vals[:] = np.sqrt(eigvals)
    components[:] = U


def merge_svd(p0, p1):
    """
    Given two sets of svd results, merge them into
    a single SVD result

    Parameters
    ----------
    p0
        Contains information abou tthe first partition, including
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
    n_components = p0.singular_vals.size
    U1, U2 = p0.left_singular, p1.left_singular
    S1, S2 = p0.singular_vals, p1.singular_vals
    assert p0.singular_vals.size == p1.singular_vals.size

    k = U1.shape[1]

    Z = np.dot(U1.T, U2)
    Q, R = la.qr(U2 - np.dot(U1, Z), overwrite_a=True)

    S1, S2 = np.diag(S1), np.diag(S2)
    block_mat = np.block([[S1, Z.dot(S2)],
                        [np.zeros((R.dot(S2).shape[0], S1.shape[1])), R.dot(S2)]])

    U_updated, D_updated, V_updated = la.svd(block_mat, full_matrices=False)
    R1, R2 = U_updated[:k, :], U_updated[k:, :]
    U_updated = U1.dot(R1) + Q.dot(R2)

    num_frame = p0.num_frame+p1.num_frame

    return pca(
        components=V_updated[:, :n_components],
        singular_vals=D_updated[:n_components],
        left_singular=U_updated[:, :n_components],
        num_frame=num_frame,
        )


def batch_merge(dest, src):
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
                num_frame=dest['num_frame'][:],
                singular_vals=dest['singular_vals'][:],
                components=dest['components'][:],
                left_singular=dest['left_singular'][:],
                )

    new = pca(
                num_frame=src['num_frame'][:],
                singular_vals=src['singular_vals'][:],
                components=src['components'][:],
                left_singular=src['left_singular'][:],
                )

    compute_merge = merge_svd(prev, new)

    num_frame = compute_merge.num_frame
    components = compute_merge.components
    left_singular = compute_merge.left_singular
    singular_vals = compute_merge.singular_vals

    dest['num_frame'][:] = num_frame
    dest['components'][:][:components.shape[0], :] = components
    dest['left_singular'][:][:left_singular.shape[0], :] = left_singular
    dest['singular_vals'][:] = singular_vals


def run_pca(ctx, dataset, n_components=9):
    frame_size = dataset.shape.sig.size
    total_frames = dataset.shape.nav.size

    pass_results = ctx.run_udf(
        dataset=dataset,
        fn=merge_frame,
        make_buffers=functools.partial(
                                        get_result_buffers,
                                        n_components=n_components,
                                        frame_size=frame_size,
                                        total_frames=total_frames
                                        ),
        merge=batch_merge
        )

    return pass_results
