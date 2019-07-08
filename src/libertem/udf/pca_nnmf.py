import collections

import numpy as np
import scipy.optimize.nnls as nnls
import fbpca

from libertem.common.buffers import BufferWrapper
from libertem.udf import UDF

pca = collections.namedtuple('pca', ['num_frame', 'singular_vals', 'components', 'left_singular'])
nnmf = collections.namedtuple('NNMF', ['score', 'loading'])


class PcaUDF(UDF):
    """
    UDF class for Principal Component Analysis
    """

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
            'num_frame': BufferWrapper(
                kind='single',
                dtype='float32'
                ),

            'singular_vals': BufferWrapper(
                kind='single',
                extra_shape=(self.params.n_components,),
                dtype='float32'
                ),

            'components': BufferWrapper(
                kind='single',
                extra_shape=(self.params.n_components, self.params.frame_size),
                dtype='float32'
                ),

            'left_singular': BufferWrapper(
                kind='single',
                extra_shape=(self.params.total_frames, self.params.n_components),
                dtype='float32'
                ),

            'mean': BufferWrapper(
                kind='single',
                extra_shape=(self.params.frame_size,),
                dtype='float32'
                ),

            'var': BufferWrapper(
                kind='single',
                extra_shape=(self.params.frame_size,),
                dtype='float32'
                ),

            'merge_components': BufferWrapper(
                kind='single',
                extra_shape=(self.params.num_frames, self.params.frame_size),
                dtype='float32'
                ),
            }

    def svd_flip(self, u, v, u_based_decision=False):
        """Sign correction to ensure deterministic output from SVD.
        Adjusts the columns of u and the rows of v such that the loadings in the
        columns in u that are largest in absolute value are always positive.
        Parameters
        ----------
        u : ndarray
            u and v are the output of `linalg.svd` or
            `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
            so one can compute `np.dot(u * s, v)`.
        v : ndarray
            u and v are the output of `linalg.svd` or
            `sklearn.utils.extmath.randomized_svd`, with matching inner dimensions
            so one can compute `np.dot(u * s, v)`.
        u_based_decision : boolean, (default=True)
            If True, use the columns of u as the basis for sign flipping.
            Otherwise, use the rows of v. The choice of which variable to base the
            decision on is generally algorithm dependent.
        Returns
        -------
        u_adjusted, v_adjusted : arrays with the same dimensions as the input.
        """
        if u_based_decision:
            # columns of u, rows of v
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, range(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
        else:
            # rows of v, columns of u
            max_abs_rows = np.argmax(np.abs(v), axis=1)
            signs = np.sign(v[range(v.shape[0]), max_abs_rows])
            u *= signs
            v *= signs[:, np.newaxis]
        return u, v

    def randomized_svd(self, X, n_components=10):
        """
        Perform randomized SVD on the given matrix
        """
        row, col = X.shape

        # transpose = False

        # if row < col:
        #     transpose = True
        #     X = X.T

        rand_matrix = np.random.normal(size=(col, n_components))
        Q, _ = np.linalg.qr(X @ rand_matrix, mode='reduced')

        smaller_matrix = Q.T @ X
        U_hat, S, V = np.linalg.svd(smaller_matrix, full_matrices=False)
        U = Q @ U_hat
        
        # if transpose:
        #     return  U.T, S.T, V.T

        # else:
        return U, S, V

    def _safe_accumulator_op(self, op, x, *args, **kwargs):
        """
        This function provides numpy accumulator functions with a float64 dtype
        when used on a floating point input. This prevents accumulator overflow on
        smaller floating point dtypes.
        Parameters
        ----------
        op : function
            A numpy accumulator function such as np.mean or np.sum
        x : numpy array
            A numpy array to apply the accumulator function
        *args : positional arguments
            Positional arguments passed to the accumulator function after the
            input x
        **kwargs : keyword arguments
            Keyword arguments passed to the accumulator function
        Returns
        -------
        result : The output of the accumulator function passed to this function
        """
        if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
            result = op(x, *args, **kwargs, dtype=np.float64)
        else:
            result = op(x, *args, **kwargs)
        return result

    def _incremental_mean_and_var(self, X, last_mean, last_variance, last_sample_count):
        """Calculate mean update and a Youngs and Cramer variance update.
        last_mean and last_variance are statistics computed at the last step by the
        function. Both must be initialized to 0.0. In case no scaling is required
        last_variance can be None. The mean is always required and returned because
        necessary for the calculation of the variance. last_n_samples_seen is the
        number of samples encountered until now.
        From the paper "Algorithms for computing the sample variance: analysis and
        recommendations", by Chan, Golub, and LeVeque.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data to use for variance update
        last_mean : array-like, shape: (n_features,)
        last_variance : array-like, shape: (n_features,)
        last_sample_count : array-like, shape (n_features,)
        Returns
        -------
        updated_mean : array, shape (n_features,)
        updated_variance : array, shape (n_features,)
            If None, only mean is computed
        updated_sample_count : array, shape (n_features,)
        Notes
        -----
        NaNs are ignored during the algorithm.

        References
        ----------
        T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
            variance: recommendations, The American Statistician, Vol. 37, No. 3,
            pp. 242-247
        """
        last_sum = last_mean * last_sample_count
        new_sum = self._safe_accumulator_op(np.nansum, X, axis=0)

        new_sample_count = np.sum(~np.isnan(X), axis=0)
        updated_sample_count = last_sample_count + new_sample_count

        updated_mean = (last_sum + new_sum) / updated_sample_count

        if last_variance is None:
            updated_variance = None
        else:
            new_unnormalized_variance = (
                self._safe_accumulator_op(np.nanvar, X, axis=0) * new_sample_count)
            last_unnormalized_variance = last_variance * last_sample_count

            with np.errstate(divide='ignore', invalid='ignore'):
                last_over_new_count = last_sample_count / new_sample_count
                updated_unnormalized_variance = (
                    last_unnormalized_variance + new_unnormalized_variance
                    + last_over_new_count / updated_sample_count
                    * (last_sum / last_over_new_count - new_sum) ** 2)

            zeros = last_sample_count == 0
            updated_unnormalized_variance[zeros] = new_unnormalized_variance[zeros]
            updated_variance = updated_unnormalized_variance / updated_sample_count

        return updated_mean, updated_variance, updated_sample_count

    def ipca(self, num_frame, components, singular_vals, mean, var, obs, process_frame=False):
        """
        IncrementalPCA sklearn method

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
        """
        X = obs

        if process_frame:
            X = obs.reshape(1, obs.size)

        n_components = components.shape[0]
        if num_frame == 0:
            mean = 0
            var = 0

        col_mean, col_var, n_total_samples = \
            self._incremental_mean_and_var(
                X, last_mean=mean, last_variance=var,
                last_sample_count=np.repeat(num_frame, X.shape[1]))
        n_total_samples = n_total_samples[0]

        if num_frame == 0:
            X -= col_mean

        else:
            col_batch_mean = np.mean(X, axis=0)
            X -= col_batch_mean
            mean_correction = \
                np.sqrt((num_frame * X.shape[0])
                    / n_total_samples) * (mean - col_batch_mean)
            X = np.vstack((singular_vals.reshape((-1, 1))
                        * components, X, mean_correction))

        # U, S, V = np.linalg.svd(X, full_matrices=False)
        if min(X.shape) < 10:
            U, S, V = self.randomized_svd(X, n_components=10)
        else:
            U, S, V = fbpca.pca(X, k=10)

        U, V = self.svd_flip(U, V)

        return U[:, :n_components], V[:n_components], S[:n_components], col_mean, col_var

    def init_nnmf(X, n_components):
        """
        Intialize loading and score martix for non-negative matrix
        factorization for use in future optimization to obtain solutions
        for loading and score matrix;

        Parameters
        ----------
        data : numpy.array
            Data matrix

        n_components : Int
            number of components 

        Returns
        -------
        A : numpy.array
            Score matrix

        H : numpy.array
            Loading matrix
        """
        nrow, ncol = X.shape

        avg = np.sqart(X.mean() / n_components)
        rng = np.random.mtrand._rand

        A = avg * rng.randn(nrow, n_components)
        H = avg * rng.randn(n_components, ncol)

        np.abs(H, H)
        np.abs(A, A)

        return A, H


    def cost(X, A, H):
        """
        Compute the l2 norm error of NNMF approximation

        Parameters
        ----------
        X : numpy.array
            Data matrix

        A : numpy.array
            Score matrix

        H : numpy.array
            Loading matrix
        
        Returns
        -------
        norm_error : float
            L2 norm error
        """
        AH = A.dot(H)
        error = X - AH
        norm_error = np.linalg.norm(error, 2)

        return norm_error

    def nnls(X, A, H, max_iter):
        """
        Compute non-negative least squares problem

        Parameters
        ----------
        X : numpy.array
            Data matrix

        A : numpy.array
            Score matrix

        H : numpy.array
            Loading matrix

        subject : Boolean
            If True, the subject of optimization is A (over rows of A).
            Otherwise, the subject of optimization is H (over columns of H).

        Returns
        -------
        A : numpy.array
            Updated score matrix

        H : numpy.array
            Updated loading matrix
        """
        n_col = H.shape[1]
        n_row = A.shape[0]

        # optimize A and H in iterative manner
        cost = 0
        tol = 1e-4 

        for i in range(max_iter):
            if i % 2 == 0:
                for j in range(n_col):
                    H[:, j] = nnls(A, X[:, j])[0]
            else:
                for k in range(n_row):
                    A[k, :] = nnls(H.T, X[j, :])[0]

            new_cost = cost(X, A, H)
            if np.abs(cost - new_cost) < tol:
                break
            cost = new_cost

        return A, H


    def nnmf(X, n_components, max_iter=200):
        """
        Perform Non-negative matrix factorization on the given data.
        This is equivalent to computing two non-negative matrices A, H
        whose product approximates the (non-negative) matrix X.

        Parameters
        ----------
        X : numpy.array
            Data matrix

        Returns
        -------
        A : numpy.array
            Score matrix solution to the non-negative matrix factorization

        H : numpy.array
            Loading matrix solution tot he non-negative matrix factorization
        """
        # Confirm that the data is nonnegative
        X_min = X.data.min()
        if X_min < 0:
            raise ValueError('The data must be have non-negative entries')

        # Intialize score and loading matrices for NNMF
        A, H = init_nnmf(X, n_components)

        # Optimize over the score matrix with fixed loading and
        # optimize over the loading matrix with fixed score
        A, H = nnls(X, A, H, max_iter)

        return A, H

    def process_frame(self, frame):
        """
        Perform incremental pca on frames
        """
        num_frame = self.results.num_frame[:]
        components = self.results.components[:]
        singular_vals = self.results.singular_vals[:]
        mean = self.results.mean[:]
        var = self.results.var[:]

        U, V, S, col_mean, col_var = \
            self.ipca(num_frame, components, singular_vals, mean, var, frame, process_frame=True)
        
        self.results.left_singular[:][:U.shape[0], :] = U
        self.results.components[:] = V
        self.results.singular_vals[:] = S
        self.results.num_frame[:] += 1
        self.results.mean[:] = col_mean
        self.results.var[:] = col_var

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
        dest_comp = dest['components'][:]
        src_comp = src['components'][:]
        
        new_comp = np.vstack([dest_comp, src_comp])

        dest['merge_components'][:][:new_comp.shape[0], :] = new_comp

def run_pca(ctx, dataset, n_components=9, roi=None):
    """
    Run PCA with n_component number of components on the given data

    Parameters
    ----------
    ctx
        Context class that contains methods for loading datasets, creating jobs on them
        and running them

    dataset
        Data on which PCA will perform

    Returns
    -------
    PCA solution with n_components number of components
    """
    frame_size = dataset.shape.sig.size
    total_frames = dataset.shape.nav.size
    num_frames = len(list(dataset.get_partitions())) * n_components
    udf = PcaUDF(frame_size=frame_size, total_frames=total_frames, n_components=n_components, num_frames=num_frames)

    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
