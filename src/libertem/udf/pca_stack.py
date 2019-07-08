import collections

import numpy as np
import fbpca

from libertem.common.buffers import BufferWrapper
from libertem.udf import UDF


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
            'stack_data': BufferWrapper(
                kind='single',
                extra_shape=(int(self.params.total_frames/self.params.num_partitions), self.params.frame_size),
                dtype='float32'
                ),

            'num_frame': BufferWrapper(
                kind='single',
                dtype='int32'
                ),

            'num_merge': BufferWrapper(
                kind='single',
                dtype='int32'
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

    def ipca(self, num_frame, components, singular_vals, mean, var, obs, n_components, process_frame=False):
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
        if min(X.shape) < n_components:
            U, S, V = self.randomized_svd(X, n_components=n_components)
        else:
            U, S, V = fbpca.pca(X, k=n_components)

        U, V = self.svd_flip(U, V)

        return U[:, :n_components], V[:n_components], S[:n_components], col_mean, col_var

    def process_frame(self, frame):
        """
        Stack data for each partition to pass onto merging
        """
        num_frame = self.results.num_frame[:]

        if num_frame == 0:
            self.results.stack_data[:][0, :] = frame.reshape(1, frame.size)

        else:
            prev = self.results.stack_data[:]
            prev = prev[~np.all(prev==0, axis=1)]
            new_stack = np.vstack([prev, frame.reshape(1, frame.size)])
            self.results.stack_data[:][:new_stack.shape[0], :] = new_stack

        self.results.num_frame[:] += 1

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
        new_data = src['stack_data'][:]
        num_frame = dest['num_merge'][:]
        mean = dest['mean'][:]
        var = dest['var'][:]
        components = dest['components'][:]
        singular_vals = dest['singular_vals'][:]

        U, V, S, col_mean, col_var = \
            self.ipca(num_frame, components, singular_vals, mean, var, new_data, self.params.n_components)
        
        dest['left_singular'][:][:U.shape[0], :] = U
        dest['components'][:] = V
        dest['singular_vals'][:] = S
        dest['num_merge'][:] += new_data.shape[0]
        dest['mean'][:] = col_mean
        dest['var'][:] = col_var


def run_pca(ctx, dataset, n_components=10, roi=None):
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
    num_partitions = len(list(dataset.get_partitions()))
    udf = PcaUDF(frame_size=frame_size, total_frames=total_frames, n_components=n_components, num_partitions=num_partitions)

    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
