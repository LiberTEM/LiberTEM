import numpy as np
from scipy.linalg import svd, qr

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
            'num_frame': BufferWrapper(
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

            'merge_num': BufferWrapper(
                kind='single',
                dtype='int32'),
            }

    def svd_flip(self, U, V):
        """
        Sign correction to ensure deterministic output from SVD.

        Parameters
        ----------
        U
            Left singular matrix

        V
            Right singular matrix

        Returns
        -------
        U_adjusted
            sign adjusted left singular matrix

        V_adjusted
            sign adjusted right singular matrix
        """
        max_abs_rows = np.argmax(np.abs(V), axis=1)
        signs = np.sign(V[range(V.shape[0]), max_abs_rows])

        U *= signs
        V *= signs[:, np.newaxis]

        return U, V

    def randomized_svd(self, X, n_components):
        """
        Perform randomized SVD on the given matrix
        """
        row, col = X.shape

        rand_matrix = np.random.normal(size=(col, n_components))
        Q, _ = qr(X @ rand_matrix, mode='reduced')

        smaller_matrix = Q.T @ X
        U_hat, S, V = svd(smaller_matrix, full_matrices=False)
        U = Q @ U_hat

        return U, S, V

    def safe_accumulator_op(self, op, x, *args, **kwargs):
        """
        This function provides numpy accumulator functions with a float64 dtype
        when used on a floating point input. This prevents accumulator overflow on
        smaller floating point dtypes.

        Parameters
        ----------
        op
            A numpy accumulator function such as np.mean or np.sum

        x
            A numpy array to apply the accumulator function

        *args
            Positional arguments passed to the accumulator function after the
            input x

        **kwargs
            Keyword arguments passed to the accumulator function

        Returns
        -------
        result
            The output of the accumulator function passed to this function
        """

        if np.issubdtype(x.dtype, np.floating) and x.dtype.itemsize < 8:
            result = op(x, *args, **kwargs, dtype=np.float64)

        else:
            result = op(x, *args, **kwargs)

        return result

    def incremental_mean_and_var(self, X, last_mean, last_variance, last_sample_count):
        """
        Calculate mean update and a Youngs and Cramer variance update.

        Parameters
        ----------
        X
            ndarray with shape (n_samples, n_features)

        last_mean
            ndarray with shape (n_features,)

        last_variance
            ndarray with shape (n_features,)

        last_sample_count
            ndarray with shape (n_features,)

        Returns
        -------
        updated_mean
            ndarray with shape (n_features,)

        updated_variance
            ndarray with shape (n_features,)

        updated_sample_count
            ndarray with shape (n_features,)

        References
        ----------
        T. Chan, G. Golub, R. LeVeque. Algorithms for computing the sample
            variance: recommendations, The American Statistician, Vol. 37, No. 3,
            pp. 242-247
        """
        last_sum = last_mean * last_sample_count
        new_sum = self.safe_accumulator_op(np.nansum, X, axis=0)

        new_sample_count = np.sum(~np.isnan(X), axis=0)
        updated_sample_count = last_sample_count + new_sample_count

        updated_mean = (last_sum + new_sum) / updated_sample_count

        if last_variance is None:
            updated_variance = None

        else:
            new_unnormalized_variance = (
                self.safe_accumulator_op(np.nanvar, X, axis=0) * new_sample_count)
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

    def ipca(self, num_frame, components, singular_vals, mean, var, obs):
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
        n_components = self.params.n_components

        X = obs.copy()

        if num_frame == 0:
            mean = 0
            var = 0

        col_mean, col_var, n_total_samples = \
            self.incremental_mean_and_var(
                obs, last_mean=mean, last_variance=var,
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

        U, S, V = svd(X, full_matrices=False)

        U, V = self.svd_flip(U, V)

        return U[:, :n_components], V[:n_components], S[:n_components], col_mean, col_var

    def process_partition(self, partition):
        """
        Perform incremental PCA on partitions
        """
        n_components = self.params.n_components

        num_frame, sig_row, sig_col = partition.shape
        obs = partition.reshape((num_frame, sig_row * sig_col))

        U, S, V = svd(obs, full_matrices=False)

        self.results.left_singular[:][:U.shape[0], :] = U[:, :n_components]
        self.results.components[:] = V[:n_components]
        self.results.singular_vals[:] = S[:n_components]
        self.results.num_frame[:] += num_frame

    def modify_loading(self, loading, component):
        """
        Modify loaidng matrix so that it is bounded by
        a hypercube box

        Parameters
        ----------
        loading
            loading matrix of the PCA

        component
            component matrix of the PCA

        Returns
        -------
        src_data
            Updated data matrix
        """
        n_sample, n_component = loading.shape

        minimum = np.amin(loading, axis=0)
        maximum = np.amax(loading, axis=0)

        idx = []

        for i in range(n_sample):
            check = np.where((loading[i, :] == minimum) | (loading[i, :] == maximum), True, False)
            if True in check:
                idx.append(i)

        src_loading = loading[np.array(idx)]
        src_data = src_loading @ component

        return src_data

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
        U = src['left_singular'][:]
        S = src['singular_vals'][:]
        V = src['components'][:]
        src_num_frame = src['num_frame'][:][0]
        col_mean = src['mean'][:]
        col_var = src['var'][:]
        U = U[:src_num_frame, :]

        src_data = self.modify_loading(U*S, V)

        merge_num = dest['merge_num'][:][0]

        if merge_num == 0:
            U, V, S, col_mean, col_var =\
                self.ipca(merge_num, V, S, col_mean, col_var, src_data)

        else:
            components = dest['components'][:]
            singular_vals = dest['singular_vals'][:]
            mean = dest['mean'][:]
            var = dest['var'][:]

            U, V, S, col_mean, col_var =\
                self.ipca(merge_num, components, singular_vals, mean, var, src_data)

        dest['left_singular'][:][:U.shape[0], :] = U
        dest['components'][:] = V
        dest['singular_vals'][:] = S
        dest['merge_num'][:] += src_data.shape[0]
        dest['mean'][:] = col_mean
        dest['var'][:] = col_var

        print(merge_num)


def run_pca(ctx, dataset, n_components=100, roi=None):
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

    udf = PcaUDF(
                frame_size=frame_size,
                total_frames=total_frames,
                n_components=n_components,
                num_frames=num_frames
                )

    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
