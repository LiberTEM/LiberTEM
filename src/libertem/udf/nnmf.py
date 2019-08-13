import numpy as np
from scipy.linalg import svd, qr, norm

from libertem.common.buffers import BufferWrapper
from libertem.udf import UDF


class NmfUDF(UDF):
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

            'score': BufferWrapper(
                kind='single',
                extra_shape=(self.params.total_frames, self.params.n_components),
                dtype='float32'
                ),

            'dictionary': BufferWrapper(
                kind='single',
                extra_shape=(self.params.n_components, self.params.frame_size),
                dtype='float32'
                ),

            'merge_score': BufferWrapper(
                kind='single',
                extra_shape=(self.params.total_frames, self.params.n_components),
                dtype='float32'
                ),

            'merge_dict': BufferWrapper(
                kind='single',
                extra_shape=(self.params.n_components, self.params.frame_size),
                dtype='float32'
                ),

            'merge_num': BufferWrapper(
                kind='single',
                dtype='int32'
                ),
            }

    def init_nnmf(self, X):
        """
        Intialize dictionary and score martix for non-negative matrix
        factorization for use in future optimization to obtain solutions
        for dictionary and score matrix;

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
            dictionary matrix
        """
        n_components = self.params.n_components
        nrow, ncol = X.shape

        avg = np.sqrt(X.mean() / n_components)
        rng = np.random.mtrand._rand

        A = avg * rng.randn(nrow, n_components)
        H = avg * rng.randn(n_components, ncol)

        np.abs(H, H)
        np.abs(A, A)

        return A, H

    def cost(self, X, A, H):
        """
        Compute the l2 norm error of NNMF approximation

        Parameters
        ----------
        X : numpy.array
            Data matrix

        A : numpy.array
            Score matrix

        H : numpy.array
            dictionary matrix
        
        Returns
        -------
        norm_error : float
            L2 norm error
        """
        AH = A.dot(H)
        error = X - AH
        norm_error = np.linalg.norm(error, 2)

        return norm_error

    def iter_solver(self, A, W, H, cur_iter):
        """
        Compute the optimal solution at a iteration.
        Based on multiplicative updating by Lee and Seung,
        Algorithms for non-negative matrix factorization

        Parameters
        ----------
        A : numpy.array
            Data matrix

        W : numpy.array
            Score matrix

        H : numpy.array
            dictionary matrix

        cur_iter : int
            current iteration

        Returns
        -------
        W : numpy.array
            Updated score matrix

        H : numpy.array
            Updated dictionary matrix
        """
        print("H: ", H.shape, "W: ", W.shape, "A: ", A.shape)
        H = H.T

        eps = 1e-16
        AtW = A.T.dot(W)
        HWtW = H.dot(W.T.dot(W)) + eps

        H = H * AtW
        H = H / HWtW

        AH = A.dot(H)
        WHtH = W.dot(H.T.dot(H)) + eps
        W = W * AH
        W = W / WHtH

        # eps = 1e-16
        # XtA = X.T.dot(A)
        # AtA = A.T.dot(A)

        # k = self.params.n_components

        # for i in iter(range(0, k)):
        #     print((H.T.dot(AtA[:, i])).shape)
        #     print(A[i, :].shape)
        #     temp = A[:, i] + XtA[:, i] - H.dot(AtA[:, i])
        #     H[:, i] = np.maximum(temp, 1e-16)

        # XH = X.dot(H)
        # HtH = H.T.dot(H)

        # for j in iter(range(0, k)):
        #     temp = W[:, j] * HtH[j, j] + XH[:, j] - A.dot(HtH[:, j])
        #     A[:, j] = np.maximum(temp, 1e-16)

        #     norm = norm(A[:, j])

        #     if norm > 0:
        #         A[:, j] = A[:, j] / norm

        return A, H.T

    def nnls(self, X, A, H, max_iter):
        """
        Compute non-negative least squares problem

        Parameters
        ----------
        X : numpy.array
            Data matrix

        A : numpy.array
            Score matrix

        H : numpy.array
            dictionary matrix

        subject : Boolean
            If True, the subject of optimization is A (over rows of A).
            Otherwise, the subject of optimization is H (over columns of H).

        Returns
        -------
        A : numpy.array
            Updated score matrix

        H : numpy.array
            Updated dictionary matrix
        """
        n_col = H.shape[1]
        n_row = A.shape[0]

        for i in range(max_iter):
            A, H = self.iter_solver(X, A, H, i)

        return A, H

    # def nnls(self, X, A, H, max_iter):
    #     """
    #     Compute non-negative least squares problem

    #     Parameters
    #     ----------
    #     X : numpy.array
    #         Data matrix

    #     A : numpy.array
    #         Score matrix

    #     H : numpy.array
    #         dictionary matrix

    #     subject : Boolean
    #         If True, the subject of optimization is A (over rows of A).
    #         Otherwise, the subject of optimization is H (over columns of H).

    #     Returns
    #     -------
    #     A : numpy.array
    #         Updated score matrix

    #     H : numpy.array
    #         Updated dictionary matrix
    #     """
    #     n_col = H.shape[1]
    #     n_row = A.shape[0]

    #     # optimize A and H in iterative manner
    #     cost = 0
    #     tol = 1e-4 

    #     for i in range(max_iter):

    #         if i % 2 == 0:
    #             for j in range(n_col):
    #                 H[:, j] = self.nnls(A, X[:, j])[0]
    #         else:
    #             for k in range(n_row):
    #                 A[k, :] = self.nnls(H.T, X[j, :])[0]

    #         new_cost = cost(X, A, H)
    #         if np.abs(cost - new_cost) < tol:
    #             break
    #         cost = new_cost

    #     return A, H

    def nnmf(self, X, n_components, max_iter=200):
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
            dictionary matrix solution tot he non-negative matrix factorization
        """
        # Confirm that the data is nonnegative
        X_min = np.amin(X)

        if X_min < 0:
            raise ValueError('The data must be have non-negative entries')

        # Intialize score and dictionary matrices for NNMF
        A, H = self.init_nnmf(X)

        # Optimize over the score matrix with fixed dictionary and
        # optimize over the dictionary matrix with fixed score
        A, H = self.nnls(X, A, H, max_iter)

        return A, H

    def process_partition(self, partition):
        """
        Perform incremental PCA on partitions
        """
        num_frame, sig_row, sig_col = partition.shape
        obs = partition.reshape((num_frame, sig_row * sig_col))
        n_components = self.params.n_components

        A, H = self.nnmf(obs, n_components)

        prev_num = self.results.num_frames[:]

        self.results.score[:][prev_num:prev_num+num_frame, :] = A
        self.results.dictionary[:] = H
        self.num_frames[:] += num_frame

    def modify_dictionary(self, score, dictionary):
        """
        Modify loaidng matrix so that it is bounded by
        a hypercube box

        Parameters
        ----------
        score
            score matrix

        dictionary
            dictionary matrix

        Returns
        -------
        src_data
            Updated data matrix
        """
        n_sample, n_component = score.shape

        minimum = np.amin(score, axis=0)
        maximum = np.amax(score, axis=0)

        idx = []

        for i in range(n_sample):
            check = np.where((score[i, :] == minimum) | (score[i, :] == maximum), True, False)
            if True in check:
                idx.append(i)

        src_score = score[np.array(idx)]
        src_data = src_score @ dictionary

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
        score = src['score'][:]
        dictionary = src['dictionary'][:]

        src_data = self.modify_dictionary(score, dictionary).T

        dictionary, score = self.nnmf(src_data)

        score = score.T
        dictionary = dictionary.T

        merge_num = dest['merge_num'][:]

        dest['merge_score'][:][merge_num:merge_num+score.shape[0], :] = score
        dest['merge_dict'][:][:,merge_num:merge_num+dictionary.shape[1]] = dictionary
        dest['merge_num'][:] += src_data.shape[0]


def run_nnmf(ctx, dataset, n_components=100, roi=None):
    """
    Run PCA with n_component number of components on the given data

    Parameters
    ----------
    ctx
        Context class that contains methods for dictionary datasets, creating jobs on them
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

    udf = NmfUDF(
                frame_size=frame_size,
                total_frames=total_frames,
                n_components=n_components,
                num_frames=num_frames
                )

    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
