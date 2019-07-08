import numpy as np 
import scipy.optimize.nnls as nnls
from sklearn.decomposition import NMF
import fbpca

from utils import MemoryDataSet, _mk_random
from libertem import api
from libertem.udf.pca_nnmf import run_pca

def eig_error(eig_approx, eig):
	"""
	Compute the correlation between the approximated eigenvalues
	and the true eigenvalues (computed using PCA on full-batch data)
	through inner product.

	Parameters
	----------
	eig_approx: numpy.array
		Approximation of eigenvalues

	eig: numpy.array
		True eigenvalues as computed by PCA on full-batch data
	"""
	corr = np.linalg.norm(eig_approx-eig)
	corr = np.inner(eig_approx, eig)
	return corr


def subspace_error(U_approx, U):
	"""
	Compute the frobenius distance between the approximated left
	singular matrix and the exact left singular matrix (computed
	using PCA on full-batch data)

	Parameters
	-----------
	U_approx: numpy.array
		Approximation of eigenspace matrix

	U: numpy.array
		True eigenspace matrix as computed by PCA on full-batch data

	Returns
	-------
	err: float32
		Approximation error defined by the frobenius distance norm
		between the approximation and the true error
	"""
	n_components = U.shape[1]
	A = U_approx.T.dot(U)
	B = U_approx.dot(U_approx.T)

	err = np.sqrt(n_components+np.trace(B.dot(B)) - 2 * np.trace(A.dot(A.T)))

	frob = np.linalg.norm(U-U_approx, ord='fro')

	return err/np.sqrt(n_components )


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

	avg = np.sqrt(X.mean() / n_components)
	rng = np.random.mtrand._rand

	A = avg * rng.randn(nrow, n_components)
	H = avg * rng.randn(n_components, ncol)

	np.abs(H, H)
	np.abs(A, A)

	return A, H


def cost_func(X, A, H):
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

def nnlss(X, A, H, max_iter=200):
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

		new_cost = cost_func(X, A, H)
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
	X_min = X.min()
	if X_min < 0:
		raise ValueError('The data must be have non-negative entries')

	# Intialize score and loading matrices for NNMF
	A, H = init_nnmf(X, n_components)

	# Optimize over the score matrix with fixed loading and
	# optimize over the loading matrix with fixed score
	A, H = nnlss(X, A, H, max_iter)

	return A, H

def diffsnorm(A, U, s, Va, n_iter=20):
	"""
	2-norm accuracy of an approx to a matrix.
	Computes an estimate snorm of the spectral norm (the operator norm
	induced by the Euclidean vector norm) of A - U diag(s) Va, using
	n_iter iterations of the power method started with a random vector;
	n_iter must be a positive integer.
	Increasing n_iter improves the accuracy of the estimate snorm of
	the spectral norm of A - U diag(s) Va.
	Notes
	-----
	To obtain repeatable results, reset the seed for the pseudorandom
	number generator.
	Parameters
	----------
	A : array_like
		first matrix in A - U diag(s) Va whose spectral norm is being
		estimated
	U : array_like
		second matrix in A - U diag(s) Va whose spectral norm is being
		estimated
	s : array_like
		vector in A - U diag(s) Va whose spectral norm is being
		estimated
	Va : array_like
		fourth matrix in A - U diag(s) Va whose spectral norm is being
		estimated
	n_iter : int, optional
		number of iterations of the power method to conduct;
		n_iter must be a positive integer, and defaults to 20
	Returns
	-------
	float
		an estimate of the spectral norm of A - U diag(s) Va (the
		estimate fails to be accurate with exponentially low prob. as
		n_iter increases; see references DM1_, DM2_, and DM3_ below)
	Examples
	--------
	>>> from fbpca import diffsnorm, pca
	>>> from numpy.random import uniform
	>>> from scipy.linalg import svd
	>>>
	>>> A = uniform(low=-1.0, high=1.0, size=(100, 2))
	>>> A = A.dot(uniform(low=-1.0, high=1.0, size=(2, 100)))
	>>> (U, s, Va) = svd(A, full_matrices=False)
	>>> A = A / s[0]
	>>>
	>>> (U, s, Va) = pca(A, 2, True)
	>>> err = diffsnorm(A, U, s, Va)
	>>> print(err)
	This example produces a rank-2 approximation U diag(s) Va to A such
	that the columns of U are orthonormal, as are the rows of Va, and
	the entries of s are all nonnegative and are nonincreasing.
	diffsnorm(A, U, s, Va) outputs an estimate of the spectral norm of
	A - U diag(s) Va, which should be close to the machine precision.
	References
	----------
	.. [DM1] Jacek Kuczynski and Henryk Wozniakowski, Estimating the
			 largest eigenvalues by the power and Lanczos methods with
			 a random start, SIAM Journal on Matrix Analysis and
			 Applications, 13 (4): 1094-1122, 1992.
	.. [DM2] Edo Liberty, Franco Woolfe, Per-Gunnar Martinsson,
			 Vladimir Rokhlin, and Mark Tygert, Randomized algorithms
			 for the low-rank approximation of matrices, Proceedings of
			 the National Academy of Sciences (USA), 104 (51):
			 20167-20172, 2007. (See the appendix.)
	.. [DM3] Franco Woolfe, Edo Liberty, Vladimir Rokhlin, and Mark
			 Tygert, A fast randomized algorithm for the approximation
			 of matrices, Applied and Computational Harmonic Analysis,
			 25 (3): 335-366, 2008. (See Section 3.4.)
	See also
	--------
	diffsnormc, pca
	"""

	(m, n) = A.shape
	(m2, k) = U.shape
	k2 = len(s)
	l = len(s)
	(l2, n2) = Va.shape

	assert m == m2
	assert k == k2
	assert l == l2
	assert n == n2

	assert n_iter >= 1

	if np.isrealobj(A) and np.isrealobj(U) and np.isrealobj(s) and \
			np.isrealobj(Va):
		isreal = True
	else:
		isreal = False

	# Promote the types of integer data to float data.
	dtype = (A * 1.0).dtype

	if m >= n:

		if isreal:
			x = np.random.normal(size=(n, 1)).astype(dtype)
		else:
			x = np.random.normal(size=(n, 1)).astype(dtype) \
				+ 1j * np.random.normal(size=(n, 1)).astype(dtype)

		x = x / norm(x)

		for it in range(n_iter):

			y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))
			x = mult(y.conj().T, A).conj().T \
				- Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

			snorm = norm(x)
			if snorm == 0:
				return 0

			x = x / snorm

		snorm = math.sqrt(snorm)

	if m < n:

		if isreal:
			y = np.random.normal(size=(m, 1)).astype(dtype)
		else:
			y = np.random.normal(size=(m, 1)).astype(dtype) \
				+ 1j * np.random.normal(size=(m, 1)).astype(dtype)

		y = y / norm(y)

		for it in range(n_iter):

			x = mult(y.conj().T, A).conj().T \
				- Va.conj().T.dot(np.diag(s).conj().T.dot(U.conj().T.dot(y)))

			y = mult(A, x) - U.dot(np.diag(s).dot(Va.dot(x)))

			snorm = norm(y)

			if snorm == 0:
				return 0

			y = y / snorm

		snorm = math.sqrt(snorm)

	return snorm

def test_pca(lt_ctx):
	"""
	Test Principal Component Analysis

	Parameters
	----------
	lt_ctx
		Context class for loading dataset and creating jobs on them
	"""
	# path = '/home/jae/Downloads/saved_data.raw'
	# data = np.random.rand(256, 256, 128, 128)
	# # data.tofile(path)
	# # data = _mk_random(size=(256, 256, 128, 128), dtype="float32")
	# dataset = np.memmap(path, dtype=np.float32, mode="w+", shape=(256, 256, 128, 128))
	# # dataset[:] = data[:]

	# for i in range(256):
	# 	for j in range(256):
	# 		dataset[i, j, :, :] = data[i, j, :, :]

	# del dataset

	# with api.Context() as ctx:
	# 		path = filename
	# 		dataset = ctx.load(
	# 		'raw',
	# 		path=path,
	# 		scan_size=(256, 256),
	# 		)


	# dataset = np.memmap(path, dtype=np.float32, shape=(256, 256, 128, 128))
	# newfp = np.memmap(path, dtype="float32", mode='r', shape=(256, 256, 128, 128))
	# data.tofile(path)


	# data = _mk_random(size=(64, 64, 64, 64), dtype="float32")
	# dataset = MemoryDataSet(data=data, tileshape=(1, 32, 32),
	# 						num_partitions=4, sig_dims=2)

	# dataset = ctx.load(
	#	 "raw",
	#	 path=filename,
	#	 dtype="float32",
	#	 scan_size=(256, 256),
	#	 detector_size_raw=(128, 128),
	#	 crop_detector_to=(128, 128),
	# )
	# path = '/home/jae/Downloads/data256.raw'
	# dataset = np.memmap(path, dtype=np.float32, mode="w+", shape=(256, 256, 128, 128))

	with api.Context() as ctx:
		path = '/home/jae/Downloads/scan_11_x256_y256.raw'
		dataset = ctx.load(
		'empad',
		path=path,
		scan_size=(256, 256),
		)
	res = run_pca(lt_ctx, dataset, n_components=10)

	assert 'merge_components' in res
	
	components = res['merge_components'].data
	U, S, V = fbpca.pca(components, k=10)
	# components = np.absolute(components)

	# model = NMF(n_components=9, init='random', random_state=0)

	# W = model.fit_transform(components)
	# H = model.components_
	# A, H = NMF(components, 9)

	# assert W.shape == (components.shape[0], 9)
	# assert H.shape == (9, components.shape[1])
	# flattened = data.reshape((1024, 1024))
	# U, D, V = np.linalg.svd(flattened)

	# # normalize the singular values for comparison
	# D = D / np.linalg.norm(D)
	# eig_approx = res['singular_vals'].data
	# eig_approx = eig_approx / np.linalg.norm(eig_approx)

	# tol = 1

	# assert eig_error(eig_approx, D[:9]) < tol
	# assert subspace_error(res['components'].data, V[:9]) < tol

