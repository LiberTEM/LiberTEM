import numpy as np
import scipy.optimize.nnls as nnls

from libertem.udf.base import UDF


nnmf = collections.namedtuple('NNMF', ['score', 'loading'])


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


# def nnmf_merge(p0, p1):
# 	"""
# 	Merge two Non-negative matrix factorization results
# 	"""
# 	U0, U1 = p0.score, p1.score
# 	V0, V1 = p0.loading, p1.loading

# 	for i in range(iterations):

# class NNMF(UDF):

# 	def get_result_buffers(self):
# 		"""
# 		Intialize BufferWrapper object for NNMF

# 		Returns
# 		-------

# 		"""
# 		return {
# 			"score": self.buffer(
# 				kind="single", extra_shape= , dtype="float32"
# 				),
# 			"loading": self.buffer(
# 				kind="single", extra_shape= , dtype="float32"
# 				)
# 		}

# 	def merge(self, dest, src):
# 		"""
#         Given two sets of partitions, with score and loading matrices,
#         update the merged score and loading matrices
# 		"""
# 		p0 = nnmf(
# 					score=dest["score"][:],
# 					loading=dest["loading"][:]
# 					)

# 		p1 = nnmf(
# 					score=src["score"][:],
# 					loading=src["loading"][:]
# 					)

# 		A, H = nnmf_merge(p0, p1)

# 		dest["score"][:] = A
# 		dest["loading"][:] = H

# 	def process_tile(self, tile):
# 		"""
# 		Given a tile, update parameters related to NNMF

#         Parameters
#         ----------
#         tile
#             single tile of the data
# 		"""

