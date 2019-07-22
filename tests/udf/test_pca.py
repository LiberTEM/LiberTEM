import numpy as np 
from scipy.linalg import norm

from utils import MemoryDataSet, _mk_random
from libertem import api
from libertem.udf.pca_data import run_pca


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


def diffsnorm(data, reconstruct):
	"""
	Norm difference between original data
	and reconstructed data from PCA

	Parameters
	----------
	data
		ndarray original data matrix

	reconstruct
		ndarray reconstructed data matrix

	Returns
	-------
	norm
		square root of sum of squares of the norm difference
		between data and reconstructed data
	"""
	return norm(data-reconstruct)/norm(data+reconstruct)


def test_pca(lt_ctx):
	"""
	Test Principal Component Analysis

	Parameters
	----------
	lt_ctx
		Context class for loading dataset and creating jobs on them
	"""
	data = _mk_random(size=(32, 32, 32, 32), dtype="float32")
	dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
							num_partitions=2, sig_dims=2)

	res = run_pca(lt_ctx, dataset, n_components=100)

	assert 'components' in res
	assert 'left_singular' in res
	assert 'singular_vals' in res
	assert 'mean' in res

	left_singular = res['left_singular'].data
	singular_vals = res['singular_vals'].data
	components = res['components'].data
	mean = res['mean'].data

	left_singular = left_singular[~np.all(left_singular==0, axis=1)]
	reconstruct_loading = left_singular @ np.diag(singular_vals)

	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]*data.shape[3]))
	projected_data = data @ components.T
	reconstruct_data = projected_data @ components + mean

	# TODO : construct a reasonable lower bound on the reconstruction error
	assert diffsnorm(data, reconstruct_data) < 1
