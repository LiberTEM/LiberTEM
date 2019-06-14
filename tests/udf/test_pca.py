import numpy as np 

from utils import MemoryDataSet, _mk_random
from libertem import api
from libertem.udf.pca import run_pca

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
    A = U_approx.dot(U)
    B = U_approx.T.dot(U_approx)

    err = np.sqrt(n_components+np.trace(B.dot(B)) - 2 * np.trace(A.dot(A.T)))

    frob = np.linalg.norm(U-U_approx.T, ord='fro')

    return err/np.sqrt(n_components )


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

	res = run_pca(lt_ctx, dataset)

	assert 'num_frame' in res
	assert 'singular_vals' in res
	assert 'components' in res
	assert 'left_singular' in res

	N = data.shape[2] * data.shape[3]
	assert res['num_frame'].data == N

	# TODO: Find the appropriate tolerance error bound