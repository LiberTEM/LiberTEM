import numba
import numpy as np
import scipy.sparse


'''
Custom methods added to scipy.sparse.csc_matrix using numba

This comes in handy when implementing customized math kernels around sparse matrices.
CSC matrices are usually superior to COO matrices for dot products and similar processing,
which means the COO matrix of sparse.pydata.org is not a suitable basis for such math kernels.
For that reason we use scipy.sparse.csc.csc_matrix as a basis.

Note that the numerical "work horses" are implemented as pure functions because @numba.njit
doesn't support the relevant object orientation features of Python. The methods of the class are
wrappers for the work horses. In the future numba might support @numba.njit for methods directly,
making this workaround unnecessary. See also Issue #350.
'''


@numba.njit
def _dot(data, indices, indptr, other, out):
    ks = other.shape[1]
    for col in range(len(indptr) - 1):
        for index in range(indptr[col], indptr[col+1]):
            row = indices[index]
            d = data[index]
            for k in range(ks):
                out[row, k] += other[col, k] * d
    return out


@numba.njit
def _binned_std(data, indices, indptr, other, avg_out, std_out):
    '''
    We assume that the bins are already normalized to save the normalization step on each run
    '''
    _dot(data=data, indices=indices, indptr=indptr, other=other, out=avg_out)

    ks = other.shape[1]
    for col in range(len(indptr[:-1])):
        colptr = indptr[col]
        for index in range(colptr, indptr[col+1]):
            row = indices[index]
            d = data[index]
            for k in range(ks):
                diff = other[col, k] - avg_out[row, k]
                std_out[row, k] += diff * np.conj(diff) * d
    np.sqrt(std_out, out=std_out)
    return (avg_out, std_out)


class CustomCSC(scipy.sparse.csc.csc_matrix):
    def custom_dot(self, other, out=None):
        '''
        Example implementation of the dot product

        Example for how to implement custom functions.
        Should be equivalent to the dot() method of the csc matrix.
        '''
        if out is None:
            out = np.zeros((self.shape[0], other.shape[1]), dtype=np.result_type(self.dtype, other))
        return _dot(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            other=other,
            out=out,
        )

    def binned_std(self, other, avg_out=None, std_out=None):
        '''
        Calculate binned average and standard deviation.

        This assumes that the bins in self are normalized to avoid
        the overhead of normalization on each execution.
        '''
        if avg_out is None:
            avg_out = np.zeros(
                (self.shape[0], other.shape[1]),
                dtype=np.result_type(self.dtype, other)
            )
        if std_out is None:
            std_out = np.zeros(
                (self.shape[0], other.shape[1]),
                dtype=np.result_type(self.dtype, other)
            )
        return _binned_std(
            data=self.data,
            indices=self.indices,
            indptr=self.indptr,
            other=other,
            avg_out=avg_out,
            std_out=std_out,
        )
