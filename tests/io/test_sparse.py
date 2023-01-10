import random

import numpy as np
import pytest

from libertem.udf.sum import SumUDF
from sparseconverter import (
    CUPY_SCIPY_CSR, NUMPY, SCIPY_COO, SPARSE_COO, SPARSE_GCXS, get_backend, for_backend
)

from utils import _mk_random, set_device_class


# SumUDF supports sparse input and CuPy
# Test that densification for CuPy and sparse input on NumPy works
class SparseInputSumUDF(SumUDF):
    def get_backends(self):
        return (SPARSE_COO, SPARSE_GCXS, SCIPY_COO, CUPY_SCIPY_CSR)

    def process_tile(self, tile):
        format = get_backend(tile)
        if self.meta.device_class == 'cpu':
            assert format in (SPARSE_COO, SPARSE_GCXS, SCIPY_COO)
        else:
            assert format in (CUPY_SCIPY_CSR, )
        return super().process_tile(tile)


# Make sure we actually use CuPy
class CupyInputSumUDF(SparseInputSumUDF):
    def process_tile(self, tile):
        assert self.meta.device_class == 'cuda'
        return super().process_tile(tile)


# Test densification due to UDF format requirements
class DenseInputSumUDF(SumUDF):
    def get_backends(self):
        return (self.BACKEND_NUMPY, self.BACKEND_CUPY)

    def process_tile(self, tile):
        backend = get_backend(tile)
        assert backend in (self.BACKEND_NUMPY, self.BACKEND_CUPY)
        return super().process_tile(tile)


# Test sparsification due to UDF format requirements
class OnlySparseSumUDF(SumUDF):
    def get_backends(self):
        return self.BACKEND_SPARSE_GCXS

    def process_tile(self, tile):
        backend = get_backend(tile)
        assert backend == self.BACKEND_SPARSE_GCXS
        return super().process_tile(tile)


@pytest.mark.parametrize(
    'format', (NUMPY, SPARSE_COO, SPARSE_GCXS)
)
def test_sparse(lt_ctx, format):
    '''
    Test that constraints are observed and selection logic works
    with different sets of UDFs.
    '''
    data = _mk_random((7, 11, 13), array_backend=format)
    ds = lt_ctx.load('memory', data=data)
    udfs = [OnlySparseSumUDF(), DenseInputSumUDF(), SumUDF(), SumUDF()]
    if format != NUMPY:
        # Make sure sparse input is used if the dataset is sparse
        udfs.append(SparseInputSumUDF())
    for i in range(5):
        subset = random.sample(udfs, k=random.randint(1, len(udfs)))
        random.shuffle(subset)
        res = lt_ctx.run_udf(
            dataset=ds,
            udf=subset
        )
        ref = for_backend(data.sum(axis=0), NUMPY)
        for r in res:
            assert np.allclose(ref, r['intensity'].raw_data)


def test_on_cuda(lt_ctx):
    with set_device_class('cupy'):
        data = _mk_random((7, 11, 13), array_backend=SPARSE_COO)
        ds = lt_ctx.load('memory', data=data)
        # Densifies input on CuPy
        res1 = lt_ctx.run_udf(dataset=ds, udf=[CupyInputSumUDF()])
        res2 = lt_ctx.run_udf(dataset=ds, udf=[OnlySparseSumUDF()])
        ref = data.sum(axis=0).todense()
        assert np.allclose(ref, res1[0]['intensity'].raw_data)
        assert np.allclose(ref, res2[0]['intensity'].raw_data)
