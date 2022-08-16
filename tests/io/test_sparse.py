import random

import numpy as np
import pytest

from libertem.udf.sum import SumUDF
from libertem.common.sparse import NUMPY, SPARSE_COO, SPARSE_GCXS, array_format, as_format

from utils import _mk_random, set_backend


# SumUDF supports sparse input and CuPy
# Test that densification for CuPy and sparse input on NumPy works
class SparseInputSumUDF(SumUDF):
    def get_formats(self):
        # Ensure sparse is preferred to make sure we receive
        # it from a sparse dataset
        return (SPARSE_COO, SPARSE_GCXS, NUMPY)

    def process_tile(self, tile):
        format = array_format(tile)
        if self.meta.device_class == 'cpu':
            assert format in (SPARSE_COO, SPARSE_GCXS)
        else:
            import cupy
            assert isinstance(tile, cupy.ndarray)
        return super().process_tile(tile)


# Make sure we actually use CuPy
class CupyInputSumUDF(SparseInputSumUDF):
    def process_tile(self, tile):
        assert self.meta.device_class == 'cuda'
        return super().process_tile(tile)


# Test densification due to UDF format requirements
class DenseInputSumUDF(SumUDF):
    def get_formats(self):
        # also test returning single item instead of iterable
        return self.FORMAT_NUMPY

    def process_tile(self, tile):
        format = array_format(tile)
        assert format == self.FORMAT_NUMPY
        return super().process_tile(tile)


# Test sparsification due to UDF format requirements
# This one causes a warning on CuPy workers
# since it uses CPU processing fallback to ensure sparse input.
class OnlySparseSumUDF(SumUDF):
    def get_formats(self):
        return self.FORMAT_SPARSE_GCXS

    def process_tile(self, tile):
        format = array_format(tile)
        assert format == self.FORMAT_SPARSE_GCXS
        return super().process_tile(tile)


# This one should trigger an exception due to requirements that are
# currently unsupported: Only sparse input, only CuPy
class OnlyCupySparseSumUDF(OnlySparseSumUDF):
    def get_backends(self):
        return ('cupy', )


@pytest.mark.parametrize(
    'format', (NUMPY, SPARSE_COO, SPARSE_GCXS)
)
def test_sparse(lt_ctx, format):
    '''
    Test that constraints are observed and selection logic works
    with different sets of UDFs.
    '''
    data = _mk_random((7, 11, 13), format=format)
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
        ref = as_format(data.sum(axis=0), NUMPY)
        for r in res:
            assert np.allclose(ref, r['intensity'].raw_data)


def test_on_cuda(lt_ctx):
    with set_backend('cupy'):
        data = _mk_random((7, 11, 13), format=SPARSE_COO)
        ds = lt_ctx.load('memory', data=data)
        # Densifies input on CuPy
        res1 = lt_ctx.run_udf(dataset=ds, udf=[CupyInputSumUDF()])
        # Falls back to NumPy backend and complains
        with pytest.warns(RuntimeWarning, match='recommended on CUDA are '):
            res2 = lt_ctx.run_udf(dataset=ds, udf=[OnlySparseSumUDF()])
        with pytest.raises(RuntimeError, match='supported on CUDA are '):
            _ = lt_ctx.run_udf(dataset=ds, udf=[OnlyCupySparseSumUDF()])
        ref = data.sum(axis=0).todense()
        assert np.allclose(ref, res1[0]['intensity'].raw_data)
        assert np.allclose(ref, res2[0]['intensity'].raw_data)
