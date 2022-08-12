import sparse
import numpy as np

from libertem.udf.sum import SumUDF
from libertem.udf.stddev import StdDevUDF
from libertem.common.sparse import NUMPY, SPARSE_COO, SPARSE_GCXS, array_format


class SparseInputSumUDF(SumUDF):
    def process_tile(self, tile):
        format = array_format(tile)
        assert format in SPARSE_COO, SPARSE_GCXS
        return super().process_tile(tile)


class DenseInputSumUDF(SumUDF):
    def get_formats(self):
        # also test returning single item instead of iterable
        return NUMPY

    def process_tile(self, tile):
        format = array_format(tile)
        assert format == NUMPY
        return super().process_tile(tile)


def test_basic_sparse(lt_ctx):
    data = sparse.random((7, 11, 13))
    ds = lt_ctx.load('memory', data=data)
    res = lt_ctx.run_udf(dataset=ds, udf=[SparseInputSumUDF(), DenseInputSumUDF(), StdDevUDF()])
    assert np.allclose(data.sum(axis=0).todense(), res[0]['intensity'].raw_data)
    assert np.allclose(data.sum(axis=0).todense(), res[1]['intensity'].raw_data)
    assert np.allclose(data.std(axis=0).todense(), res[2]['std'].raw_data)
