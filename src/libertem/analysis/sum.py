import numpy as np

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.udf import UDF
from libertem.analysis.getroi import get_roi


class SumUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(kind='sig', dtype=self.params.dtype)
        }

    def process_tile(self, tile):
        self.results.intensity[:] += np.sum(tile, axis=0)

    def merge(self, dest, src):
        dest['intensity'][:] += src['intensity']


class SumAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        dest_dtype = np.dtype(self.dataset.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        return SumUDF(dtype=dest_dtype)

    def get_roi(self):
        return get_roi(params=self.parameters, shape=self.dataset.shape.nav)

    def get_udf_results(self, udf_results, roi):
        if udf_results['intensity'].data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    udf_results['intensity'].data,
                    key_prefix="intensity",
                    title="intensity",
                    desc="sum of all frames",
                )
            )

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               udf_results['intensity'].data, logarithmic=True
                           ),
                           key="intensity", title="intensity", desc="sum of frames"),
        ])
