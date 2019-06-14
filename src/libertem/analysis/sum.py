import numpy as np

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.udf import UDF


class SumUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(kind='sig', dtype=self.params.dtype)
        }

    def process_tile(self, tile, tile_slice):
        self.results.intensity += np.sum(tile, axis=0)

    def merge(self, dest, src):
        dest['intensity'][:] += src['intensity']


class SumAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        dest_dtype = np.dtype(self.dataset.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        return SumUDF(dtype=dest_dtype)

    def get_udf_results(self, udf_results):
        if udf_results.intensity.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    udf_results.intensity,
                    key_prefix="intensity",
                    title="intensity",
                    desc="sum of all frames",
                )
            )

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results.intensity,
                           visualized=visualize_simple(udf_results.intensity, logarithmic=True),
                           key="intensity", title="intensity", desc="sum of all frames"),
        ])
