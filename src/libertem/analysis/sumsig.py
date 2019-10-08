from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import libertem.udf.sumsigudf as sumsigudf


class SumSigAnalysis(BaseAnalysis):
    TYPE = "UDF"

    def get_udf(self):
        return sumsigudf.SumSigUDF()

    def get_udf_results(self, udf_results, roi):
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'],
                           visualized=visualize_simple(
                               udf_results['intensity'].data.reshape(self.dataset.shape.nav)
                           ),
                           key="intensity", title="intensity",
                           desc="result from frame integration"),
        ])
