from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import libertem.udf.stddev as std
from libertem.analysis.getroi import get_roi


class SDAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        return std.StdDevUDF()

    def get_roi(self):
        return get_roi(params=self.parameters, shape=self.dataset.shape.nav)

    def get_udf_results(self, udf_results, roi):
        udf_results = std.consolidate_result(udf_results)
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['var'].data,
                           visualized=visualize_simple(
                               udf_results['var'].data, logarithmic=True),
                           key="intensity", title="intensity",
                           desc="SD of frames"),
        ])
