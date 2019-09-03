from libertem.viz import visualize_simple

from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet

import libertem.udf.FEM as FEM


class FEMAnalysis(BaseAnalysis):
    TYPE = "UDF"

    def get_udf(self):
        center = (self.parameters["cy"], self.parameters["cx"])
        rad_in = self.parameters["ri"]
        rad_out = self.parameters["ro"]
        return FEM.FEMUDF(center=center, rad_in=rad_in, rad_out=rad_out)

    def get_udf_results(self, udf_results):

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               udf_results['intensity'].data.reshape(self.dataset.shape.nav)),
                           key="intensity", title="intensity",
                           desc="result from SD calculation over ring"),
        ])
