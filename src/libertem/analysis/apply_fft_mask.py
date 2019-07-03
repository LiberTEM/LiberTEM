from libertem.viz import visualize_simple

from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet

import libertem.udf.crystallinity as crystal
crystal.run_analysis_crystall


class ApplyFFTMask(BaseAnalysis):
    TYPE = "UDF"

    def get_udf(self):
        rad_in = self.parameters["rad_in"]
        rad_out = self.parameters["rad_out"]
        real_center = (self.parameters["real_centery"], self.parameters["real_centerx"])
        real_rad = self.parameters["real_rad"]
        return crystal.CrystallinityUDF(rad_in=rad_in, rad_out=rad_out, real_center=real_center,
         real_rad=real_rad)

    def get_udf_results(self, udf_results):

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results.intensity,
                           visualized=visualize_simple(
                               udf_results.intensity.reshape(self.dataset.shape.nav)),
                           key="intensity", title="intensity",
                           desc="result from integration over mask in Fourier space"),
        ])
