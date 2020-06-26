from libertem.viz import visualize_simple

from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from .helper import GeneratorHelper
import libertem.udf.FEM as FEM


class FEMTemplate(GeneratorHelper):

    short_name = "fem"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.analysis import FEMAnalysis"]

    def get_docs(self):
        docs = ["# FEM Analysis",
                "***about fem analysis ***"]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"fem_analysis = FEMAnalysis(dataset=ds, parameters={self.params})",
                         "fem_result = ctx.run(fem_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(fem_result.intensity.visualized)"]
        return '\n'.join(plot)


class FEMAnalysis(BaseAnalysis, id_="FEM"):
    TYPE = "UDF"

    def get_udf(self):
        center = (self.parameters["cy"], self.parameters["cx"])
        rad_in = self.parameters["ri"]
        rad_out = self.parameters["ro"]
        return FEM.FEMUDF(center=center, rad_in=rad_in, rad_out=rad_out)

    def get_udf_results(self, udf_results, roi):

        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               udf_results['intensity'].data.reshape(self.dataset.shape.nav)),
                           key="intensity", title="intensity",
                           desc="result from SD calculation over ring"),
        ])

    @classmethod
    def get_template_helper(cls):
        return FEMTemplate
