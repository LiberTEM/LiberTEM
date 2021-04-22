from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import libertem.udf.sumsigudf as sumsigudf
from .helper import GeneratorHelper


class SumSigTemplate(GeneratorHelper):

    short_name = "sumsig"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.analysis import SumSigAnalysis"]

    def get_docs(self):
        docs = ["# SumSig Analysis"]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"sumsig_analysis = SumSigAnalysis(dataset=ds, parameters={self.params})",
                         "sumsig_result = ctx.run(sumsig_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(sumsig_result.intensity.raw_data)",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class SumSigAnalysis(BaseAnalysis, id_="SUM_SIG"):
    TYPE = "UDF"

    def get_udf(self):
        return sumsigudf.SumSigUDF()

    def get_udf_results(self, udf_results, roi, damage):
        from libertem.viz import visualize_simple
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'],
                           visualized=visualize_simple(
                               udf_results['intensity'].data.reshape(self.dataset.shape.nav),
                               damage=damage
                           ),
                           key="intensity", title="intensity",
                           desc="result from frame integration"),
        ])

    @classmethod
    def get_template_helper(cls):
        return SumSigTemplate
