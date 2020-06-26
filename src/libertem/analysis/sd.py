from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import libertem.udf.stddev as std
from libertem.analysis.getroi import get_roi
from .helper import GeneratorHelper


class SDTemplate(GeneratorHelper):

    short_name = "sd"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.analysis import SDAnalysis"]

    def get_docs(self):
        docs = ["# SD Analysis",
                "Calculates standard deviation of all detector frames."]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"sd_analysis = SDAnalysis(dataset=ds, parameters={self.params})",
                         "sd_result = ctx.run(sd_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(sd_result.intensity.visualized)"]
        return '\n'.join(plot)


class SDAnalysis(BaseAnalysis, id_="SD_FRAMES"):
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

    @classmethod
    def get_template_helper(cls):
        return SDTemplate
