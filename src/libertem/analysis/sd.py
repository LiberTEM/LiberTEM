import inspect
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
        return [
            "from matplotlib import colors",
            "from libertem.analysis.getroi import get_roi",
            "from libertem.udf.stddev import StdDevUDF"
            ]

    def get_docs(self):
        title = "SD Analysis"
        docs_rst = inspect.getdoc(std.StdDevUDF)
        docs = self.format_docs(title, docs_rst)
        return docs

    def get_analysis(self):
        temp_analysis = [
                    f"roi_params = {self.params['roi']}",
                    "roi = get_roi(roi_params, ds.shape.nav)",
                    "sd_udf  = StdDevUDF()",
                    "sd_result = ctx.run_udf(dataset=ds, udf=sd_udf)",
                    ]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = [
            "plt.figure()",
            "plt.imshow(sd_result['varsum'], norm=colors.LogNorm())",
            "plt.colorbar()",
        ]
        return '\n'.join(plot)

    def get_save(self):
        save = "np.save('sd_result.npy', sd_result['varsum'])"
        return save


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
