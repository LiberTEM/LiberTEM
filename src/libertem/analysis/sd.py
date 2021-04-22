import inspect
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
            "from libertem import masks",
            "from libertem.udf.stddev import StdDevUDF"
            ]

    def get_docs(self):
        title = "SD Analysis"
        docs_rst = inspect.getdoc(std.StdDevUDF)
        docs = self.format_docs(title, docs_rst)
        return docs

    def get_analysis(self):
        roi = self.get_roi()
        temp_analysis = [
                    f"{roi}",
                    "sd_udf  = StdDevUDF()",
                    "sd_result = ctx.run_udf(dataset=ds, roi=roi, udf=sd_udf)",
                    ]
        return '\n'.join(temp_analysis)

    # FIXME  `plt.colorbar()` creates error while testing
    # moreinfo  https://github.com/LiberTEM/LiberTEM/pull/801#pullrequestreview-453173083
    def get_plot(self):
        plot = [
            "plt.figure()",
            "plt.imshow(sd_result['varsum'].data, norm=colors.LogNorm())",
        ]
        return ['\n'.join(plot)]

    def get_save(self):
        save = "np.save('sd_result.npy', sd_result['varsum'])"
        return save


class SDAnalysis(BaseAnalysis, id_="SD_FRAMES"):
    TYPE = 'UDF'

    def get_udf(self):
        return std.StdDevUDF()

    def get_roi(self):
        return get_roi(params=self.parameters, shape=self.dataset.shape.nav)

    def get_udf_results(self, udf_results, roi, damage):
        from libertem.viz import visualize_simple
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=udf_results['std'].data,
                visualized=visualize_simple(
                    udf_results['std'].data, logarithmic=True,
                    damage=True
                ),
                key="intensity", title="intensity [log]",
                desc="Standard deviation of frames log-scaled"
            ),
            AnalysisResult(
                raw_data=udf_results['std'],
                visualized=visualize_simple(
                    udf_results['std'].data, logarithmic=False,
                    damage=True
                ),
                key="intensity_lin",
                title="intensity [lin]",
                desc="Standard deviation of frames lin-scaled"
            ),
        ])

    @classmethod
    def get_template_helper(cls):
        return SDTemplate
