import inspect

from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from .helper import GeneratorHelper
import libertem.udf.FEM as FEM


class FEMTemplate(GeneratorHelper):

    short_name = "fem"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.udf.FEM import FEMUDF"]

    def get_docs(self):
        title = "FEM Analysis"
        docs_rst = inspect.getdoc(FEM.FEMUDF)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        cx = self.params['cx']
        cy = self.params['cy']
        ri = self.params['ri']
        ro = self.params['ro']
        params = f"center=({cx},{cy}), rad_in={ri}, rad_out={ro}"
        return params

    def get_analysis(self):
        params = self.convert_params()
        temp_analysis = [
                    f"fem_udf = FEMUDF({params})",
                    "fem_result = ctx.run_udf(dataset=ds, udf=fem_udf)",
        ]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = [
            "plt.figure()",
            "plt.imshow(fem_result['intensity'])",
            "plt.colorbar()",
        ]
        return ['\n'.join(plot)]


class FEMAnalysis(BaseAnalysis, id_="FEM"):
    TYPE = "UDF"

    def get_udf(self):
        center = (self.parameters["cy"], self.parameters["cx"])
        rad_in = self.parameters["ri"]
        rad_out = self.parameters["ro"]
        return FEM.FEMUDF(center=center, rad_in=rad_in, rad_out=rad_out)

    def get_udf_results(self, udf_results, roi, damage):
        from libertem.viz import visualize_simple
        return AnalysisResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               udf_results['intensity'].data.reshape(self.dataset.shape.nav),
                               damage=damage
                            ),
                           key="intensity", title="intensity",
                           desc="result from SD calculation over ring"),
        ])

    @classmethod
    def get_template_helper(cls):
        return FEMTemplate
