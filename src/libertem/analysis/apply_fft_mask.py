from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import libertem.udf.crystallinity as crystal
from .helper import GeneratorHelper


class FFTMaskTemplate(GeneratorHelper):

    short_name = "fft"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.analysis import ApplyFFTMask"]

    # FIXME write and include documentation
    def get_docs(self):
        docs = ["# FFT Analysis"]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"fft_analysis = ApplyFFTMask(dataset=ds, parameters={self.params})",
                         "fft_result = ctx.run(fft_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(fft_result.intensity)",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class ApplyFFTMask(BaseAnalysis, id_="APPLY_FFT_MASK"):
    TYPE = "UDF"

    def get_udf(self):
        rad_in = self.parameters["rad_in"]
        rad_out = self.parameters["rad_out"]
        real_center = (self.parameters["real_centery"], self.parameters["real_centerx"])
        real_rad = self.parameters["real_rad"]
        return crystal.CrystallinityUDF(
            rad_in=rad_in, rad_out=rad_out,
            real_center=real_center, real_rad=real_rad,
        )

    def get_udf_results(self, udf_results, roi, damage):
        from libertem.viz import visualize_simple
        data = udf_results['intensity'].data
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data,
                visualized=visualize_simple(
                    data.reshape(self.dataset.shape.nav),
                    damage=True
                ),
                key="intensity", title="intensity",
                desc="result from integration over mask in Fourier space"
            ),
        ])

    @classmethod
    def get_template_helper(cls):
        return FFTMaskTemplate
