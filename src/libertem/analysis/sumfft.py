from libertem.analysis.sum import SumAnalysis
from libertem.masks import _make_circular_mask
from .base import AnalysisResult, AnalysisResultSet
from .helper import GeneratorHelper

import numpy as np


class SumfftTemplate(GeneratorHelper):

    short_name = "sumfft"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return [
            "from libertem.analysis import SumfftAnalysis",
            "from matplotlib import colors",
        ]

    # FIXME write and include documentation
    def get_docs(self):
        docs = ["# SUM FFT Analysis"]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"sumfft_analysis = SumfftAnalysis(dataset=ds, parameters={self.params})",
                         "sumfft_result = ctx.run(sumfft_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(sumfft_result.intensity, norm=colors.LogNorm())",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class SumfftAnalysis(SumAnalysis, id_="FFTSUM_FRAMES"):
    TYPE = 'UDF'

    def get_udf_results(self, udf_results, roi, damage):
        from libertem.viz import visualize_simple
        sum_results = np.array(udf_results['intensity'])
        real_rad = self.parameters.get("real_rad")
        real_center = (self.parameters.get("real_centery"), self.parameters.get("real_centerx"))

        if not (real_center is None or real_rad is None):
            sigshape = sum_results.shape
            real_mask = 1-1*_make_circular_mask(
                    real_center[1], real_center[0], sigshape[1], sigshape[0], real_rad
                )
            fft_result = np.log(abs(np.fft.fftshift(np.fft.fft2(sum_results*real_mask)))+1)
        else:
            fft_result = np.log(abs(np.fft.fftshift(np.fft.fft2(sum_results)))+1)
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=sum_results,
                visualized=visualize_simple(fft_result, damage=True),
                key="intensity", title="intensity", desc="fft of sum of all frames"
            ),
        ])

    @classmethod
    def get_template_helper(cls):
        return SumfftTemplate
