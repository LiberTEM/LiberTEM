import numpy as np
from libertem.masks import _make_circular_mask
from .raw import PickFrameAnalysis
from .helper import GeneratorHelper


class PickfftTemplate(GeneratorHelper):

    short_name = "pickfft"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from libertem.analysis import PickFFTFrameAnalysis"]

    def get_docs(self):
        docs = ["# PICK FFT Analysis",
                "***about pickfft analysis ***"]
        return '\n'.join(docs)

    def get_analysis(self):
        temp_analysis = [f"pickfft_analysis = PickFFTFrameAnalysis(dataset=ds, parameters={self.params})",
                         "pickfft_result = ctx.run(pickfft_analysis, progress=True)"]
        return '\n'.join(temp_analysis)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(pickfft_result.intensity.visualized)"]
        return '\n'.join(plot)


class PickFFTFrameAnalysis(PickFrameAnalysis, id_="PICK_FFT_FRAME"):

    def get_results(self, job_results):
        # Make sure we don't use legacy code from superclasses
        raise NotImplementedError

    def get_job(self):
        # Make sure we don't use legacy code from superclasses
        raise NotImplementedError

    def get_udf_results(self, udf_results, roi):
        data = udf_results['intensity'].data[0]
        real_rad = self.parameters.get("real_rad")
        real_center = (self.parameters.get("real_centery"), self.parameters.get("real_centerx"))
        if data.dtype.kind == 'c':
            return self.get_generic_results(data)
        if not (real_center is None or real_rad is None):
            sigshape = data.shape
            real_mask = 1-1*_make_circular_mask(
                real_center[1], real_center[0], sigshape[1], sigshape[0], real_rad
            )
            fft_data = np.fft.fftshift(abs(np.fft.fft2(data*real_mask)))
        else:
            fft_data = np.fft.fftshift(abs(np.fft.fft2(data)))
        return self.get_generic_results(fft_data)

    @classmethod
    def get_template_helper(cls):
        return PickfftTemplate
