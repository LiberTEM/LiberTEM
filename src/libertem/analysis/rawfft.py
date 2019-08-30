import numpy as np
from libertem.viz import visualize_simple
from libertem.masks import _make_circular_mask
from .base import AnalysisResult, AnalysisResultSet
from .raw import PickFrameAnalysis


class PickFFTFrameAnalysis(PickFrameAnalysis):
    def get_results(self, job_results):
        data, coords = self.get_results_base(job_results)
        real_rad = self.parameters.get("real_rad")
        real_center = (self.parameters.get("real_centery"), self.parameters.get("real_centerx"))
        if data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    job_results,
                    key_prefix="intensity",
                    title="intensity",
                    desc="the frame at %s" % (coords,),
                )
            )
        if not (real_center is None or real_rad is None):
            sigshape = job_results.shape
            real_mask = 1-1*_make_circular_mask(
                    real_center[1], real_center[0], sigshape[1], sigshape[0], real_rad
                )
            visualized = visualize_simple(np.fft.fftshift(abs(np.fft.fft2(data*real_mask))),
                logarithmic=True)
        else:
            visualized = visualize_simple(np.fft.fftshift(abs(np.fft.fft2(data))), logarithmic=True)
        return AnalysisResultSet([
            AnalysisResult(raw_data=data, visualized=visualized,
                           key="intensity", title="intensity",
                           desc="the frame at %s" % (coords,)),
        ])
