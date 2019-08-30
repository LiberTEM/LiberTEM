from libertem.viz import visualize_simple
from libertem.analysis.sum import SumAnalysis
from libertem.masks import _make_circular_mask
from .base import AnalysisResult, AnalysisResultSet

import numpy as np


class SumfftAnalysis(SumAnalysis):
    TYPE = 'UDF'

    def get_udf_results(self, udf_results):
        job_results = np.array(udf_results['intensity'])
        real_rad = self.parameters.get("real_rad")
        real_center = (self.parameters.get("real_centery"), self.parameters.get("real_centerx"))
        if not (real_center is None or real_rad is None):
            sigshape = job_results.shape
            real_mask = 1-1*_make_circular_mask(
                    real_center[1], real_center[0], sigshape[1], sigshape[0], real_rad
                )
            fft_result = np.log(abs(np.fft.fftshift(np.fft.fft2(job_results*real_mask)))+1)
        else:
            fft_result = np.log(abs(np.fft.fftshift(np.fft.fft2(job_results)))+1)
        return AnalysisResultSet([
            AnalysisResult(raw_data=job_results, visualized=visualize_simple(fft_result),
                   key="intensity", title="intensity", desc="fft of sum of all frames"),
        ])
