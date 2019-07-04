from libertem.viz import visualize_simple
from libertem.job.sum import SumFramesJob
from libertem.masks import _make_circular_mask
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import numpy as np


class SumfftAnalysis(BaseAnalysis):
    def get_job(self):
        return SumFramesJob(dataset=self.dataset)

    def get_results(self, job_results):
        real_rad = self.parameters["real_rad"]
        real_center = (self.parameters["real_centery"], self.parameters["real_centerx"])
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
                   key="intensity", title="intensity", desc="sum of all frames"),
        ])
