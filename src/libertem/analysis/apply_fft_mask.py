from libertem.viz import visualize_simple
from libertem.job.sum import SumFramesJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
import numpy as np
import libertem.udf.crystallinity as crystal
crystal.run_analysis_crystall

class SumfftAnalysis(BaseAnalysis):
    TYPE = "UDF"
    def get_job(self):
        return SumFramesJob(dataset=self.dataset)

    def get_results(self, job_results):
        fft_result = np.log(abs(np.fft.fftshift(np.fft.fft2(job_results)))+1)
        return AnalysisResultSet([
            AnalysisResult(raw_data=job_results, visualized=visualize_simple(fft_result),
                   key="intensity", title="intensity", desc="sum of all frames"),
        ])
