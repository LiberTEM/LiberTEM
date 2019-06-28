import numpy as np
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .raw import PickFrameAnalysis


class PickFFTFrameAnalysis(PickFrameAnalysis):   
    def get_results(self, job_results):
        data, coords = self.get_results_base(job_results)
        
        if data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    job_results,
                    key_prefix="intensity",
                    title="intensity",
                    desc="the frame at %s" % (coords,),
                )
            )
        visualized = visualize_simple(np.fft.fftshift(abs(np.fft.fft2(data))), logarithmic=True)
        return AnalysisResultSet([
            AnalysisResult(raw_data=data, visualized=visualized,
                           key="intensity", title="intensity",
                           desc="the frame at %s" % (coords,)),
        ])