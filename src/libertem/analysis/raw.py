from libertem.viz import visualize_simple
from libertem.common.slice import Slice
from libertem.job.raw import PickFrameJob
from .base import BaseAnalysis, AnalysisResult


class PickFrameAnalysis(BaseAnalysis):
    def get_job(self):
        x, y = self.parameters['x'], self.parameters['y']
        shape = self.dataset.shape
        return PickFrameJob(
            dataset=self.dataset,
            slice_=Slice(
                origin=(y, x, 0, 0),
                shape=(1, 1) + shape[2:],
            ),
            squeeze=True,
        )

    def get_results(self, job_results):
        x, y = self.parameters['x'], self.parameters['y']
        return [
            AnalysisResult(raw_data=job_results, visualized=visualize_simple(job_results),
                           title="intensity", desc="the frame at x=%d y=%d" % (x, y)),
        ]
