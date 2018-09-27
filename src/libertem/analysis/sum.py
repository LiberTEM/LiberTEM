from libertem.viz import visualize_simple
from libertem.job.sum import SumFramesJob
from .base import BaseAnalysis, AnalysisResult


class SumAnalysis(BaseAnalysis):
    def get_job(self):
        return SumFramesJob(dataset=self.dataset)

    def get_results(self, job_results):
        data = job_results
        return [
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                   title="intensity", desc="sum of all frames"),
        ]
