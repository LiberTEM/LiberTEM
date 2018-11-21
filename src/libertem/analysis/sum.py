from libertem.viz import visualize_simple
from libertem.job.sum import SumFramesJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class SumAnalysis(BaseAnalysis):
    def get_job(self):
        return SumFramesJob(dataset=self.dataset)

    def get_results(self, job_results):
        data = job_results
        return AnalysisResultSet([
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                   key="intensity", title="intensity", desc="sum of all frames"),
        ])
