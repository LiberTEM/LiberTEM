from libertem.viz import visualize_simple
from libertem.job.sum import SumFramesJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class SumAnalysis(BaseAnalysis):
    def get_job(self):
        return SumFramesJob(dataset=self.dataset)

    def get_results(self, job_results):
        if job_results.dtype.kind == 'c':
            results = []
            return AnalysisResultSet(
                self.get_complex_results(
                    job_results,
                    key_prefix="intensity",
                    title="intensity",
                    desc="sum of all frames",
                )
            )
            return AnalysisResultSet(results)
        return AnalysisResultSet([
            AnalysisResult(raw_data=job_results, visualized=visualize_simple(job_results),
                   key="intensity", title="intensity", desc="sum of all frames"),
        ])
