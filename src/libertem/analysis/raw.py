from libertem.viz import visualize_simple
from libertem.common import Slice, Shape
from libertem.job.raw import PickFrameJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class PickFrameAnalysis(BaseAnalysis):
    def get_job(self):
        x, y = self.parameters['x'], self.parameters['y']
        shape = self.dataset.shape
        assert shape.nav.dims == 2, "can only handle 2D nav currently"
        return PickFrameJob(
            dataset=self.dataset,
            slice_=Slice(
                origin=(y, x) + tuple([0] * shape.sig.dims),
                shape=Shape(tuple([1] * shape.nav.dims) + tuple(shape.sig),
                            sig_dims=shape.sig.dims),
            ),
            squeeze=True,
        )

    def get_results(self, job_results):
        x, y = self.parameters['x'], self.parameters['y']
        return AnalysisResultSet([
            AnalysisResult(raw_data=job_results, visualized=visualize_simple(job_results),
                           key="intensity", title="intensity",
                           desc="the frame at x=%d y=%d" % (x, y)),
        ])
