import numpy as np
from libertem.viz import visualize_simple
from libertem.common import Slice, Shape
from libertem.job.raw import PickFrameJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class PickFrameAnalysis(BaseAnalysis):
    def get_job(self):
        assert self.dataset.shape.nav.dims == 2, "can only handle 2D nav currently"
        x, y = self.parameters['x'], self.parameters['y']
        shape = self.dataset.raw_shape

        if shape.nav.dims == 2:
            origin = (y, x)
        else:
            origin = (np.ravel_multi_index((y, x), self.dataset.shape.nav),)

        return PickFrameJob(
            dataset=self.dataset,
            slice_=Slice(
                origin=origin + tuple([0] * shape.sig.dims),
                shape=Shape(tuple([1] * shape.nav.dims) + tuple(shape.sig),
                            sig_dims=shape.sig.dims),
            ),
            squeeze=True,
        )

    def get_results(self, job_results):
        x, y = self.parameters['x'], self.parameters['y']
        shape = tuple(self.dataset.shape.sig)
        data = job_results.reshape(shape)

        return AnalysisResultSet([
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                           key="intensity", title="intensity",
                           desc="the frame at x=%d y=%d" % (x, y)),
        ])
