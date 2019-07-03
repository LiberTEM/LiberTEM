import numpy as np
from libertem.viz import visualize_simple
from libertem.common import Slice, Shape
from libertem.job.raw import PickFrameJob
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class PickFrameAnalysis(BaseAnalysis):
    def get_job(self):
        assert self.dataset.shape.nav.dims in (1, 2, 3), "can only handle 1D/2D/3D nav currently"
        x, y, z = (
            self.parameters.get('x'),
            self.parameters.get('y'),
            self.parameters.get('z'),
        )

        if self.dataset.shape.nav.dims == 1:
            if x is None:
                raise ValueError("need x to index 1D nav datasets")
            if y is not None or z is not None:
                raise ValueError("y and z must not be specified for 1D nav dataset")
            origin = (x,)
        elif self.dataset.shape.nav.dims == 2:
            if x is None or y is None:
                raise ValueError("need x/y to index 2D nav datasets")
            if z is not None:
                raise ValueError("z must not be specified for 2D nav dataset")
            origin = (y, x)
        elif self.dataset.shape.nav.dims == 3:
            if x is None or y is None or z is None:
                raise ValueError("need x/y/z to index 3D nav datasets")
            origin = (z, y, x)
        else:
            raise ValueError("cannot operate on datasets with more than 3 nav dims")

        origin = (np.ravel_multi_index(origin, self.dataset.shape.nav),)
        shape = self.dataset.shape

        origin = origin + tuple([0] * self.dataset.shape.sig.dims)

        return PickFrameJob(
            dataset=self.dataset,
            slice_=Slice(
                origin=origin,
                shape=Shape((1,) + tuple(shape.sig),
                            sig_dims=shape.sig.dims),
            ),
            squeeze=True,
        )

    def get_results_base(self, job_results):
        parameters = self.parameters
        coords = [
            "%s=%d" % (axis, parameters.get(axis))
            for axis in ['x', 'y', 'z']
            if axis in parameters
        ]
        coords = " ".join(coords)
        shape = tuple(self.dataset.shape.sig)
        data = job_results.reshape(shape)
        return data, coords

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
        visualized = visualize_simple(data, logarithmic=True)
        return AnalysisResultSet([
            AnalysisResult(raw_data=data, visualized=visualized,
                           key="intensity", title="intensity",
                           desc="the frame at %s" % (coords,)),
        ])
