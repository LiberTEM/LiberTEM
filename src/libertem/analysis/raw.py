import numpy as np
from libertem.viz import visualize_simple
from libertem.common import Slice, Shape
from libertem.job.raw import PickFrameJob
from libertem.udf.raw import PickUDF
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet


class PickResultSet(AnalysisResultSet):
    """
    Running a :class:`PickFrameAnalysis` via :meth:`libertem.api.Context.run`
    returns an instance of this class.

    If the dataset contains complex numbers, the regular result attribute carries the
    absolute value of the result and additional attributes with real part, imaginary part,
    phase and full complex result are available.

    .. versionadded:: 0.3.0

    Attributes
    ----------
    intensity : libertem.analysis.base.AnalysisResult
        The specified detector frame. Absolute value if the dataset
        contains complex numbers.
    intensity_real : libertem.analysis.base.AnalysisResult
        Real part of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_angle : libertem.analysis.base.AnalysisResult
        Phase angle of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_complex : libertem.analysis.base.AnalysisResult
        Complex result of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    """
    pass


class PickFrameAnalysis(BaseAnalysis):
    TYPE = 'UDF'
    """
    Pick a single, complete frame from a dataset
    """

    def get_origin(self):
        assert self.dataset.shape.nav.dims in (1, 2, 3), "can only handle 1D/2D/3D nav currently"
        x, y, z = (
            self.parameters.get('x'),
            self.parameters.get('y'),
            self.parameters.get('z'),
        )

        funcs = {
            1: self.get_origin_1d,
            2: self.get_origin_2d,
            3: self.get_origin_3d,
        }

        def value_error(x, y, z):
            raise ValueError("cannot operate on datasets with more than 3 nav dims")

        f = funcs.get(self.dataset.shape.nav.dims, value_error)
        return f(x, y, z)

    def get_origin_1d(self, x, y, z):
        if x is None:
            raise ValueError("need x to index 1D nav datasets")
        if y is not None or z is not None:
            raise ValueError("y and z must not be specified for 1D nav dataset")
        return (x,)

    def get_origin_2d(self, x, y, z):
        if x is None or y is None:
            raise ValueError("need x/y to index 2D nav datasets")
        if z is not None:
            raise ValueError("z must not be specified for 2D nav dataset")
        return (y, x)

    def get_origin_3d(self, x, y, z):
        if x is None or y is None or z is None:
            raise ValueError("need x/y/z to index 3D nav datasets")
        return (z, y, x)

    def get_job(self):
        origin = self.get_origin()
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

    def get_udf(self):
        return PickUDF()

    def get_roi(self):
        roi = np.zeros(self.dataset.shape.nav, dtype=bool)
        roi[self.get_origin()] = True
        return roi

    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.sig)
        data = job_results.reshape(shape)
        return self.get_generic_results(data)

    def get_udf_results(self, udf_results, roi):
        return self.get_generic_results(udf_results['intensity'].data[0])

    def get_coords(self):
        parameters = self.parameters
        coords = [
            "%s=%d" % (axis, parameters.get(axis))
            for axis in ['x', 'y', 'z']
            if axis in parameters
        ]
        return " ".join(coords)

    def get_generic_results(self, data):
        coords = self.get_coords()

        if data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    data,
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
