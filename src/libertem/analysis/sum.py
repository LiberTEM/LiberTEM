import numpy as np

from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from libertem.analysis.getroi import get_roi
from libertem.udf.sum import SumUDF


class SumResultSet(AnalysisResultSet):
    """
    Running a :class:`SumAnalysis` via :meth:`libertem.api.Context.run`
    returns an instance of this class.

    If the dataset contains complex numbers, the regular result attribute carries the
    absolute value of the result and additional attributes with real part, imaginary part,
    phase and full complex result are available.

    .. versionadded:: 0.3.0

    Attributes
    ----------
    intensity : libertem.analysis.base.AnalysisResult
        Sum of all detector frames along the navigation dimension,
        preserving the signal dimension. Absolute value of the sum if the dataset
        contains complex numbers.
    intensity_real : libertem.analysis.base.AnalysisResult
        Real part of the sum of all detector frames along the navigation dimension,
        preserving the signal dimension. This is only available if the dataset
        contains complex numbers.
    intensity_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of the sum of all detector frames along the navigation dimension,
        preserving the signal dimension. This is only available if the dataset
        contains complex numbers.
    intensity_angle : libertem.analysis.base.AnalysisResult
        Phase angle of the sum of all detector frames along the navigation dimension,
        preserving the signal dimension. This is only available if the dataset
        contains complex numbers.
    intensity_complex : libertem.analysis.base.AnalysisResult
        Complex result of the sum of all detector frames along the navigation dimension,
        preserving the signal dimension. This is only available if the dataset
        contains complex numbers.
    """
    pass


class SumAnalysis(BaseAnalysis):
    TYPE = 'UDF'

    def get_udf(self):
        dest_dtype = np.dtype(self.dataset.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        return SumUDF(dtype=dest_dtype)

    def get_roi(self):
        return get_roi(params=self.parameters, shape=self.dataset.shape.nav)

    def get_udf_results(self, udf_results, roi):
        if udf_results['intensity'].data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    udf_results['intensity'].data,
                    key_prefix="intensity",
                    title="intensity",
                    desc="sum of all frames",
                )
            )

        return SumResultSet([
            AnalysisResult(raw_data=udf_results['intensity'].data,
                           visualized=visualize_simple(
                               udf_results['intensity'].data, logarithmic=True
                           ),
                           key="intensity", title="intensity", desc="sum of frames"),
        ])
