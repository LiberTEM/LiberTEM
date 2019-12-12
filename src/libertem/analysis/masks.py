from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResultSet, AnalysisResult
from libertem.job.masks import ApplyMasksJob


class BaseMasksAnalysis(BaseAnalysis):
    """
    Base class for any masks-based analysis; you only need to implement
    ``get_results`` and ``get_mask_factories``.
    Overwrite  ``get_use_sparse`` to return True to calculate with sparse mask matrices.
    """

    def get_job(self):
        mask_factories = self.get_mask_factories()
        use_sparse = self.get_use_sparse()
        mask_count = self.get_preset_mask_count()
        mask_dtype = self.get_preset_mask_dtype()
        dtype = self.get_preset_dtype()
        job = ApplyMasksJob(
            dataset=self.dataset,
            mask_factories=mask_factories,
            use_sparse=use_sparse,
            mask_count=mask_count,
            mask_dtype=mask_dtype,
            dtype=dtype)
        return job

    def get_mask_factories(self):
        raise NotImplementedError()

    def get_use_sparse(self):
        return self.parameters.get('use_sparse', None)

    def get_preset_mask_count(self):
        return self.parameters.get('mask_count', None)

    def get_preset_mask_dtype(self):
        return self.parameters.get('mask_dtype', None)

    def get_preset_dtype(self):
        return self.parameters.get('dtype', None)


class MasksResultSet(AnalysisResultSet):
    """
    Running a :class:`MasksAnalysis` via :meth:`libertem.api.Context.run` on a dataset
    returns an instance of this class.

    If any of the masks or the dataset contain complex numbers, the regular mask results
    attributes carry the absolute value of the results, and additional attributes with real
    part, imaginary part, phase and full complex result are available.

    .. versionadded:: 0.3

    Attributes
    ----------
    mask_0, mask_1, ..., mask_<n> : libertem.analysis.base.AnalysisResult
        For dataset and masks containing only real numbers:
        Results of the element-wise multiplication and sum of each individual mask with
        each detector frame. Each mask result has the shape of the navigation dimension.
        These keys contain the absolute value of the result if dataset or masks contain
        complex numbers.
    mask_0, mask_0_real, mask_0_imag, mask_0_angle, mask_0_complex,\
    mask_1, mask_1_real, ...,\
    mask_<n>, ..., mask_<n>_complex : libertem.analysis.base.AnalysisResult
        If masks or dataset contain complex numbers: Absolute, real part, imaginary part,
        phase angle, complex result of the element-wise multiplication and sum of each individual
        mask with each detector frame. Each mask result has the shape of the navigation dimension.
    """
    pass


class SingleMaskResultSet(AnalysisResultSet):
    """
    A number of Analyses that are based on applying a single mask create an instance of this class
    as a result when executed via :meth:`libertem.api.Context.run`.

    If the dataset contains complex numbers, the regular result attribute carries the
    absolute value of the result, and additional attributes with real part, imaginary part,
    phase and full complex result are available.

    .. versionadded:: 0.3

    Attributes
    ----------
    intensity : libertem.analysis.base.AnalysisResult
        Sum of the selected region for each detector frame, with shape of
        the navigation dimension. Absolute of the result if the dataset or mask contains
        complex numbers.
    intensity_real : libertem.analysis.base.AnalysisResult
        Real part of the sum of the selected region. This is only available if the dataset
        or mask contains complex numbers.
    intensity_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of the sum of the selected region. This is only available if the dataset
        contains complex numbers.
    intensity_angle : libertem.analysis.base.AnalysisResult
        Phase angle of the sum of the selected region. This is only available if the dataset
        or mask contains complex numbers.
    intensity_complex : libertem.analysis.base.AnalysisResult
        Complex result of the sum of the selected region. This is only available if the dataset
        or mask contains complex numbers.
    """
    pass


class MasksAnalysis(BaseMasksAnalysis):
    def get_mask_factories(self):
        return self.parameters['factories']

    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.nav)
        if job_results.dtype.kind == 'c':
            results = []
            for idx, job_result in enumerate(job_results):
                results.extend(
                    self.get_complex_results(
                        job_result.reshape(shape),
                        key_prefix="mask_%d" % idx,
                        title="mask %d" % idx,
                        desc="integrated intensity for mask %d" % idx,
                    )
                )
            return MasksResultSet(results)
        return MasksResultSet([
            AnalysisResult(
                raw_data=mask_result.reshape(shape),
                visualized=visualize_simple(mask_result.reshape(shape)),
                key="mask_%d" % i,
                title="mask %d" % i,
                desc="integrated intensity for mask %d" % i)
            for i, mask_result in enumerate(job_results)
        ])
