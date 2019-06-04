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
            return AnalysisResultSet(results)
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=mask_result.reshape(shape),
                visualized=visualize_simple(mask_result.reshape(shape)),
                key="mask_%d" % i,
                title="mask %d" % i,
                desc="integrated intensity for mask %d" % i)
            for i, mask_result in enumerate(job_results)
        ])
