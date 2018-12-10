import numpy as np
from libertem.viz import visualize_simple
from .base import BaseAnalysis, AnalysisResultSet, AnalysisResult
from libertem.job.masks import ApplyMasksJob


class BaseMasksAnalysis(BaseAnalysis):
    """
    Base class for any masks-based analysis; you only need to implement
    ``get_results`` and ``get_mask_factories``.
    Overwrite  ``get_use_sparse`` to return True to calculate with sparse mask matrices.
    """

    @property
    def dtype(self):
        return np.dtype(
            self.dataset.dtype).kind == 'f' and self.dataset.dtype or "float32"

    def get_job(self):
        mask_factories = self.get_mask_factories()
        use_sparse = self.get_use_sparse()
        job = ApplyMasksJob(
            dataset=self.dataset,
            mask_factories=mask_factories,
            use_sparse=use_sparse)
        return job

    def get_mask_factories(self):
        raise NotImplementedError()

    def get_use_sparse(self):
        return False


class MasksAnalysis(BaseMasksAnalysis):
    def get_mask_factories(self):
        return self.parameters['factories']

    def get_use_sparse(self):
        return self.parameters['use_sparse']

    def get_results(self, job_results):
        shape = tuple(self.dataset.effective_shape.nav)
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=mask_result.reshape(shape),
                visualized=visualize_simple(mask_result.reshape(shape)),
                key="mask_%d" % i,
                title="mask %d" % i,
                desc="intensity for mask %d" % i)
            for i, mask_result in enumerate(job_results)
        ])
