import numpy as np
from .base import BaseAnalysis
from libertem.job.masks import ApplyMasksJob


class MasksAnalysis(BaseAnalysis):
    """
    base class for any masks-based analysis; you only need to implement
    ``get_results`` and ``get_mask_factories``
    """

    @property
    def dtype(self):
        return np.dtype(self.dataset.dtype).kind == 'f' and self.dataset.dtype or "float32"

    def get_job(self):
        mask_factories = self.get_mask_factories()
        job = ApplyMasksJob(dataset=self.dataset, mask_factories=mask_factories)
        return job

    def get_mask_factories(self):
        raise NotImplementedError()
