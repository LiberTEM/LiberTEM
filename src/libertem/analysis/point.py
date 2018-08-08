import numpy as np
from libertem.viz import visualize_simple
from .base import AnalysisResult
from .masks import MasksAnalysis


class PointMaskAnalysis(MasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        return [
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                   title="intensity", desc="intensity of the integration over the selected point"),
        ]

    def get_mask_factories(self):
        cx = self.parameters['cx']
        cy = self.parameters['cy']

        def _point_inner():
            a = np.zeros(self.dataset.shape[2:])
            a[int(cy), int(cx)] = 1
            return a.astype(self.dtype)
        return [_point_inner]
