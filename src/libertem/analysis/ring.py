import numpy as np

from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


class RingMaskAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.nav)
        data = job_results[0].reshape(shape)
        if data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    data.reshape(shape),
                    key_prefix='intensity',
                    title='intensity',
                    desc="intensity of the integration over the selected ring",
                )
            )
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data.reshape(shape),
                visualized=visualize_simple(data),
                key="intensity",
                title="intensity",
                desc="intensity of the integration over the selected ring"),
        ])

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters['cx']
        cy = self.parameters['cy']
        ri = self.parameters['ri']
        ro = self.parameters['ro']

        def _ring_inner():
            return masks.ring(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=ro,
                radius_inner=ri)

        return [_ring_inner]

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        ro = parameters.get('ro', min(detector_y, detector_x) / 2)
        ri = parameters.get('ri', ro * 0.8)
        use_sparse = parameters.get('use_sparse', False)

        return {
            'cx': cx,
            'cy': cy,
            'ri': ri,
            'ro': ro,
            'use_sparse': use_sparse,
            'mask_count': 1,
            'mask_dtype': np.float32,
        }
