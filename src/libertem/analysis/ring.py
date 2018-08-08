from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult
from .masks import MasksAnalysis


class RingMaskAnalysis(MasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        return [
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                   title="intensity", desc="intensity of the integration over the selected ring"),
        ]

    def get_mask_factories(self):
        cx = self.parameters['cx']
        cy = self.parameters['cy']
        ri = self.parameters['ri']
        ro = self.parameters['ro']
        frame_size = self.dataset.shape[2:]

        def _ring_inner():
            return masks.ring(
                centerX=cx, centerY=cy,
                imageSizeX=frame_size[1],
                imageSizeY=frame_size[0],
                radius=ro,
                radius_inner=ri
            )
        return [_ring_inner]
