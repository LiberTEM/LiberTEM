from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


class RingMaskAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data,
                visualized=visualize_simple(data),
                key="intensity",
                title="intensity",
                desc="intensity of the integration over the selected ring"),
        ])

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters.get('cx', detector_x / 2)
        cy = self.parameters.get('cy', detector_y / 2)
        ro = self.parameters.get('ro', min(detector_y, detector_x) / 2)
        ri = self.parameters.get('ri', ro * 0.8)

        def _ring_inner():
            return masks.ring(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=ro,
                radius_inner=ri)

        return [_ring_inner]
