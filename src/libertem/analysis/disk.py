from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


class DiskMaskAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        shape = tuple(self.dataset.effective_shape.nav)
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data.reshape(shape),
                visualized=visualize_simple(data.reshape(shape)),
                key="intensity",
                title="intensity",
                desc="intensity of the integration over the selected disk"),
        ])

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters.get('cx', detector_x / 2)
        cy = self.parameters.get('cy', detector_y / 2)
        r = self.parameters.get('r', min(detector_y, detector_x) / 2 * 0.3)

        def disk_mask():
            return masks.circular(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=r,
            )

        return [
            disk_mask,
        ]
