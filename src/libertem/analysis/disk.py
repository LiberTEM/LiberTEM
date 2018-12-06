from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


class DiskMaskAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data,
                visualized=visualize_simple(data),
                key="intensity",
                title="intensity",
                desc="intensity of the integration over the selected disk"),
        ])

    def get_mask_factories(self):
        (detector_y, detector_x) = self.dataset.shape[2:]

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
