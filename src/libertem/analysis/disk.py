from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult
from .masks import MasksAnalysis


class DiskMaskAnalysis(MasksAnalysis):
    def get_results(self, job_results):
        data = job_results[0]
        return [
            AnalysisResult(raw_data=data, visualized=visualize_simple(data),
                   title="intensity", desc="intensity of the integration over the selected disk"),
        ]

    def get_mask_factories(self):
        cx = self.parameters['cx']
        cy = self.parameters['cy']
        r = self.parameters['r']
        frame_size = self.dataset.shape[2:]

        def disk_mask():
            return masks.circular(
                centerX=cx, centerY=cy,
                imageSizeX=frame_size[1],
                imageSizeY=frame_size[0],
                radius=r,
            )
        return [
            disk_mask,
        ]
