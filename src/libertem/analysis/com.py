from functools import reduce
import logging

import numpy as np

from libertem import masks
from libertem.viz import CMAP_CIRCULAR_DEFAULT, visualize_simple
from .base import AnalysisResult
from .masks import MasksAnalysis


log = logging.getLogger(__name__)


def divergence(arr):
    return reduce(np.add, [np.gradient(arr[i], axis=i)
                           for i in range(len(arr))])


class COMAnalysis(MasksAnalysis):
    def get_results(self, job_results):
        img_sum, img_x, img_y = job_results[0], job_results[1], job_results[2]
        ref_x = self.parameters["cx"]
        ref_y = self.parameters["cy"]
        x_centers = np.divide(img_x, img_sum, where=img_sum != 0)
        y_centers = np.divide(img_y, img_sum, where=img_sum != 0)
        x_centers[img_sum == 0] = ref_x
        y_centers[img_sum == 0] = ref_y
        x_centers -= ref_x
        y_centers -= ref_y
        d = divergence([x_centers, y_centers])
        m = np.sqrt(x_centers**2 + y_centers**2)
        f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((y_centers, x_centers))

        return [
            AnalysisResult(raw_data=(x_centers, y_centers), visualized=f,
                   title="field", desc="cubehelix colorwheel visualization"),
            AnalysisResult(raw_data=m, visualized=visualize_simple(m),
                   title="magnitude", desc="magnitude of the vector field"),
            AnalysisResult(raw_data=d, visualized=visualize_simple(d),
                   title="divergence", desc="divergence of the vector field"),
            AnalysisResult(raw_data=x_centers, visualized=visualize_simple(x_centers),
                   title="x", desc="x component of the center"),
            AnalysisResult(raw_data=y_centers, visualized=visualize_simple(y_centers),
                   title="y", desc="y component of the center"),
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
            lambda: masks.gradient_x(
                imageSizeX=frame_size[1],
                imageSizeY=frame_size[0],
                dtype=self.dtype,
            ) * (np.ones(frame_size) * disk_mask()),
            lambda: masks.gradient_y(
                imageSizeX=frame_size[1],
                imageSizeY=frame_size[0],
                dtype=self.dtype,
            ) * (np.ones(frame_size) * disk_mask()),
        ]
