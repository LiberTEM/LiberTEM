from functools import reduce
import logging

import numpy as np

from libertem import masks
from libertem.viz import CMAP_CIRCULAR_DEFAULT, visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


log = logging.getLogger(__name__)


def divergence(arr):
    return reduce(np.add, [np.gradient(arr[i], axis=i)
                           for i in range(len(arr))])


class COMAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.nav)
        img_sum, img_x, img_y = (
            job_results[0].reshape(shape),
            job_results[1].reshape(shape),
            job_results[2].reshape(shape)
        )
        ref_x = self.parameters["cx"]
        ref_y = self.parameters["cy"]
        x_centers = np.divide(img_x, img_sum, where=img_sum != 0)
        y_centers = np.divide(img_y, img_sum, where=img_sum != 0)
        x_centers[img_sum == 0] = ref_x
        y_centers[img_sum == 0] = ref_y
        x_centers -= ref_x
        y_centers -= ref_y

        if img_sum.dtype.kind == 'c':
            x_real, x_imag = np.real(x_centers), np.imag(x_centers)
            y_real, y_imag = np.real(y_centers), np.imag(y_centers)

            return AnalysisResultSet([
                AnalysisResult(raw_data=x_real, visualized=visualize_simple(x_real),
                       key="x_real", title="x [real]", desc="x component of the center"),
                AnalysisResult(raw_data=y_real, visualized=visualize_simple(y_real),
                       key="y_real", title="y [real]", desc="y component of the center"),
                AnalysisResult(raw_data=x_imag, visualized=visualize_simple(x_imag),
                       key="x_imag", title="x [imag]", desc="x component of the center"),
                AnalysisResult(raw_data=y_imag, visualized=visualize_simple(y_imag),
                       key="y_imag", title="y [imag]", desc="y component of the center"),
            ])
        else:
            f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((y_centers, x_centers))
            d = divergence([x_centers, y_centers])
            m = np.sqrt(x_centers**2 + y_centers**2)

            return AnalysisResultSet([
                AnalysisResult(raw_data=(x_centers, y_centers), visualized=f,
                       key="field", title="field", desc="cubehelix colorwheel visualization"),
                AnalysisResult(raw_data=m, visualized=visualize_simple(m),
                       key="magnitude", title="magnitude", desc="magnitude of the vector field"),
                AnalysisResult(raw_data=d, visualized=visualize_simple(d),
                       key="divergence", title="divergence", desc="divergence of the vector field"),
                AnalysisResult(raw_data=x_centers, visualized=visualize_simple(x_centers),
                       key="x", title="x", desc="x component of the center"),
                AnalysisResult(raw_data=y_centers, visualized=visualize_simple(y_centers),
                       key="y", title="y", desc="y component of the center"),
            ])

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")

        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters['cx']
        cy = self.parameters['cy']
        r = self.parameters['r']

        def disk_mask():
            return masks.circular(
                centerX=cx, centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=r,
            )

        return [
            disk_mask,
            lambda: masks.gradient_x(
                imageSizeX=detector_x,
                imageSizeY=detector_y,
            ) * disk_mask(),
            lambda: masks.gradient_y(
                imageSizeX=detector_x,
                imageSizeY=detector_y,
            ) * disk_mask(),
        ]

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        r = parameters.get('r', float('inf'))
        use_sparse = parameters.get('use_sparse', False)

        return {
            'cx': cx,
            'cy': cy,
            'r': r,
            'use_sparse': use_sparse,
            'mask_count': 3,
            'mask_dtype': np.float32,
        }
