import logging

import numpy as np

from libertem import masks
from libertem.viz import CMAP_CIRCULAR_DEFAULT, visualize_simple, cmaps
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis


log = logging.getLogger(__name__)


class RadialFourierAnalysis(BaseMasksAnalysis):
    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.nav)
        sets = []
        orders = self.parameters['max_order'] + 1
        for b in range(self.parameters['n_bins']):
            for o in range(orders):
                index = b * orders + 0
                data = job_results[index].reshape(shape)
                absolute = np.absolute(data)
                angle = np.angle(data)
                sets.append(
                    AnalysisResult(
                        raw_data=absolute,
                        visualized=visualize_simple(absolute),
                        key="absolute_%s_%s" % (b, o),
                        title="absolute of bin %s order %s" % (b, o),
                        desc="Absolute value of Fourier component",
                    )
                )
                sets.append(
                    AnalysisResult(
                        raw_data=angle,
                        visualized=visualize_simple(angle, colormap=cmaps['perception_circular']),
                        key="phase_%s_%s" % (b, o),
                        title="phase of bin %s order %s" % (b, o),
                        desc="Phase of Fourier component",
                    )
                )
                f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((data.imag, data.real))
                sets.append(
                    AnalysisResult(
                        raw_data=data,
                        visualized=f,
                        key="complex_%s_%s" % (b, o),
                        title="bin %s order %s" % (b, o),
                        desc="Fourier component",
                    )
                )
        return AnalysisResultSet(sets)

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")

        (detector_y, detector_x) = self.dataset.shape.sig
        p = self.parameters

        cx = p['cx']
        cy = p['cy']
        ri = p['ri']
        ro = p['ro']
        n_bins = p['n_bins']
        max_order = p['max_order']

        def stack():
            rings = masks.radial_bins(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=ro,
                radius_inner=ri,
                n_bins=n_bins
            )
            orders = np.arange(max_order + 1)
            r, phi = masks.polar_map(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y
            )
            modulator = np.exp(phi * orders[:, np.newaxis, np.newaxis] * 1j)
            rings = rings.todense()
            s = rings[:, np.newaxis, ...] * modulator
            return s.reshape((-1, detector_y, detector_x))

        return stack

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        ri = parameters.get('ri', 0)
        ro = parameters.get('ro', None)
        n_bins = parameters.get('n_bins', 1)
        max_order = parameters.get('max_order', 24)

        use_sparse = parameters.get('use_sparse', 'scipy.sparse')

        return {
            'cx': cx,
            'cy': cy,
            'ri': ri,
            'ro': ro,
            'n_bins': n_bins,
            'max_order': max_order,
            'use_sparse': use_sparse,
            'mask_count': n_bins * (max_order + 1),
            'mask_dtype': np.complex64,
        }
