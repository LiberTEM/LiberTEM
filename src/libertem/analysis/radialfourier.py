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
        n_bins = self.parameters['n_bins']
        job_results = job_results.reshape((n_bins, orders, *shape))
        absolute = np.absolute(job_results)
        normal = absolute[:, 0]
        min_absolute = np.min(absolute[:, 1:, ...] / normal[:, np.newaxis, ...])
        max_absolute = np.max(absolute[:, 1:, ...] / normal[:, np.newaxis, ...])
        angle = np.angle(job_results)
        for b in range(self.parameters['n_bins']):
            sets.append(
                AnalysisResult(
                    raw_data=absolute[b, 0],
                    visualized=visualize_simple(absolute[b, 0]),
                    key="absolute_%s_%s" % (b, 0),
                    title="absolute of bin %s order %s" % (b, 0),
                    desc="Absolute value of Fourier component",
                )
            )
            for o in range(1, orders):
                sets.append(
                    AnalysisResult(
                        raw_data=absolute[b, o],
                        visualized=visualize_simple(
                            absolute[b, o] / normal[b], vmin=min_absolute, vmax=max_absolute
                        ),
                        key="absolute_%s_%s" % (b, o),
                        title="absolute of bin %s order %s" % (b, o),
                        desc="Absolute value of Fourier component",
                    )
                )
        for b in range(self.parameters['n_bins']):
            for o in range(orders):
                sets.append(
                    AnalysisResult(
                        raw_data=angle[b, o],
                        visualized=visualize_simple(
                            angle[b, o], colormap=cmaps['perception_circular']
                        ),
                        key="phase_%s_%s" % (b, o),
                        title="phase of bin %s order %s" % (b, o),
                        desc="Phase of Fourier component",
                    )
                )
        for b in range(self.parameters['n_bins']):
            data = job_results[b, 0]
            f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((data.imag, data.real))
            sets.append(
                AnalysisResult(
                    raw_data=data,
                    visualized=f,
                    key="complex_%s_%s" % (b, 0),
                    title="bin %s order %s" % (b, 0),
                    desc="Fourier component",
                )
            )
            for o in range(1, orders):
                data = job_results[b, o] / normal[b]
                f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((data.imag, data.real), vmax=max_absolute)
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
