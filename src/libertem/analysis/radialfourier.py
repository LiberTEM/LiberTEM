import logging
import inspect
from functools import partial

import numpy as np
import sparse
import numba

from libertem import masks
from libertem.common.math import prod
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis
from .helper import GeneratorHelper

log = logging.getLogger(__name__)


class RadialTemplate(GeneratorHelper):

    short_name = "radial"
    api = "create_radial_fourier_analysis"
    temp = GeneratorHelper.temp_analysis
    temp_analysis = temp + ["print(radial_result)"]

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return [
            "import matplotlib.cm as cm",
            "from libertem.viz import rgb_from_2dvector, libertem_cyclic"
        ]

    def get_docs(self):
        title = "Radial Fourier Analysis"
        from libertem.api import Context
        docs_rst = inspect.getdoc(Context.create_radial_fourier_analysis)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        params = ['dataset=ds']
        for k in ['cx', 'cy', 'ri', 'ro', 'n_bins', 'max_order']:
            params.append(f'{k}={self.params[k]}')
        return ', '.join(params)

    def get_plot(self):
        cells = []
        cells.append([
            "fig, axes = plt.subplots()",
            'axes.set_title("dominant_0")',
            "plt.imshow(radial_result.dominant_0, cmap=cm.tab20, vmin=0, vmax=20)",
            "fig, axes = plt.subplots()",
            'axes.set_title("absolute_0_0")',
            "axes.imshow(radial_result.absolute_0_0)",
        ])
        cells.append([
            "imag = radial_result.complex_0_1.raw_data.imag",
            "real = radial_result.complex_0_1.raw_data.real",
            "fig, axes = plt.subplots()",
            'axes.set_title("complex_0_1")',
            "plt.imshow(rgb_from_2dvector(x=real, y=imag))",
            "fig, axes = plt.subplots()",
            'axes.set_title("phase_0_1")',
            'plt.imshow(radial_result.phase_0_1.raw_data, cmap=libertem_cyclic)'
        ])
        return ['\n'.join(cell) for cell in cells]

    def get_save(self):
        save = []
        channels = ["absolute_0_0", "absolute_0_1"]
        for channel in channels:
            result = f"radial_result['{channel}'].raw_data"
            save.append(f"np.save('radial_result_{channel}.npy', {result})")
        return '\n'.join(save)


class RadialFourierResultSet(AnalysisResultSet):
    """
    Result set of a :class:`RadialFourierAnalysis`

    Running a :class:`RadialFourierAnalysis` via :meth:`libertem.api.Context.run` on a dataset
    returns an instance of this class. It contains the Fourier coefficients
    for each bin. See :meth:`libertem.api.Context.create_radial_fourier_analysis` for
    available parameters and :ref:`radialfourier app` for a description of the application!

    .. versionadded:: 0.3.0

    Attributes
    ----------
    dominant_0, absolute_0_0, absolute_0_1, ..., absolute_0_<max_order>,\
    phase_0_0, ..., phase_0_<max_order>,\
    complex_0_0, ..., complex_0_<max_order>;\
    dominant_1, absolute_1_0, ..., complex_1_<max_order>;\
    dominant_<nbins-1>, ..., complex_<nbins-1>_<max_order> : libertem.analysis.base.AnalysisResult
        Results for each bin: dominant Fourier coefficient (indicates symmetry),
        absolute values of each Fourier coefficient,
        phase values of each Fourier coefficient, complex values of each Fourier coefficient.
        The results have the shape of the navigation dimension.
    raw_results : numpy.ndarray
        Complex numbers, shape (<n_bins>, <max_order + 1>, \\*(<ds.shape.nav>))
    """
    pass


def radial_mask_factory(
        detector_y, detector_x, cx, cy, ri, ro, n_bins, max_order,
        use_sparse, dtype=np.complex64):
    dtype = np.result_type(dtype, np.complex64)

    def stack():
        rings = masks.radial_bins(
            centerX=cx,
            centerY=cy,
            imageSizeX=detector_x,
            imageSizeY=detector_y,
            radius=ro,
            radius_inner=ri,
            n_bins=n_bins,
            use_sparse=use_sparse,
            dtype=dtype
        )

        orders = np.arange(max_order + 1, dtype=dtype)

        r, phi = masks.polar_map(
            centerX=cx,
            centerY=cy,
            imageSizeX=detector_x,
            imageSizeY=detector_y
        )
        modulator = np.exp(phi.astype(dtype) * orders[:, np.newaxis, np.newaxis] * 1j)

        if use_sparse:
            rings = rings.reshape((rings.shape[0], 1, *rings.shape[1:]))
            ring_stack = [rings] * len(orders)
            ring_stack = sparse.concatenate(ring_stack, axis=1)
            _radial_mask_product(
                ring_stack_coords=ring_stack.coords,
                ring_stack_data_inout=ring_stack.data,
                modulator=modulator
            )
        else:
            ring_stack = rings[:, np.newaxis, ...] * modulator
        return ring_stack.reshape((-1, detector_y, detector_x))
    return stack


@numba.njit(cache=True, nogil=True)
def _radial_mask_product(ring_stack_coords, ring_stack_data_inout, modulator):
    '''
    Perform the product between rings and modulator for radial_mask_factory()

    Work on the COO data structure since `sparse` doesn't support
    efficient sparse-dense product with broadcasting since 0.16, apparently.
    '''
    for i in range(ring_stack_data_inout.shape[0]):
        order = ring_stack_coords[1, i]
        y = ring_stack_coords[2, i]
        x = ring_stack_coords[3, i]
        ring_stack_data_inout[i] *= modulator[order, y, x]


class RadialFourierAnalysis(BaseMasksAnalysis, id_="RADIAL_FOURIER"):
    '''
    The Radial Fourier Analysis can be used to characterize
    atomic ordering in materials, in particular for low intensities where
    Fluctualtion EM :cite:`Gibson1997` has a hard time to distinguish speckle
    from shot noise. Reference :cite:`6980942` describes a previous application
    of this method to characterize features in images.

    This analysis doesn't use fast Fourier transforms, but calculates the
    Fourier coefficients using sparse matrices in a dot product following the
    `definition of Fourier series
    <https://en.wikipedia.org/wiki/Fourier_series#Complex-valued_functions>`_.

    See :meth:`libertem.api.Context.create_radial_fourier_analysis` for
    available parameters and :ref:`radialfourier app` for a description of the
    application!
    '''

    TYPE = 'UDF'

    def get_udf_results(self, udf_results, roi, damage):
        '''
        The AnalysisResults are calculated lazily in this function to reduce
        overhead.
        '''
        shape = tuple(self.dataset.shape.nav)
        # NOTE: transposed for historical reasons
        udf_results = udf_results['intensity'].data.reshape((prod(shape), -1)).T
        orders = self.parameters['max_order'] + 1
        n_bins = self.parameters['n_bins']
        udf_results = udf_results.reshape((n_bins, orders, *shape))

        def resultlist():
            from libertem.viz import rgb_from_2dvector, visualize_simple, libertem_cyclic
            import matplotlib.cm as cm
            sets = []
            absolute = np.absolute(udf_results)
            normal = np.maximum(1, absolute[:, 0])
            # New local variable since this is a closure over damage
            dam = damage & np.all(np.isfinite(absolute), axis=(0, 1))
            normalized = absolute[:, 1:, ...] / normal[:, np.newaxis, ...]
            min_absolute = np.min(normalized[..., dam])
            max_absolute = np.max(normalized[..., dam])
            angle = np.angle(udf_results)
            threshold_map = absolute[:, 1:, ...].reshape((n_bins, -1)).max(axis=1) * 0.2
            below_threshold = np.all(
                absolute[:, 1:, ...] < threshold_map[:, np.newaxis, np.newaxis, np.newaxis],
                axis=1
            )
            dominant = np.argmax(absolute[:, 1:], axis=1) + 1
            dominant[below_threshold] = 0
            for b in range(n_bins):
                sets.append(
                    AnalysisResult(
                        raw_data=dominant[b],
                        visualized=partial(
                            visualize_simple, dominant[b], colormap=cm.tab20, vmin=0, vmax=20
                        ),
                        key="dominant_%s" % b,
                        title="dominant order of bin %s" % b,
                        desc="Dominant Fourier component",
                    )
                )
                sets.append(
                    AnalysisResult(
                        raw_data=absolute[b, 0],
                        visualized=partial(visualize_simple, absolute[b, 0], damage=dam),
                        key=f"absolute_{b}_{0}",
                        title=f"absolute of bin {b} order {0}",
                        desc="Absolute value of Fourier component",
                    )
                )
                for o in range(1, orders):
                    sets.append(
                        AnalysisResult(
                            raw_data=absolute[b, o],
                            visualized=partial(visualize_simple,
                                absolute[b, o] / normal[b], vmin=min_absolute, vmax=max_absolute,
                                damage=dam
                            ),
                            key=f"absolute_{b}_{o}",
                            title=f"absolute of bin {b} order {o}",
                            desc="Absolute value of Fourier component",
                        )
                    )
            for b in range(n_bins):
                for o in range(orders):
                    sets.append(
                        AnalysisResult(
                            raw_data=angle[b, o],
                            visualized=partial(visualize_simple,
                                angle[b, o], colormap=libertem_cyclic,
                                damage=dam
                            ),
                            key=f"phase_{b}_{o}",
                            title=f"phase of bin {b} order {o}",
                            desc="Phase of Fourier component",
                        )
                    )
            for b in range(n_bins):
                data = udf_results[b, 0]
                f = partial(
                    rgb_from_2dvector,
                    x=data.real, y=data.imag,
                    vmax=np.max(np.abs(data[..., dam]))
                )
                sets.append(
                    AnalysisResult(
                        raw_data=data,
                        visualized=f,
                        key=f"complex_{b}_{0}",
                        title=f"bin {b} order {0}",
                        desc="Fourier component",
                    )
                )
                for o in range(1, orders):
                    data = udf_results[b, o] / normal[b]
                    f = partial(
                        rgb_from_2dvector,
                        x=data.real, y=data.imag, vmax=max_absolute
                    )
                    sets.append(
                        AnalysisResult(
                            raw_data=data,
                            visualized=f,
                            key=f"complex_{b}_{o}",
                            title=f"bin {b} order {o}",
                            desc="Fourier component",
                        )
                    )
            return sets
        return RadialFourierResultSet(resultlist, raw_results=udf_results)

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")

        (detector_y, detector_x) = self.dataset.shape.sig
        p = self.parameters

        return radial_mask_factory(
            detector_y=detector_y,
            detector_x=detector_x,
            cx=p['cx'],
            cy=p['cy'],
            ri=p['ri'],
            ro=p['ro'],
            n_bins=p['n_bins'],
            max_order=p['max_order'],
            use_sparse=p['use_sparse'],
        )

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        ri = parameters.get('ri', 0)
        ro = parameters.get(
            'ro',
            masks.bounding_radius(cx, cy, detector_x, detector_y)
        )
        n_bins = parameters.get('n_bins', 1)
        max_order = parameters.get('max_order', 24)

        mask_count = n_bins * (max_order + 1)
        bin_width = (ro - ri) / n_bins
        bin_area = np.pi * ro**2 - np.pi * (ro - bin_width)**2
        stack_size = mask_count * detector_y * detector_x * 8

        default = 'scipy.sparse'
        # If the mask stack comfortably fits the L3 cache
        # FIXME more testing for optimum backend
        if stack_size < 2**18:
            default = False
        # Masks are actually dense
        elif bin_area / (detector_x * detector_y) > 0.05 and n_bins < 10:
            default = False

        use_sparse = parameters.get('use_sparse', default)
        return {
            'cx': cx,
            'cy': cy,
            'ri': ri,
            'ro': ro,
            'n_bins': n_bins,
            'max_order': max_order,
            'use_sparse': use_sparse,
            'mask_count': mask_count,
            'mask_dtype': np.complex64,
        }

    @classmethod
    def get_template_helper(cls):
        return RadialTemplate
