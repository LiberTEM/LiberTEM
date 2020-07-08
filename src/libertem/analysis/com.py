import logging
import inspect
import numpy as np

from libertem import masks
from libertem.viz import CMAP_CIRCULAR_DEFAULT, visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis
from libertem.corrections.coordinates import rotate_deg, flip_y, identity
from .helper import GeneratorHelper

log = logging.getLogger(__name__)


class ComTemplate(GeneratorHelper):

    short_name = "com"
    api = "create_com_analysis"
    temp = GeneratorHelper.temp_analysis
    temp_analysis = temp + ["com_result = com_analysis.get_udf_results(com_result, roi)"]
    temp_analysis.append("print(com_result)")

    def __init__(self, params):
        self.params = params

    def get_docs(self):
        title = "COM Analysis"
        from libertem.api import Context
        docs_rst = inspect.getdoc(Context.create_com_analysis)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        params = ['dataset=ds']
        for k in ['cx', 'cy']:
            params.append(f'{k}={self.params[k]}')
        params.append(f"mask_radius={self.params['r']}")
        return ', '.join(params)

    def get_plot(self):
        plot = []
        for channel in ["field", "magnitude", "curl"]:
            plot.append("fig, axes = plt.subplots()")
            plot.append(f'axes.set_title("{channel}")')
            plot.append(f'axes.imshow(com_result.{channel}.visualized)')

        return '\n'.join(plot)


def com_masks_factory(detector_y, detector_x, cy, cx, r):
    def disk_mask():
        return masks.circular(
            centerX=cx, centerY=cy,
            imageSizeX=detector_x,
            imageSizeY=detector_y,
            radius=r,
        )

    return [
        disk_mask,
        lambda: masks.gradient_y(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * disk_mask(),
        lambda: masks.gradient_x(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * disk_mask(),
    ]


def center_shifts(img_sum, img_y, img_x, ref_y, ref_x):
    x_centers = np.divide(img_x, img_sum, where=img_sum != 0)
    y_centers = np.divide(img_y, img_sum, where=img_sum != 0)
    x_centers[img_sum == 0] = ref_x
    y_centers[img_sum == 0] = ref_y
    x_centers -= ref_x
    y_centers -= ref_y
    return (y_centers, x_centers)


def divergence(y_centers, x_centers):
    return np.gradient(y_centers, axis=0) + np.gradient(x_centers, axis=1)


def curl_2d(y_centers, x_centers):
    # https://en.wikipedia.org/wiki/Curl_(mathematics)#Usage
    # DFy/dx - dFx/dy
    # axis 0 is y, axis 1 is x
    return np.gradient(y_centers, axis=1) - np.gradient(x_centers, axis=0)


def magnitude(y_centers, x_centers):
    return np.sqrt(y_centers**2 + x_centers**2)


class COMResultSet(AnalysisResultSet):
    """
    Running a :class:`COMAnalysis` via :meth:`libertem.api.Context.run` on a dataset
    returns an instance of this class.

    This analysis is usually applied to datasets with real values. If the dataset contains
    complex numbers, this result contains the keys :attr:`x_real`, :attr:`y_real`,
    :attr:`x_imag`, :attr:`y_imag` instead of the vector field.

    By default, the shift is given in pixel coordinates, i.e. positive x shift goes to the right
    and positive y shift goes to the bottom. See also :ref:`concepts`.

    .. versionchanged:: 0.6.0.dev0
        The COM analysis now supports flipping the y axis and rotating the vectors.

    .. versionadded:: 0.3.0

    Attributes
    ----------
    field : libertem.analysis.base.AnalysisResult
        Center of mass shift relative to the center given to the analysis within the given radius
        as a vector field with components (x, y). The visualized result uses a
        cubehelix color wheel.
    magnitude : libertem.analysis.base.AnalysisResult
        Magnitude of the center of mass shift.
    divergence : libertem.analysis.base.AnalysisResult
        Divergence of the center of mass vector field at a given point
    curl : libertem.analysis.base.AnalysisResult
        Curl of the center of mass 2D vector field at a given point. Added in 0.6.0.dev0
    x : libertem.analysis.base.AnalysisResult
        X component of the center of mass shift
    y : libertem.analysis.base.AnalysisResult
        Y component of the center of mass shift
    x_real : libertem.analysis.base.AnalysisResult
        Real part of the x component of the center of mass shift (complex dataset only)
    y_real : libertem.analysis.base.AnalysisResult
        Real part of y component of the center of mass shift (complex dataset only)
    x_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of the x component of the center of mass shift (complex dataset only)
    y_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of y component of the center of mass shift (complex dataset only)
    """
    pass


class COMAnalysis(BaseMasksAnalysis, id_="CENTER_OF_MASS"):
    TYPE = 'UDF'

    # FIXME remove this after UDF version is final
    def get_results(self, job_results):
        shape = tuple(self.dataset.shape.nav)
        img_sum, img_y, img_x = (
            job_results[0].reshape(shape),
            job_results[1].reshape(shape),
            job_results[2].reshape(shape)
        )
        return self.get_generic_results(img_sum, img_y, img_x)

    def get_udf_results(self, udf_results, roi):
        data = udf_results['intensity'].data
        img_sum, img_y, img_x = (
            data[..., 0],
            data[..., 1],
            data[..., 2],
        )
        return self.get_generic_results(img_sum, img_y, img_x)

    def get_generic_results(self, img_sum, img_y, img_x):
        ref_x = self.parameters["cx"]
        ref_y = self.parameters["cy"]
        y_centers_raw, x_centers_raw = center_shifts(img_sum, img_y, img_x, ref_y, ref_x)
        shape = y_centers_raw.shape
        if self.parameters["flip_y"]:
            transform = flip_y()
        else:
            transform = identity()
        # Transformations are applied right to left
        transform = rotate_deg(self.parameters["scan_rotation"]) @ transform
        y_centers, x_centers = transform @ (y_centers_raw.reshape(-1), x_centers_raw.reshape(-1))

        y_centers = y_centers.reshape(shape)
        x_centers = x_centers.reshape(shape)

        if img_sum.dtype.kind == 'c':
            x_real, x_imag = np.real(x_centers), np.imag(x_centers)
            y_real, y_imag = np.real(y_centers), np.imag(y_centers)

            return COMResultSet([
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
            d = divergence(y_centers, x_centers)
            c = curl_2d(y_centers, x_centers)
            m = magnitude(y_centers, x_centers)

            return COMResultSet([
                AnalysisResult(raw_data=(x_centers, y_centers), visualized=f,
                       key="field", title="field", desc="cubehelix colorwheel visualization",
                       include_in_download=False),
                AnalysisResult(raw_data=m, visualized=visualize_simple(m),
                       key="magnitude", title="magnitude", desc="magnitude of the vector field"),
                AnalysisResult(raw_data=d, visualized=visualize_simple(d),
                       key="divergence", title="divergence", desc="divergence of the vector field"),
                AnalysisResult(raw_data=c, visualized=visualize_simple(c),
                       key="curl", title="curl", desc="curl of the 2D vector field"),
                AnalysisResult(raw_data=x_centers, visualized=visualize_simple(x_centers),
                       key="x", title="x", desc="x component of the center"),
                AnalysisResult(raw_data=y_centers, visualized=visualize_simple(y_centers),
                       key="y", title="y", desc="y component of the center"),
            ])

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        return com_masks_factory(
            detector_y=self.dataset.shape.sig[0],
            detector_x=self.dataset.shape.sig[1],
            cx=self.parameters['cx'],
            cy=self.parameters['cy'],
            r=self.parameters['r'],
        )

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        r = parameters.get('r', float('inf'))
        scan_rotation = parameters.get('scan_rotation', 0.)
        flip_y = parameters.get('flip_y', False)
        use_sparse = parameters.get('use_sparse', False)

        return {
            'cx': cx,
            'cy': cy,
            'r': r,
            'scan_rotation': scan_rotation,
            'flip_y': flip_y,
            'use_sparse': use_sparse,
            'mask_count': 3,
            'mask_dtype': np.float32,
        }

    @classmethod
    def get_template_helper(cls):
        return ComTemplate
