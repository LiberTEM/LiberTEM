import logging
import inspect
from typing import Dict, NamedTuple, Optional, Tuple, Type, Union, TYPE_CHECKING

import numpy as np

from libertem import masks
from libertem.web.rpc import ProcedureProtocol
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis
from libertem.corrections import coordinates
from .helper import GeneratorHelper

if TYPE_CHECKING:
    from libertem.web.rpc import RPCContext

log = logging.getLogger(__name__)


class ComTemplate(GeneratorHelper):
    short_name = "com"
    api = "create_com_analysis"
    temp = GeneratorHelper.temp_analysis
    temp_analysis = temp + ["print(com_result)"]
    channels = [
        "field",
        "magnitude",
        "divergence",
        "curl",
        "x",
        "y"
    ]

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return [
            "from empyre.vis.colors import ColormapCubehelix"
        ]

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
        if self.params.get('flip_y', False):
            params.append("flip_y=True")
        if self.params.get('scan_rotation') is not None:
            params.append(f"scan_rotation={self.params['scan_rotation']}")
        if self.params.get('ri') is not None:
            params.append(f"mask_radius_inner={self.params['ri']}")
        return ', '.join(params)

    def get_plot(self):
        plot = [
            "fig, axes = plt.subplots()",
            'axes.set_title("field")',
            "x_centers, y_centers = com_result.field.raw_data",
            "ch = ColormapCubehelix(start=1, rot=1, minLight=0.5, maxLight=0.5, sat=2)",
            "axes.imshow(ch.rgb_from_vector((x_centers, y_centers, 0)))"
        ]
        for channel in self.channels[1:3]:
            plot.append("fig, axes = plt.subplots()")
            plot.append(f'axes.set_title("{channel}")')
            plot.append(f'axes.imshow(com_result.{channel}.raw_data)')

        return ['\n'.join(plot)]

    def get_save(self):
        save = []
        for channel in self.channels:
            save.append(f"np.save('com_result_{channel}.npy', com_result['{channel}'].raw_data)")

        return '\n'.join(save)


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


def com_masks_generic(detector_y, detector_x, base_mask_factory):
    """
    Create a CoM mask stack with a generic selection mask factory

    Parameters
    ----------
    detector_y : int
        The detector height
    detector_x : int
        The detector width
    base_mask_factory : () -> np.array
        A factory function for creating the selection mask

    Returns
    -------
    List[Function]
        The mask stack as a list of factory functions
    """
    return [
        base_mask_factory,
        lambda: masks.gradient_y(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * base_mask_factory(),
        lambda: masks.gradient_x(
            imageSizeX=detector_x,
            imageSizeY=detector_y,
        ) * base_mask_factory(),
    ]


def center_shifts(img_sum, img_y, img_x, ref_y, ref_x):
    x_centers = np.divide(img_x, img_sum, where=img_sum != 0)
    y_centers = np.divide(img_y, img_sum, where=img_sum != 0)
    x_centers[img_sum == 0] = ref_x
    y_centers[img_sum == 0] = ref_y
    x_centers -= ref_x
    y_centers -= ref_y
    return (y_centers, x_centers)


def apply_correction(y_centers, x_centers, scan_rotation, flip_y, forward=True):
    shape = y_centers.shape
    if flip_y:
        transform = coordinates.flip_y()
    else:
        transform = coordinates.identity()
    # Transformations are applied right to left
    transform = coordinates.rotate_deg(scan_rotation) @ transform
    y_centers = y_centers.reshape(-1)
    x_centers = x_centers.reshape(-1)
    if not forward:
        transform = np.linalg.inv(transform)
    y_transformed, x_transformed = transform @ (y_centers, x_centers)
    y_transformed = y_transformed.reshape(shape)
    x_transformed = x_transformed.reshape(shape)
    return (y_transformed, x_transformed)


def divergence(y_centers, x_centers):
    return np.gradient(y_centers, axis=0) + np.gradient(x_centers, axis=1)


def curl_2d(y_centers, x_centers):
    # https://en.wikipedia.org/wiki/Curl_(mathematics)#Usage
    # DFy/dx - dFx/dy
    # axis 0 is y, axis 1 is x
    return np.gradient(y_centers, axis=1) - np.gradient(x_centers, axis=0)


def magnitude(y_centers, x_centers):
    return np.sqrt(y_centers**2 + x_centers**2)


def coordinate_check(y_centers, x_centers, roi=None):
    '''
    Calculate the RMS curl as a function of :code:`scan_rotation` and :code:`flip_y`.

    The curl for a purely electrostatic field is zero. That means
    the correct settings for :code:`scan_rotation` and :code:`flip_y` should
    minimize the RMS curl for atomic resolution STEM of non-magnetic specimens
    along a high-symmetry zone axis.

    Parameters
    ----------
    y_centers, x_centers : numpy.ndarray
        2D arrays with y and x component of the center of mass shift for each
        scan position, as returned by :meth:`center_shifts` or
        :meth:`apply_correction`
    roi : Optional[numpy.ndarray]
        Selector for values to consider in the statistics, compatible
        with indexing an array with the shape of y_centers and x_centers.
        By default, everything except the last row and last column are used
        since these contain artefacts.

    Returns
    -------
    (straight, flipped)
        Root mean square of the curl as a function of :code:`scan_rotation` from
        0 to 359 degrees in steps of one, with :code:`flip_y=False` (straight)
        and code:`flip_y=True` (flipped).
    '''
    straight = np.zeros(360)
    flipped = np.zeros(360)
    if roi is None:
        # The last row and column contain artifacts
        roi = (slice(0, -1), slice(0, -1))
    for angle in range(360):
        for flip_y in (True, False):
            y_transformed, x_transformed = apply_correction(
                y_centers, x_centers, scan_rotation=angle, flip_y=flip_y
            )
            curl = curl_2d(y_transformed, x_transformed)
            result = np.sqrt(np.mean(curl[roi]**2))
            if flip_y:
                flipped[angle] = result
            else:
                straight[angle] = result
    return (straight, flipped)


class GuessResult(NamedTuple):
    scan_rotation: int
    flip_y: bool
    cy: float
    cx: float


def guess_corrections(
    y_centers: np.ndarray,
    x_centers: np.ndarray,
    roi: Optional[Union[np.ndarray, Tuple[slice, ...]]] = None,
) -> GuessResult:
    '''
    Guess corrections for center shift, :code:`scan_rotation` and :code:`flip_y` from CoM data

    This function can generate a CoM parameter guess for atomic resolution 4D STEM data
    by using the following assumptions:

    * The field is purely electrostatic, i.e. the RMS curl should be minimized
    * There is no net field over the field of view and no descan error,
      i.e. the mean deflection is zero.
    * Atomic resolution STEM means that the divergence will be negative at atom columns
      and consequently the histogram of divergence will have a stronger tail towards
      negative values than towards positive values.

    If any corrections were applied when generating the input data, please note that the corrections
    should be applied relative to these previous value. In particular, the
    center corrections returned by this function have to be back-transformed to the uncorrected
    coordinate system, for example with :code:`apply_corrections(..., forward=False)

    Parameters
    ----------
    y_centers, x_centers : numpy.ndarray
        2D arrays with y and x component of the center of mass shift for each
        scan position, as returned by :meth:`center_shifts` or
        :meth:`apply_correction`
    roi : Optional[numpy.ndarray]
        Selector for values to consider in the statistics, compatible
        with indexing an array with the shape of y_centers and x_centers.
        By default, everything except the last row and last column are used
        since these contain artefacts.


    Returns
    -------
    GuessResult : relative to current values
    '''
    if roi is None:
        # The last row and column contain artefacts
        roi = (slice(0, -1), slice(0, -1))
    straight, flipped = coordinate_check(y_centers, x_centers, roi=roi)
    # The one with lower minima is the correct one
    flip_y = bool(np.min(flipped) < np.min(straight))
    if flip_y:
        angle = np.argmin(flipped)
    else:
        angle = np.argmin(straight)

    corrected_y, corrected_x = apply_correction(
        y_centers, x_centers, scan_rotation=angle, flip_y=flip_y
    )
    # There are two equivalent angles that minimize RMS curl since a 180°
    # rotation inverts the coordinates and just flips the sign for divergence
    # and curl. To distinguish the two, the distribution of the divergence is
    # analyzed. With negative electrons and positive nuclei, the beam is
    # deflected towards the nuclei and the divergence is negative there. Since
    # the beam is deflected most strongly near the nuclei, the histogram should
    # have more values at the negative end of the range than at the positive
    # end.
    div = divergence(corrected_y, corrected_x)[roi]
    all_range = np.maximum(-np.min(div), np.max(div))
    hist, bins = np.histogram(div, range=(-all_range, all_range), bins=5)
    polarity_off = np.sum(hist[:1]) < np.sum(hist[-1:])
    if polarity_off:
        angle += 180
    if angle > 180:
        angle -= 360
    return GuessResult(
        scan_rotation=int(angle),
        flip_y=flip_y,
        cy=np.mean(y_centers[roi]),
        cx=np.mean(x_centers[roi])
    )


class COMResultSet(AnalysisResultSet):
    """
    Running a :class:`COMAnalysis` via :meth:`libertem.api.Context.run` on a dataset
    returns an instance of this class.

    This analysis is usually applied to datasets with real values. If the dataset contains
    complex numbers, this result contains the keys :attr:`x_real`, :attr:`y_real`,
    :attr:`x_imag`, :attr:`y_imag` instead of the vector field.

    By default, the shift is given in pixel coordinates, i.e. positive x shift goes to the right
    and positive y shift goes to the bottom. See also :ref:`concepts`.

    .. versionchanged:: 0.6.0
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
        Curl of the center of mass 2D vector field at a given point.

        .. versionadded:: 0.6.0

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


class ParameterGuessProc:
    def __call__(self, rpc_context: "RPCContext") -> Dict:
        comp_ana = rpc_context.get_compound_analysis()
        analyses = comp_ana["details"]["analyses"]
        if len(analyses) != 2:
            return {
                "status": "error",
                "message": "no analyses found, there should be 2",
            }
        analysis_details = [
            rpc_context.get_analysis_details(a)
            for a in analyses
        ]
        try:
            com_analysis = [
                a
                for a in analysis_details
                if a["details"]["analysisType"] == "CENTER_OF_MASS"
            ][0]
        except IndexError:
            return {
                "status": "error",
                "message": "no CoM analysis found",
            }
        com_analysis_id = com_analysis["analysis"]
        if not rpc_context.have_analysis_results(com_analysis_id):
            # run with the current analysis parameters as set in the GUI:
            rpc_context.run_analysis(com_analysis_id)
        result_info = rpc_context.get_analysis_results(com_analysis_id)
        res = result_info.results
        old_params = result_info.details["parameters"]
        guess = guess_corrections(res.y.raw_data, res.x.raw_data)
        # NOTE: convert guess results to absolute values to make sure we don't
        # run into any nasty synchronization issues, for example, if state goes
        # stale after the guess button was clicked.
        flip_y = bool(old_params["flip_y"]) != bool(guess.flip_y)
        return {
            'status': 'ok',
            'guess': {
                'cx': guess.cx + old_params["cx"],
                'cy': guess.cy + old_params["cy"],
                'scan_rotation': guess.scan_rotation + old_params["scan_rotation"],
                'flip_y': flip_y,
            },
        }


class COMAnalysis(BaseMasksAnalysis, id_="CENTER_OF_MASS"):
    TYPE = 'UDF'

    def get_udf_results(self, udf_results, roi, damage):
        data = udf_results['intensity'].data
        img_sum, img_y, img_x = (
            data[..., 0],
            data[..., 1],
            data[..., 2],
        )
        return self.get_generic_results(img_sum, img_y, img_x, damage=damage)

    def get_generic_results(self, img_sum, img_y, img_x, damage):
        from libertem.viz import CMAP_CIRCULAR_DEFAULT, visualize_simple
        ref_x = self.parameters["cx"]
        ref_y = self.parameters["cy"]
        y_centers_raw, x_centers_raw = center_shifts(img_sum, img_y, img_x, ref_y, ref_x)
        shape = y_centers_raw.shape
        y_centers, x_centers = apply_correction(
            y_centers_raw, x_centers_raw,
            scan_rotation=self.parameters["scan_rotation"],
            flip_y=self.parameters["flip_y"]
        )

        if img_sum.dtype.kind == 'c':
            x_real, x_imag = np.real(x_centers), np.imag(x_centers)
            y_real, y_imag = np.real(y_centers), np.imag(y_centers)

            return COMResultSet([
                AnalysisResult(raw_data=x_real, visualized=visualize_simple(x_real, damage=damage),
                       key="x_real", title="x [real]", desc="x component of the center"),
                AnalysisResult(raw_data=y_real, visualized=visualize_simple(y_real, damage=damage),
                       key="y_real", title="y [real]", desc="y component of the center"),
                AnalysisResult(raw_data=x_imag, visualized=visualize_simple(x_imag, damage=damage),
                       key="x_imag", title="x [imag]", desc="x component of the center"),
                AnalysisResult(raw_data=y_imag, visualized=visualize_simple(y_imag, damage=damage),
                       key="y_imag", title="y [imag]", desc="y component of the center"),
            ])
        else:
            damage = damage & np.isfinite(x_centers) & np.isfinite(y_centers)
            # Make sure that an all-False `damage` is handled since np.max()
            # trips on an empty array.
            # As a remark -- the NumPy error message
            # "zero-size array to reduction operation maximum which has no identity"
            # is probably wrong since -np.inf is the identity element for maximum on
            # floating point numbers and should be returned here.
            if np.count_nonzero(damage) > 0:
                vmax = np.sqrt(np.max(x_centers[damage]**2 + y_centers[damage]**2))
            else:
                vmax = 1
            f = CMAP_CIRCULAR_DEFAULT.rgb_from_vector((x_centers, y_centers, 0), vmax=vmax)
            m = magnitude(y_centers, x_centers)
            # Create results which are valid for any nav_shape
            results_list = [
                AnalysisResult(
                    raw_data=(x_centers, y_centers),
                    visualized=f,
                    key="field", title="field", desc="cubehelix colorwheel visualization",
                    include_in_download=False
                ),
                AnalysisResult(
                    raw_data=m,
                    visualized=visualize_simple(m, damage=damage),
                    key="magnitude", title="magnitude", desc="magnitude of the vector field"
                ),
                AnalysisResult(
                    raw_data=x_centers,
                    visualized=visualize_simple(x_centers, damage=damage),
                    key="x", title="x", desc="x component of the center"
                ),
                AnalysisResult(
                    raw_data=y_centers,
                    visualized=visualize_simple(y_centers, damage=damage),
                    key="y", title="y", desc="y component of the center"
                ),
            ]
            # Add results which depend on np.gradient, i.e. all(nav_shape) > 1
            if all([s > 1 for s in shape]):
                d = divergence(y_centers, x_centers)
                c = curl_2d(y_centers, x_centers)
                extra_results = [
                    AnalysisResult(
                        raw_data=d,
                        visualized=visualize_simple(d, damage=damage),
                        key="divergence", title="divergence", desc="divergence of the vector field"
                    ),
                    AnalysisResult(
                        raw_data=c,
                        visualized=visualize_simple(c, damage=damage),
                        key="curl", title="curl", desc="curl of the 2D vector field"
                    ),
                ]
                # Insert the results at position 2 for backwards compatibility/tests
                # This could later be replaced with results_list.extend(extra_results)
                results_list[2:2] = extra_results

        return COMResultSet(results_list)

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        if self.parameters.get('ri'):
            # annular CoM:
            return com_masks_generic(
                detector_y=self.dataset.shape.sig[0],
                detector_x=self.dataset.shape.sig[1],
                base_mask_factory=lambda: masks.ring(
                    imageSizeY=self.dataset.shape.sig[0],
                    imageSizeX=self.dataset.shape.sig[1],
                    centerY=self.parameters['cy'],
                    centerX=self.parameters['cx'],
                    radius=self.parameters['r'],
                    radius_inner=self.parameters['ri'],
                )
            )
        else:
            # CoM with radius cut-off:
            return com_masks_factory(
                detector_y=self.dataset.shape.sig[0],
                detector_x=self.dataset.shape.sig[1],
                cx=self.parameters['cx'],
                cy=self.parameters['cy'],
                r=self.parameters['r'],
            )

    def get_parameters(self, parameters: Dict) -> Dict:
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        r = parameters.get('r', float('inf'))
        ri = parameters.get('ri', 0.0)
        scan_rotation = parameters.get('scan_rotation', 0.)
        flip_y = parameters.get('flip_y', False)
        use_sparse = parameters.get('use_sparse', False)

        return {
            'cx': cx,
            'cy': cy,
            'r': r,
            'ri': ri,
            'scan_rotation': scan_rotation,
            'flip_y': flip_y,
            'use_sparse': use_sparse,
            'mask_count': 3,
            'mask_dtype': np.float32,
        }

    @classmethod
    def get_template_helper(cls) -> Type[GeneratorHelper]:
        return ComTemplate

    @classmethod
    def get_rpc_definitions(cls) -> Dict[str, Type[ProcedureProtocol]]:
        return {
            "guess_parameters": ParameterGuessProc,
        }

    def need_rerun(self, old_params: Dict, new_params: Dict) -> bool:
        """
        Don't need to re-run UDF if only `flip_y` or `scan_rotation`
        have changed.
        """
        ignore_keys = {"flip_y", "scan_rotation"}
        old_without_ignored = {
            k: v
            for k, v in old_params.items()
            if k not in ignore_keys
        }
        new_without_ignored = {
            k: v
            for k, v in new_params.items()
            if k not in ignore_keys
        }
        return old_without_ignored != new_without_ignored
