import logging
import inspect
from typing import TYPE_CHECKING

import numpy as np

from libertem import masks
from libertem.web.rpc import ProcedureProtocol
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis

from .helper import GeneratorHelper

# Keep imports for backwards compatibility!
from libertem.udf.com import (  # noqa: 401
    com_masks_factory, com_masks_generic, center_shifts, apply_correction,
    divergence, curl_2d, magnitude,
    coordinate_check, GuessResult, guess_corrections
)

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
            "from libertem.viz import rgb_from_2dvector"
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
            "axes.imshow(rgb_from_2dvector(x=x_centers, y=y_centers))"
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
    async def __call__(self, rpc_context: "RPCContext") -> dict:
        comp_ana = rpc_context.get_compound_analysis()
        analyses = comp_ana["details"]["analyses"]
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
            await rpc_context.run_analysis(com_analysis_id)
        result_info = rpc_context.get_analysis_results(com_analysis_id)
        res = result_info.results
        old_params = result_info.details["parameters"]
        guess = await rpc_context.run_sync(guess_corrections, res.y.raw_data, res.x.raw_data)
        # NOTE: convert guess results to absolute values to make sure we don't
        # run into any nasty synchronization issues, for example, if state goes
        # stale after the guess button was clicked.
        flip_y = bool(old_params["flip_y"]) != bool(guess.flip_y)
        backtransformed = apply_correction(
            y_centers=np.array((guess.cy, )),
            x_centers=np.array((guess.cx, )),
            scan_rotation=old_params["scan_rotation"],
            flip_y=old_params["flip_y"],
            forward=False,
        )
        return {
            'status': 'ok',
            'guess': {
                'cx': backtransformed[1][0] + old_params["cx"],
                'cy': backtransformed[0][0] + old_params["cy"],
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
        from libertem.viz import rgb_from_2dvector, visualize_simple
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
            f = rgb_from_2dvector(x=x_centers, y=y_centers, vmax=vmax)
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

    def get_parameters(self, parameters: dict) -> dict:
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
    def get_template_helper(cls) -> type[GeneratorHelper]:
        return ComTemplate

    @classmethod
    def get_rpc_definitions(cls) -> dict[str, type[ProcedureProtocol]]:
        return {
            "guess_parameters": ParameterGuessProc,
        }

    def need_rerun(self, old_params: dict, new_params: dict) -> bool:
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
