import inspect

import numpy as np
import sparse

from libertem.udf.raw import PickUDF
from .base import BaseAnalysis, AnalysisResult, AnalysisResultSet
from .helper import GeneratorHelper


class PickTemplate(GeneratorHelper):

    short_name = "pick"
    api = "create_pick_analysis"

    def __init__(self, params):
        self.params = params

    def get_dependency(self):
        return ["from matplotlib import colors"]

    def get_docs(self):
        title = "Pick Analysis"
        from libertem.api import Context
        docs_rst = inspect.getdoc(Context.create_pick_analysis)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        params = ['dataset=ds']
        for k in ['x', 'y']:
            params.append(f'{k}={self.params[k]}')
        return ', '.join(params)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(pick_result['intensity'], norm=colors.LogNorm())",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class PickResultSet(AnalysisResultSet):
    """
    Running a :class:`PickFrameAnalysis` via :meth:`libertem.api.Context.run`
    returns an instance of this class.

    If the dataset contains complex numbers, the regular result attribute carries the
    absolute value of the result and additional attributes with real part, imaginary part,
    phase and full complex result are available.

    .. versionadded:: 0.3.0

    .. versionchanged:: 0.4.0
        Picking now returns data in the native dtype of the dataset with the new UDF back-end.

    Attributes
    ----------
    intensity : libertem.analysis.base.AnalysisResult
        The specified detector frame. Absolute value if the dataset
        contains complex numbers. Log-scaled visualization.
    intensity_lin : libertem.analysis.base.AnalysisResult
        The specified detector frame. Absolute value if the dataset
        contains complex numbers. Linear visualization.

        .. versionadded:: 0.6.0

    intensity_real : libertem.analysis.base.AnalysisResult
        Real part of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_imag : libertem.analysis.base.AnalysisResult
        Imaginary part of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_angle : libertem.analysis.base.AnalysisResult
        Phase angle of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    intensity_complex : libertem.analysis.base.AnalysisResult
        Complex result of the specified detector frame. This is only available if the dataset
        contains complex numbers.
    """
    pass


class PickFrameAnalysis(BaseAnalysis, id_="PICK_FRAME"):
    TYPE = 'UDF'
    """
    Pick a single, complete frame from a dataset
    """

    def get_origin(self):
        dims = self.dataset.shape.nav.dims
        if dims not in (1, 2, 3):
            raise ValueError(
                "can only handle 1D/2D/3D nav currently, received %s dimensions" % dims
            )
        zyx = (
            self.parameters.get('z'),
            self.parameters.get('y'),
            self.parameters.get('x'),
        )
        messages = {
            1: "Need x, not y and not z to index 1D dataset, received z=%s, y=%s, x=%s",
            2: "Need x, y and not z to index 2D dataset, received z=%s, y=%s, x=%s",
            3: "Need x, y z to index 3D dataset, received z=%s, y=%s, x=%s",
        }
        keep = zyx[-dims:]
        drop = zyx[:-dims]

        if (None in keep) or not all(d is None for d in drop):
            raise ValueError(messages[dims] % zyx)

        return keep

    def get_udf(self):
        return PickUDF()

    def get_roi(self):
        coords = np.array(self.get_origin())[:, np.newaxis]
        roi = sparse.COO(coords=coords, data=True, fill_value=False, shape=self.dataset.shape.nav)
        return roi

    def get_udf_results(self, udf_results, roi, damage):
        return self.get_generic_results(udf_results['intensity'].data[0], damage=True)

    def get_coords(self):
        parameters = self.parameters
        coords = [
            "%s=%d" % (axis, parameters.get(axis))
            for axis in ['x', 'y', 'z']
            if axis in parameters
        ]
        return " ".join(coords)

    def get_generic_results(self, data, damage):
        from libertem.viz import visualize_simple
        coords = self.get_coords()

        if data.dtype.kind == 'c':
            return AnalysisResultSet(
                self.get_complex_results(
                    data,
                    key_prefix="intensity",
                    title="intensity",
                    desc=f"the frame at {coords}",
                    damage=True,
                    default_lin=False,
                )
            )
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data,
                visualized=visualize_simple(data, logarithmic=True, damage=True),
                key="intensity", title="intensity [log]",
                desc=f"the frame at {coords} log-scaled"
            ),
            AnalysisResult(
                raw_data=data,
                visualized=visualize_simple(data, logarithmic=False, damage=True),
                key="intensity_lin", title="intensity [lin]",
                desc=f"the frame at {coords} lin-scaled"
            ),
        ])

    @classmethod
    def get_template_helper(cls):
        return PickTemplate
