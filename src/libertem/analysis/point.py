import numpy as np
import sparse
import inspect

from .masks import SingleMaskAnalysis
from .helper import GeneratorHelper


class PointTemplate(GeneratorHelper):

    short_name = "point"
    api = "create_point_analysis"

    def __init__(self, params):
        self.params = params

    def get_docs(self):
        title = "Point Analysis"
        from libertem.api import Context
        docs_rst = inspect.getdoc(Context.create_point_analysis)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        params = ['dataset=ds']
        x = self.params['cx']
        y = self.params['cy']
        params.append(f'x={x}')
        params.append(f'y={y}')
        return ', '.join(params)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(point_result['intensity'])",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class PointMaskAnalysis(SingleMaskAnalysis, id_="APPLY_POINT_SELECTOR"):
    TYPE = 'UDF'

    def get_description(self):
        return "intensity of the integration over the selected point"

    def get_use_sparse(self):
        return True

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")

        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters['cx']
        cy = self.parameters['cy']

        sig_shape = self.dataset.shape.sig

        def _point_inner():
            a = sparse.COO(
                data=np.array([1]),
                coords=np.array(([int(cy)], [int(cx)])),
                shape=sig_shape
            )
            return a
        return [_point_inner]

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        return {
            'cx': cx,
            'cy': cy,
            'mask_count': 1,
            'mask_dtype': np.float32,
        }

    @classmethod
    def get_template_helper(cls):
        return PointTemplate
