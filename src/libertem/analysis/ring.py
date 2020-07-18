import numpy as np
import inspect

from libertem import masks
from .masks import SingleMaskAnalysis
from .helper import GeneratorHelper


class RingTemplate(GeneratorHelper):

    short_name = "ring"
    api = "create_ring_analysis"

    def __init__(self, params):
        self.params = params

    def get_docs(self):
        title = "Ring Analysis"
        from libertem.api import Context
        docs_rst = inspect.getdoc(Context.create_ring_analysis)
        docs = self.format_docs(title, docs_rst)
        return docs

    def convert_params(self):
        params = ['dataset=ds']
        for k in ['cx', 'cy', 'ri', 'ro']:
            params.append(f'{k}={self.params[k]}')
        return ', '.join(params)

    def get_plot(self):
        plot = ["plt.figure()",
                "plt.imshow(ring_result['intensity'])",
                "plt.colorbar()"]
        return ['\n'.join(plot)]


class RingMaskAnalysis(SingleMaskAnalysis, id_="APPLY_RING_MASK"):
    TYPE = 'UDF'

    def get_description(self):
        return "intensity of the integration over the selected ring"

    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters['cx']
        cy = self.parameters['cy']
        ri = self.parameters['ri']
        ro = self.parameters['ro']

        def _ring_inner():
            return masks.ring(
                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                radius=ro,
                radius_inner=ri)

        return [_ring_inner]

    def get_parameters(self, parameters):
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        ro = parameters.get('ro', min(detector_y, detector_x) / 2)
        ri = parameters.get('ri', ro * 0.8)
        use_sparse = parameters.get('use_sparse', False)

        return {
            'cx': cx,
            'cy': cy,
            'ri': ri,
            'ro': ro,
            'use_sparse': use_sparse,
            'mask_count': 1,
            'mask_dtype': np.float32,
        }

    @classmethod
    def get_template_helper(cls):
        return RingTemplate
