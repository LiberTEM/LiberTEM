from libertem import masks
from libertem.viz import visualize_simple
from .base import AnalysisResult, AnalysisResultSet
from .masks import BaseMasksAnalysis

class SquareMaskAnalysis(BaseMasksAnalysis):
    def get_results(self,job_results):
        data=job_results[0]
        shape = tuple(self.dataset.shape.nav)

        if data.dtype.kind=='c':
            return AnalysisResultSet(
                self.get_complex_results(
                    data,
                    key_prefix='intensity',
                    title='intensity',
                    desc='intensity over the selected the selected square',
                )
            )
        return AnalysisResultSet([
            AnalysisResult(
                raw_data=data.reshape(shape),
                visualized=visualize_simple(data),
                key="intensity",
                title="intensity",
                desc="intensity of the integeration over the selected disk"),
        ])
    
    def get_mask_factories(self):
        if self.dataset.shape.sig.dims != 2:
            raise ValueError("can only handle 2D signals currently")
        (detector_y, detector_x) = self.dataset.shape.sig

        cx = self.parameters['cx']
        cy = self.parameters['cy']
        side= self.parameters['side']

        def square_mask():
            return mask.square(

                centerX=cx,
                centerY=cy,
                imageSizeX=detector_x,
                imageSizeY=detector_y,
                sideValue=side,
            )
        
        return [
            square_mask,
        ]

    def get_parameters(self,parameters):
        (detector_y,detector_x)=self.dataset.shape.sig

        cx = parameters.get('cx', detector_x / 2)
        cy = parameters.get('cy', detector_y / 2)
        side=parameters.get('side',min(detector_y,detector_x)/2*0.3)
        return{
            'cx':cx,
            'cy':cy,
            'side':side,
        }
