import numpy as np

from libertem.udf import UDF


class AutoUDF(UDF):
    '''
    Generate a UDF for reduction along the signal axis from a regular function
    that processes one frame, generating result buffers automatically.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters:

        f:
            Function that accepts a frame as a single parameter. It will
            be called with np.ones(tuple(dataset.shape.sig)) to determine the output
            type and shape
        '''
        super().__init__(*args, **kwargs)

    def auto_buffer(self, var):
        return self.buffer(kind='nav', extra_shape=var.shape, dtype=var.dtype)

    def get_result_buffers(self):
        mock_frame = np.ones(tuple(self.meta.dataset_shape.sig), dtype=self.meta.dataset_dtype)
        result = self.params.f(mock_frame)

        return {
            'result': self.auto_buffer(result)
        }

    def process_frame(self, frame):
        res = self.params.f(frame)
        self.results.result[:] = res


def run_auto(ctx, dataset, f, roi=None):
    udf = AutoUDF(f=f)
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
