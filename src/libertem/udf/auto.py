import logging

import numpy as np

from libertem.udf import UDF


log = logging.getLogger(__name__)


class AutoUDF(UDF):
    '''
    Generate a UDF for reduction along the signal axis from a regular function
    that processes one frame, generating result buffers automatically.

    Parameters
    ----------

    f:
        Function that accepts a frame as a single parameter. It will
        be called with np.ones(tuple(dataset.shape.sig)) to determine the output
        type and shape.
    '''
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

    def auto_buffer(self, var):
        return self.buffer(kind='nav', extra_shape=var.shape, dtype=var.dtype)

    def get_result_buffers(self):
        '''
        Auto-generate result buffers based on the return value of f() called with a mock frame.
        '''
        mock_frame = np.ones(tuple(self.meta.dataset_shape.sig), dtype=self.meta.input_dtype)
        result = np.array(self.params.f(mock_frame))

        try:
            # FIXME Thresholds chosen somewhat arbitrarily
            if result.nbytes > max(1024, mock_frame.nbytes / (2**7)):
                log.warn(
                    "Return value of function has size %s, "
                    "not strongly reduced compared to input size %s"
                    % (result.nbytes, mock_frame.nbytes)
                )
        # Numpy arrays of dtype "object" can throw an AttributeError
        # upon size calculations
        except AttributeError:
            pass

        return {
            'result': self.auto_buffer(result)
        }

    def process_frame(self, frame):
        '''
        Call f() for the frame and assign return value to the result buffer slot.
        '''
        res = self.params.f(frame)
        self.results.result[:] = np.array(res)
