import logging

import numpy as np

from libertem.udf import UDF


log = logging.getLogger(__name__)


class AutoUDF(UDF):
    '''
    Generate a UDF for reduction along the signal axes from a regular function
    that processes one frame, generating result buffers automatically.

    Parameters
    ----------
    f: Callable
        Function that accepts a frame as a single parameter. It will be called
        with :code:`np.ones(tuple(dataset.shape.sig))` to determine the output type and
        shape.
    monitor: bool, optional
        .. versionadded:: 0.16.0

        If True, the UDF will include a
        :ref:`result-only<udf final post processing>` monitoring buffer with the last
        valid result for live plotting. Defaults to False.
    '''
    def __init__(self, f, monitor=False):
        super().__init__(f=f, monitor=monitor)

    def auto_buffer(self, var):
        return self.buffer(kind='nav', extra_shape=var.shape, dtype=var.dtype)

    def auto_monitor_buffer(self, var):
        return self.buffer(
            kind='single',
            extra_shape=var.shape, dtype=var.dtype,
            use='result_only'
        )

    def get_result_buffers(self):
        '''
        Auto-generate result buffers based on the return value of f() called with a mock frame.
        '''
        mock_frame = np.ones(tuple(self.meta.dataset_shape.sig), dtype=self.meta.input_dtype)
        result = np.array(self.params.f(mock_frame))
        buffers = {
            'result': self.auto_buffer(result)
        }
        if self.params.monitor:
            buffers['monitor'] = self.auto_monitor_buffer(result)
        return buffers

    def process_frame(self, frame):
        '''
        Call f() for the frame and assign return value to the result buffer slot.
        '''
        res = self.params.f(frame)
        self.results.result[:] = np.array(res)

    def get_results(self):
        if self.params.monitor:
            # valid nav mask is flat
            valid_nav_mask = self.meta.get_valid_nav_mask()
            valid_indices = np.nonzero(valid_nav_mask)
            if len(valid_indices):
                # shape (n_dim, n), with n_dim == 1, see above
                last_index = valid_indices[0][-1]
            else:
                # return initial value
                last_index = 0
            return {
                'monitor': self.results.result[last_index]
            }
        else:
            return {}
