import logging

import numpy as np

from libertem.udf import UDF


log = logging.getLogger(__name__)


class PickUDF(UDF):
    '''
    Load raw data from ROI

    This UDF is meant for frame picking with a very small ROI, usually a single frame.

    .. versionadded:: 0.4.0
    '''
    def __init__(self):
        super().__init__()

    def get_preferred_input_dtype(self):
        ''
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        ''
        dtype = self.meta.input_dtype
        sigshape = tuple(self.meta.dataset_shape.sig)
        if self.meta.roi is not None:
            navsize = np.count_nonzero(self.meta.roi)
        else:
            navsize = np.prod(self.meta.dataset_shape.nav, dtype=np.int64)
        warn_limit = 2**28
        loaded_size = np.prod(sigshape, dtype=np.int64) * navsize * np.dtype(dtype).itemsize
        if loaded_size > warn_limit:
            log.warning("PickUDF is loading %s bytes, exceeding warning limit %s. "
                "Consider using or implementing an UDF to process data on the worker "
                "nodes instead." % (loaded_size, warn_limit))
        # We are using a "single" buffer since we mostly load single frames. A
        # "sig" buffer would work as well, but would require a transpose to
        # accomodate multiple frames in the last and not first dimension.
        # A "nav" buffer would allocate a NaN-filled buffer for the whole dataset.
        return {
            'intensity': self.buffer(
                kind='single', extra_shape=(navsize, ) + sigshape, dtype=dtype
            )
        }

    def process_tile(self, tile):
        ''
        # We work in flattened nav space with ROI applied
        sl = self.meta.slice.get()
        self.results.intensity[sl] = tile

    def merge(self, dest, src):
        ''
        # We receive full-size buffers from each node that
        # contributes at least one frame and rely on the rest being filled
        # with zeros correctly.
        dest.intensity[:] += src.intensity
