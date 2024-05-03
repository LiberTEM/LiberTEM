import logging

import numpy as np

from libertem.common.math import prod, count_nonzero
from libertem.udf import UDF
import scipy.ndimage


log = logging.getLogger(__name__)


class PickUDF(UDF):
    '''
    Load raw data from ROI

    This UDF is meant for frame picking with a very small ROI, usually a single frame.

    .. versionadded:: 0.4.0

    Examples
    --------
    >>> udf = PickUDF()
    >>> roi = np.zeros(dataset.shape.nav, dtype=bool)
    >>> roi[0] = True
    >>> result = ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
    >>> result["intensity"].raw_data[0].shape
    (32, 32)
    '''
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_preferred_input_dtype(self):
        ''
        return self.USE_NATIVE_DTYPE

    def get_result_buffers(self):
        ''
        dtype = self.meta.input_dtype
        sigshape = tuple(self.meta.dataset_shape.sig)
        if self.meta.roi is not None:
            navsize = count_nonzero(self.meta.roi)
        else:
            navsize = prod(self.meta.dataset_shape.nav)
        warn_limit = 2**28
        loaded_size = prod(sigshape) * navsize * np.dtype(dtype).itemsize
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

    def merge_all(self, ordered_results):
        ''
        intensity_chunks = [b.intensity for b in ordered_results.values()]
        intensity_sum = np.stack(intensity_chunks, axis=0).sum(axis=0)
        return {'intensity': intensity_sum}


class PickShiftedUDF(UDF):
    '''
    Load raw data from ROI and applies a corrective shift.

    This UDF is useful to compensate for descan, separating the
    effect from material characteristics.

    Parameters
    ----------
    regression_coefficients : (3, 2) matrix
        A vector of regression coefficients for x- and y-directions.
        The regression coefficients make up two planes that describe
        the descan and are used to remove it.
        The coefficients are arranged in the following manner:
        ((y, x), (dy/dy, dx/dy), (dy/dx, dx/dx)) where y and x are
        the shifts at coordinates (0, 0). The displayed image will be
        shifted in x- and y-directions by these parameters. By default
        `None`, which corresponds to no shifts.
    '''
    def __init__(self, regression_coefficients=None):
        if regression_coefficients is None:
            regression_coefficients = np.zeros((3, 2))

        super().__init__(regression_coefficients=regression_coefficients)

    def process_tile(self, tile):
        ''
        # We work in flattened nav space with ROI applied
        sl = self.meta.slice.get()
        self.results.intensity[sl] = tile
        self.results.coordinates[:] = self.meta.coordinates

    def merge(self, dest, src):
        ''
        # We receive full-size buffers from each node that
        # contributes at least one frame and rely on the rest being filled
        # with zeros correctly.
        dest.intensity[:] += src.intensity
        dest.coordinates[:] = src.coordinates

    def merge_all(self, ordered_results):
        ''
        intensity_chunks = [b.intensity for b in ordered_results.values()]
        intensity_sum = np.stack(intensity_chunks, axis=0).sum(axis=0)
        coordinates = np.stack([b.coordinates for b in ordered_results.values()], axis=0)
        return {'intensity': intensity_sum, "coordinates": coordinates}

    def get_results(self):
        coordinates = self.results.get_buffer('coordinates').raw_data

        coordinates = np.concatenate((np.ones((*coordinates.shape[:-1], 1)), coordinates), axis=-1)
        shifts = np.dot(coordinates, self.params.regression_coefficients)
        intensity = self.results.get_buffer('intensity').data

        shifted = []
        for intens, shift in zip(intensity, shifts):
            shifted.append(scipy.ndimage.shift(intens, -shift, mode="constant"))

        return {
            "intensity": np.stack(shifted),
            "coordinates": self.results.get_buffer('coordinates').raw_data
        }

    def get_result_buffers(self):
        ''
        dtype = self.meta.input_dtype
        sigshape = tuple(self.meta.dataset_shape.sig)
        if self.meta.roi is not None:
            navsize = count_nonzero(self.meta.roi)
        else:
            navsize = prod(self.meta.dataset_shape.nav)
        warn_limit = 2**28
        loaded_size = prod(sigshape) * navsize * np.dtype(dtype).itemsize
        if loaded_size > warn_limit:
            log.warning("PickShiftedUDF is loading %s bytes, exceeding warning limit %s. "
                "Consider using or implementing an UDF to process data on the worker "
                "nodes instead." % (loaded_size, warn_limit))
        # We are using a "single" buffer since we mostly load single frames. A
        # "sig" buffer would work as well, but would require a transpose to
        # accomodate multiple frames in the last and not first dimension.
        # A "nav" buffer would allocate a NaN-filled buffer for the whole dataset.
        nav_dims = len(self.meta.dataset_shape.nav)
        return {
            'intensity': self.buffer(
                kind='single', extra_shape=(navsize, ) + sigshape, dtype=dtype
            ),
            'coordinates': self.buffer(
                kind="nav",
                dtype=int,
                extra_shape=(nav_dims, ),
            )
        }
