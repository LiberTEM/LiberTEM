import numpy as np

from libertem.common.math import prod
from libertem.common import Shape, Slice
from libertem.io.dataset.base import _roi_to_nd_indices


def get_coordinates(slice_: Slice, ds_shape: Shape, roi=None) -> np.ndarray:
    """
    Returns `numpy.ndarray` of coordinates that correspond to the frames in the actual
    navigation space which are part of the current tile or partition.

    Parameters
    ----------
    slice_: Slice
        Describes the location within the dataset with navigation
        dimension flattened and reduced to the ROI.

    ds_shape: Shape
        The original shape of the whole dataset, not influenced by the ROI

    roi: numpy.ndarray, optional
        Array of type bool, matching the navigation shape of the dataset
    """
    o = slice_.origin
    s = slice_.shape
    sig_dims = s.sig.dims
    start_idx = o[0]
    end_idx = o[0] + s[0]
    nav_shape = ds_shape[:-sig_dims]
    if roi is None:
        flat_nav_shape = tuple((int(prod(nav_shape)),))
        coordinates = np.stack(
            np.unravel_index(
                np.ravel_multi_index([np.arange(start_idx, end_idx)], flat_nav_shape),
                nav_shape
            ),
            axis=1
        )
    else:
        ds_shape = Shape(ds_shape, sig_dims=sig_dims)
        ds_slice = Slice(origin=[0] * len(ds_shape), shape=ds_shape)
        roi = roi.reshape(nav_shape)
        indices = _roi_to_nd_indices(roi, ds_slice)
        coordinates = np.array(list(indices)[start_idx:end_idx])
    return coordinates
