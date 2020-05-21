import numba
import numpy as np

from libertem.common import Slice


@numba.njit
def _roi_to_indices(roi, start, stop, sync_offset=0):
    """
    Helper function to calculate indices from roi mask. Indices
    are flattened.

    Parameters
    ----------

    roi : numpy.ndarray of type bool, matching the navigation shape of the dataset

    start : int
        start frame index, relative to dataset start
        can for example be the start frame index of a partition

    stop : int
        stop before this frame index, relative to dataset
        can for example be the stop frame index of a partition

    sync_offset : int
        if positive, number of frames to skip from the start
        if negative, number of blank frames to insert at the start
        sync_offset should be in (-shape.nav.size, shape.nav.size)
    """
    roi = roi.reshape((-1,))
    part_roi = roi[start - sync_offset:stop - sync_offset]
    indices = np.arange(start, stop)
    return indices[part_roi]


def _roi_to_nd_indices(roi, part_slice: Slice):
    """
    Helper function to calculate indices from roi mask

    Parameters
    ----------

    roi : numpy.ndarray of type bool, matching the navigation shape of the dataset

    part_slice
        Slice indicating what part of the roi to operate on, for example,
        corresponding to a partition.
    """
    roi_slice = roi[part_slice.get(nav_only=True)]
    nav_dims = part_slice.shape.nav.dims
    total = 0
    frames_in_roi = np.count_nonzero(roi)
    for idx, value in np.ndenumerate(roi_slice):
        if not value:
            continue
        yield tuple(a + b
                    for a, b in zip(idx, part_slice.origin[:nav_dims]))
        # early exit: we know we don't have more frames in the roi
        total += 1
        if total == frames_in_roi:
            break
