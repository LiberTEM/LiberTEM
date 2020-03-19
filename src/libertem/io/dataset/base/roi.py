import numpy as np

from libertem.common import Slice


def _roi_to_indices(roi, start, stop):
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
        stop frame index, relative to dataset start
        can for example be the stop frame index of a partition
    """
    roi = roi.reshape((-1,))
    frames_in_roi = np.count_nonzero(roi)
    total = 0
    for flag, idx in zip(roi[start:stop], range(start, stop)):
        if flag:
            yield idx
            # early exit: we know we don't have more frames in the roi
            total += 1
            if total == frames_in_roi:
                break


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
