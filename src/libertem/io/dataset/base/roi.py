import typing
import numpy as np

from libertem.common import Slice
from libertem.common.math import count_nonzero, ndenumerate

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.partition import Partition


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
    frames_in_roi = count_nonzero(roi)
    for idx, value in ndenumerate(roi_slice):
        if not value:
            continue
        yield tuple(a + b
                    for a, b in zip(idx, part_slice.origin[:nav_dims]))
        # early exit: we know we don't have more frames in the roi
        total += 1
        if total == frames_in_roi:
            break


def roi_for_partition(roi: np.ndarray, partition: 'Partition') -> np.ndarray:
    return roi.reshape(-1)[partition.slice.get(nav_only=True)]
