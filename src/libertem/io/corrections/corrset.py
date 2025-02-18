import functools
from typing import Optional

import numpy as np
import numba
import sparse
import sparseconverter

from libertem.common import Slice
from libertem.io.corrections.detector import correct, RepairDescriptor


@numba.njit(cache=True)
def disjunct_multiplier(excluded, sig_shape, base_shape=1, target=1):
    '''
    Calculate an integer i close to target which is a multiple of base_shape
    and for which i * n not in "excluded" for any n > 0, i * n < sig_shape.

    Cases with a bad pixel at 0 are handled downstream.

    Parameters
    ----------

    excluded: Sequence
        Forbidden tile boundary positions
    sig_shape: int
        Signal shape of the adjusted dimension
    base_shape: int
        Base shape of the adjusted dimension
    target: int
        Target value for the tile size, find a solution close to this value.

    Returns
    -------

    int
    '''
    approx_multiplier = int(np.round(target / base_shape))
    current_value = base_shape * approx_multiplier
    max_excluded = np.max(excluded)
    excluded_map = np.zeros(max_excluded + 1, dtype=np.uint8)
    excluded_map[excluded] = 1
    if current_value >= target:
        sign = 1
    else:
        sign = -1
    for offset in range(max_excluded // base_shape + 1):
        # plus 0, minus 1, plus 2, ...
        current_value += offset * sign * base_shape
        sign *= -1
        if current_value <= 0:
            continue
        clear = True
        for multiplier in range(1, max_excluded // current_value + 1):
            index = current_value * multiplier
            bad = (
                index >= 0
                and index < sig_shape
                and index <= max_excluded
                and excluded_map[index]
            )
            if bad:
                clear = False
                break
        if clear:
            return current_value
    # target value is max_excluded + 1,
    # i.e. if the modulo is 0 we have to add one more
    multiple = max_excluded // base_shape + 1
    return min(multiple * base_shape, sig_shape)


class CorrectionSet:
    """
    A set of corrections to apply.

    .. versionadded:: 0.6.0

    Parameters
    ----------
    dark : np.ndarray
        An array containing a dark frame to substract from all frames,
        its shape needs to match the signal shape of the dataset.

    gain : np.ndarray
        An array containing a gain map to multiply with each frame,
        its shape needs to match the signal shape of the dataset.

    excluded_pixels : sparse.COO
        A "sparse pydata" COO array containing only entries for pixels
        that should be excluded. The shape needs to match the signal
        shape of the dataset. Can also be anything that is directly
        compatible with the :code:`sparse.COO` constructor, for example a
        "roi-like" NumPy array. A :code:`sparse.COO` array can be
        directly constructed from a coordinate array, using
        :code:`sparse.COO(coords=coords, data=1, shape=ds.shape.sig)`
    allow_empty : bool
        Do not throw an exception if a repair environment for an excluded pixel
        is empty. The pixel is left uncorrected in that case.
    """
    def __init__(
        self,
        dark: Optional[np.ndarray] = None,
        gain: Optional[np.ndarray] = None,
        excluded_pixels: Optional[sparse.COO] = None,
        allow_empty: bool = False
    ):
        self._dark = dark
        self._gain = gain
        if excluded_pixels is not None:
            excluded_pixels = sparseconverter.for_backend(
                excluded_pixels, sparseconverter.SPARSE_COO
            )
        self._excluded_pixels = excluded_pixels
        self._allow_empty = allow_empty
        if not allow_empty and excluded_pixels is not None:
            # Construct the environment for checking so that an exception is thrown
            # when the CorrectionSet is instantiated and not when workers try to apply it.
            _ = RepairDescriptor(
                sig_shape=excluded_pixels.shape,
                excluded_pixels=excluded_pixels.coords,
                allow_empty=False
            )

    def get_dark_frame(self) -> Optional[np.ndarray]:
        return self._dark

    def get_gain_map(self) -> Optional[np.ndarray]:
        return self._gain

    def get_excluded_pixels(self) -> Optional[sparse.COO]:
        return self._excluded_pixels

    def have_corrections(self) -> bool:
        corrs = [
            self.get_dark_frame(),
            self.get_gain_map(),
            self.get_excluded_pixels(),
        ]
        return any(c is not None for c in corrs)

    def apply(self, data: np.ndarray, tile_slice: Slice) -> None:
        """
        Apply corrections in-place to `data`, cropping the
        correction data to the `tile_slice`.
        """
        dark_frame = self.get_dark_frame()
        gain_map = self.get_gain_map()

        if not self.have_corrections():
            return

        sig_slice = tile_slice.get(sig_only=True)

        if dark_frame is not None:
            dark_frame = dark_frame[sig_slice]
        if gain_map is not None:
            gain_map = gain_map[sig_slice]

        correct(
            buffer=data,
            dark_image=dark_frame,
            gain_map=gain_map,
            repair_descriptor=self.repair_descriptor(tile_slice.discard_nav()),
            inplace=True,
            sig_shape=tuple(tile_slice.shape.sig),
            allow_empty=self._allow_empty
        )

    @functools.lru_cache(maxsize=512)
    def repair_descriptor(self, sig_slice):
        excluded_pixels = self.get_excluded_pixels()
        if excluded_pixels is not None:
            excluded_pixels = excluded_pixels[sig_slice.get(sig_only=True)]
            excluded_pixels = excluded_pixels.coords
        return RepairDescriptor(
            sig_shape=tuple(sig_slice.shape.sig),
            excluded_pixels=excluded_pixels,
            allow_empty=self._allow_empty
        )

    def adjust_tileshape(self, tile_shape, sig_shape, base_shape):
        excluded_pixels = self.get_excluded_pixels()
        if excluded_pixels is None:
            return tile_shape
        if excluded_pixels.nnz == 0:
            return tile_shape
        excluded_list = excluded_pixels.coords
        adjusted_shape = np.array(tile_shape)
        sig_shape = np.array(sig_shape)
        base_shape = np.array(base_shape)

        adjust(
            adjusted_shape_inout=adjusted_shape,
            sig_shape=sig_shape,
            base_shape=base_shape,
            excluded_list=excluded_list
        )
        invalid = np.logical_or(
            adjusted_shape <= 0,
            adjusted_shape > sig_shape
        )
        adjusted_shape[invalid] = sig_shape[invalid]
        return tuple(adjusted_shape)


def adjust(adjusted_shape_inout, sig_shape, base_shape, excluded_list):
    '''
    Adjust the tile shape to avoid collisions with patched pixels.

    Find a tile shape that is a multiple of base_shape in such a way that
    excluded pixels are not touching a tile boundary.

    The proposed tile shape in adjusted_shape_inout is used as a starting value.
    If collisions can't be avoided, use sig_shape as tile shape for that dimension.

    Parameters
    ----------

    adjusted_shape_inout: np.ndarray, size n_dim
        Shape to adjust, modified in place
    sig_shape: sequence, size n_dim
        Signal shape
    base_shape: sequence, size n_dim
        The adjusted shape is a multiple of base_shape
    excluded_list: 2D sequence
        Coordinates of exluded pixels of shape (n_dim, n_pixels)
    '''
    for dim in range(0, len(adjusted_shape_inout)):
        # Nothing to adjust, could trip downstream logic
        if sig_shape[dim] <= 1:
            continue
        unique = np.unique(excluded_list[dim])
        # Very many pixels in the way, low chances of a solution
        if len(unique) > sig_shape[dim] / 3:
            adjusted_shape_inout[dim] = sig_shape[dim]
        else:
            stop = sig_shape[dim]
            # Left and right side of an invalid pixel are forbidden
            forbidden = np.concatenate((unique, unique + 1))
            forbidden = forbidden[forbidden <= stop]
            # Invalid pixel at zero is handled separately
            nonzero_filter = forbidden != 0
            m = min(
                stop,
                disjunct_multiplier(
                    excluded=forbidden[nonzero_filter],
                    sig_shape=sig_shape[dim],
                    base_shape=base_shape[dim],
                    target=adjusted_shape_inout[dim]
                )
            )
            # handle zero
            if not np.all(nonzero_filter):
                min_size = max(m, 2)
            else:
                min_size = m
            # Current shape is not a clean multiple
            # of the ideal shape or too small: adjust
            if adjusted_shape_inout[dim] < min_size or adjusted_shape_inout[dim] % m != 0:
                adjusted_shape_inout[dim] = m
