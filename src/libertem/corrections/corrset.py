import numpy as np
import sparse

from libertem.common import Slice
from libertem.corrections.detector import correct


class CorrectionSet:
    """
    A set of corrections to apply.

    .. versionadded:: 0.6.0
    """
    def __init__(self, dark=None, gain=None, excluded_pixels=None):
        """
        Parameters
        ----------
        dark : np.ndarray
            A ND array containing a dark frame to substract from all frames,
            its shape needs to match the signal shape of the dataset.

        gain : np.ndarray
            A ND array containing a gain map to multiply with each frame,
            its shape needs to match the signal shape of the dataset.

        excluded_pixels : sparse.COO
            A "sparse pydata" COO array containing only entries for pixels
            that should be excluded. The shape needs to match the signal
            shape of the dataset. Can also be anything that is directly
            compatible with the :code:`sparse.COO` constructor, for example a
            "roi-like" numpy array. A :code:`sparse.COO` array can be
            directly constructed from a coordinate array, using
            :code:`sparse.COO(coords=coords, data=1, shape=ds.shape.sig)`
        """
        self._dark = dark
        self._gain = gain
        if excluded_pixels is not None:
            excluded_pixels = sparse.COO(excluded_pixels, prune=True)
        self._excluded_pixels = excluded_pixels

    def get_dark_frame(self):
        return self._dark

    def get_gain_map(self):
        return self._gain

    def get_excluded_pixels(self):
        return self._excluded_pixels

    def have_corrections(self):
        corrs = [
            self.get_dark_frame(),
            self.get_gain_map(),
            self.get_excluded_pixels(),
        ]
        return any(c is not None for c in corrs)

    def apply(self, data: np.ndarray, tile_slice: Slice):
        """
        Apply corrections in-place to `data`, cropping the
        correction data to the `tile_slice`.
        """
        dark_frame = self.get_dark_frame()
        gain_map = self.get_gain_map()
        excluded_pixels = self.get_excluded_pixels()

        if not self.have_corrections():
            return

        sig_slice = tile_slice.get(sig_only=True)

        if dark_frame is not None:
            dark_frame = dark_frame[sig_slice]
        if gain_map is not None:
            gain_map = gain_map[sig_slice]
        if excluded_pixels is not None:
            excluded_pixels = excluded_pixels[sig_slice]
            excluded_pixels = excluded_pixels.coords
        correct(
            buffer=data,
            dark_image=dark_frame,
            gain_map=gain_map,
            excluded_pixels=excluded_pixels,
            inplace=True,
            sig_shape=tuple(tile_slice.shape.sig),
        )
