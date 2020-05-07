import numpy as np

from libertem.common import Slice
from libertem.corrections.detector import correct


class CorrectionSet:
    def __init__(self, dark=None, gain=None):
        """
        A set of corrections to apply.

        .. versionadded:: 0.6.0

        Parameters
        ----------
        dark : np.ndarray
            A 2D array containing a dark frame to substract from all frames,
            its shape needs to match the signal shape of the dataset.

        gain : np.ndarray
            A 2D array containing a gain map to multiply with each frame,
            its shape needs to match the signal shape of the dataset.
        """
        self._dark = dark
        self._gain = gain

    def get_dark_frame(self):
        return self._dark

    def get_gain_map(self):
        return self._gain

    def have_corrections(self):
        dark_frame = self.get_dark_frame()
        gain_map = self.get_gain_map()
        return dark_frame is not None or gain_map is not None

    def apply(self, data: np.ndarray, tile_slice: Slice):
        """
        Apply corrections in-place to `data`, cropping the
        correction data to the `tile_slice`.
        """
        dark_frame = self.get_dark_frame()
        gain_map = self.get_gain_map()

        if not self.have_corrections():
            return

        if dark_frame is not None:
            dark_frame = dark_frame[tile_slice.get(sig_only=True)]
        if gain_map is not None:
            gain_map = gain_map[tile_slice.get(sig_only=True)]
        correct(
            buffer=data,
            dark_image=dark_frame,
            gain_map=gain_map,
            excluded_pixels=None,
            inplace=True,
        )
