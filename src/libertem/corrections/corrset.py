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

    def adjust_tileshape(self, tile_shape, sig_shape, base_shape):
        excluded_pixels = self.get_excluded_pixels()
        if excluded_pixels is None:
            return tile_shape
        if excluded_pixels.nnz == 0:
            return tile_shape
        excluded_list = excluded_pixels.coords
        adjusted_shape = np.array(tile_shape)
        base_shape = np.array(base_shape)
        # Map of dimensions that should be shrunk
        # We have to grow or shrink monotonously to not cycle
        shrink = (adjusted_shape >= base_shape * 4)
        # Try to iteratively reduce or increase tile size
        # per signal dimension in case of intersections of tile boundaries
        # with the patching environment
        # until we avoid all excluded pixels.
        # This may fail in case of many excluded pixels or full excluded rows/columns,
        # depending on the tiling scheme. In that case,
        # swith to full frames while preserving tile size if possible.
        # FIXME may have to be adjusted to be balenced to real data and excluded pixel
        # incidence
        # FIXME benchmark to gauge performance impact
        for repeat in range(32):
            clean = True
            for dim in range(0, len(adjusted_shape)):
                start = adjusted_shape[dim]
                stop = sig_shape[dim]
                step = adjusted_shape[dim]
                for boundary in range(start, stop, step):
                    # Pixel on the left side of boundary
                    if boundary - 1 in excluded_list[dim]:
                        if shrink[dim]:
                            # If base_shape[dim] is 1, 2 is valid as well
                            adjusted_shape[dim] -= max(2, base_shape[dim])
                        else:
                            adjusted_shape[dim] += base_shape[dim]
                        clean = False
                        break
                    # Pixel on the right side of boundary
                    if boundary in excluded_list[dim]:
                        if shrink[dim]:
                            adjusted_shape[dim] -= base_shape[dim]
                        else:
                            # If base_shape[dim] is 1, 2 is valid as well
                            adjusted_shape[dim] += max(2, base_shape[dim])
                        clean = False
                        break
            if clean:
                break
            if np.any(adjusted_shape <= 0) or np.any(adjusted_shape > sig_shape):
                # We didn't find a solution
                clean = False
                break
        if clean:
            return tuple(adjusted_shape)
        else:
            # No solution found, switch to full frames
            return sig_shape
