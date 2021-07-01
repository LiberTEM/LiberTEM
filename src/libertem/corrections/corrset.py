import functools

import numpy as np
import sparse
import primesieve.numpy

from libertem.common import Slice
from libertem.corrections.detector import correct, RepairDescriptor


def factorizations(n, primes):
    n = np.array(n)
    factorization = np.zeros((len(n), len(primes)), dtype=n.dtype)
    if np.max(n) > np.max(primes) * 2:
        raise ValueError(
            "np.max(n) > np.max(primes) * 2, probably not enough primes."
        )
    while np.any(n > 1):
        zero_modulos = (n[:, np.newaxis] % primes[np.newaxis, :]) == 0
        factorization[zero_modulos] += 1
        f = np.prod(primes[np.newaxis, :]**zero_modulos, axis=1)
        n = n // f

    return factorization


def min_disjunct_multiplier(excluded):
    '''
    Calculate a small integer i for which i * n not in "excluded" for any n > 0

    To make sure that the tile shape negotiation retains as much flexibility as possible,
    it is important to find a small integer and not just any integer that fulfills
    this condition.
    '''
    if len(excluded):
        # If two integers are equal, their prime factor decompositions
        # are equal, too, and vice versa.
        # We find the global maximum power for each of the prime factors
        # that construct the elements of "excluded".
        # By choosing a number that has one power more, we make sure
        # that multiples of that number can never be equal to one of the excluded
        # elements: The prime factor decompositions of any multiple of that number
        # will always contain that additional power and thus never be equal the prime
        # factor decomposition of an excluded pixel.
        primes = primesieve.numpy.primes(max(np.max(excluded), 2))
        fac = factorizations(np.array(excluded), primes)
        ceiling = primes ** (np.max(fac, axis=0) + 1)
        return int(np.min(ceiling))
    else:
        return 1


class CorrectionSet:
    """
    A set of corrections to apply.

    .. versionadded:: 0.6.0

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
    allow_empty : bool
        Do not throw an exception if a repair environment is empty. The pixel
        is left uncorrected in that case.
    """
    def __init__(self, dark=None, gain=None, excluded_pixels=None, allow_empty=False):
        self._dark = dark
        self._gain = gain
        if excluded_pixels is not None:
            excluded_pixels = sparse.COO(excluded_pixels, prune=True)
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
        for repeat in range(7):
            clean = adjust_iteration(
                adjusted_shape_inout=adjusted_shape,
                sig_shape=sig_shape,
                base_shape=base_shape,
                shrink=shrink,
                excluded_list=excluded_list
            )
            if np.all(clean):
                break
        invalid = np.logical_or(
            np.logical_not(clean),
            adjusted_shape <= 0,
            adjusted_shape > sig_shape
        )
        adjusted_shape[invalid] = sig_shape[invalid]
        return tuple(adjusted_shape)


def adjust_iteration(adjusted_shape_inout, sig_shape, base_shape, shrink, excluded_list):
    clean = np.ones(len(adjusted_shape_inout), dtype=bool)
    for dim in range(0, len(adjusted_shape_inout)):
        # Nothing to adjust, could trip downstream logic
        if sig_shape[dim] <= 1:
            continue
        unique = np.unique(excluded_list[dim])
        # Very many pixels in the way, low chances of a solution
        if len(unique) > sig_shape[dim] / 3:
            clean[dim] = False
            adjusted_shape_inout[dim] = sig_shape[dim]
        elif adjusted_shape_inout[dim] == 1 and base_shape[dim] == 1:
            clean[dim] = adjust_direct(
                clean=clean[dim],
                adjusted_shape_inout=adjusted_shape_inout,
                sig_shape=sig_shape,
                dim=dim,
                excluded_list=unique,
            )
        else:
            clean[dim] = adjust_heuristic(
                clean=clean[dim],
                adjusted_shape_inout=adjusted_shape_inout,
                base_shape=base_shape,
                sig_shape=sig_shape,
                shrink=shrink,
                dim=dim,
                excluded_list=unique,
            )
    return clean


def adjust_direct(clean, adjusted_shape_inout, sig_shape, dim, excluded_list):
    stop = sig_shape[dim]
    forbidden = np.concatenate((excluded_list, excluded_list + 1))
    forbidden = forbidden[forbidden < stop]
    nonzero_filter = forbidden != 0
    m = min(stop, min_disjunct_multiplier(forbidden[nonzero_filter]))
    # In case we have a zero where the existing logic doesn't work
    if not np.all(nonzero_filter):
        m = max(m, 2)
    if adjusted_shape_inout[dim] != m:
        adjusted_shape_inout[dim] = m
        clean = False
    return clean


def adjust_heuristic(clean, adjusted_shape_inout, base_shape, sig_shape, shrink, dim,
        excluded_list):
    start = adjusted_shape_inout[dim]
    stop = sig_shape[dim]
    step = adjusted_shape_inout[dim]
    excluded_set = frozenset(excluded_list)
    right_boundary_set = frozenset(range(start, stop, step))
    left_boundary_set = frozenset(range(start-1, stop-1, step))

    right_of = not right_boundary_set.isdisjoint(excluded_set)
    left_of = not left_boundary_set.isdisjoint(excluded_set)

    # Pixel on the left side of boundary
    if left_of:
        if shrink[dim]:
            # If base_shape[dim] is 1, 2 is valid as well
            adjusted_shape_inout[dim] -= max(2, base_shape[dim])
        else:
            adjusted_shape_inout[dim] += base_shape[dim]
        clean = False
    # Pixel on the right side of boundary
    if right_of:
        if shrink[dim]:
            adjusted_shape_inout[dim] -= base_shape[dim]
        else:
            # If base_shape[dim] is 1, 2 is valid as well
            adjusted_shape_inout[dim] += max(2, base_shape[dim])
        clean = False
    return clean
