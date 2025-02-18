import numpy as np
import numba
import sparse
import sparseconverter

from libertem.common.math import prod
from libertem.common.sparse import is_sparse
from libertem.common.numba import (
    numba_ravel_multi_index_multi, numba_unravel_index_multi
)


@numba.njit(cache=True, nogil=True)
def _correct_numba_inplace(buffer, dark_image, gain_map, exclude_pixels, repair_environments,
        repair_counts):
    '''
    Numerical work horse to perform detector corrections

    This function uses blocked processing for cache efficiency, hence the nested loops. It is
    about 4x faster than a naive numpy implementation.

    Parameters
    ----------

    buffer:
        (n, m) with data, modified in-place.

    dark_image:
        (m) with dark frame to be subtracted first

    gain_map:
        (m) with gain map to multiply the data with after subtraction

    exclude_pixels:
        int(k) array of indices in the flattened signal dimension to patch

    repair_environments:
        Array with environments for each pixel to use as a reference.
        Array of shape (k, max_repair_counts)

    repair_counts:
        Array(int) of length k with number of valid entries in each entry in repair_environments.


    Returns
    -------
    buffer (modified in-place)
    '''
    nav_blocksize = 4
    sig_blocksize = 2**20 // (512 * 4 * nav_blocksize)

    nav_blocks = buffer.shape[0] // nav_blocksize
    nav_remainder = buffer.shape[0] % nav_blocksize

    sig_blocks = buffer.shape[1] // sig_blocksize
    sig_remainder = buffer.shape[1] % sig_blocksize

    def _get_dark_px(sig):
        if dark_image is None:
            return 0
        return dark_image[sig]

    def _get_gain_px(sig):
        if gain_map is None:
            return 1
        return gain_map[sig]

    for nav_block in range(nav_blocks):
        # Dark and gain blocked in sig
        for sig_block in range(sig_blocks):
            for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
                for sig in range(sig_block * sig_blocksize, (sig_block + 1) * sig_blocksize):
                    buffer[nav, sig] = (buffer[nav, sig] - _get_dark_px(sig)) * _get_gain_px(sig)

        # Dark and gain remainder of sig blocks
        for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
            for sig in range(sig_blocks*sig_blocksize, sig_blocks*sig_blocksize + sig_remainder):
                buffer[nav, sig] = (buffer[nav, sig] - _get_dark_px(sig)) * _get_gain_px(sig)

        # Hole repair blocked in nav
        for i, p in enumerate(exclude_pixels):
            if repair_counts[i] > 0:  # Avoid div0
                for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
                    acc = 0
                    for index in repair_environments[i, :repair_counts[i]]:
                        acc += buffer[nav, index]
                    buffer[nav, p] = acc / repair_counts[i]

    # Processing unblocked nav remainder
    for nav in range(nav_blocks * nav_blocksize, nav_blocks * nav_blocksize + nav_remainder):
        # Dark and gain unblocked in sig
        for sig in range(buffer.shape[1]):
            buffer[nav, sig] = (buffer[nav, sig] - _get_dark_px(sig)) * _get_gain_px(sig)

        # Hole repair
        for i, p in enumerate(exclude_pixels):
            if repair_counts[i] > 0:  # Avoid div0
                acc = 0
                for index in repair_environments[i, :repair_counts[i]]:
                    acc += buffer[nav, index]
                buffer[nav, p] = acc / repair_counts[i]

    return buffer


@numba.njit(cache=True, nogil=True)
def environments(excluded_pixels, sigshape):
    '''
    Calculate a hypercube surface around a pixel, excluding frame boundaries

    Returns
    -------
    repairs, repair_counts
        repairs : numpy.ndarray
            Array with shape (exclude_pixels, sig_dims, indices)
        repair_counts : numpy.ndarray
            Array with length exclude_pixels, containing the number of pixels
            in the repair environment
    '''

    max_repair_count = 3**len(sigshape) - 1
    num_pixels = len(excluded_pixels[0])
    repairs = np.zeros((num_pixels, len(sigshape), max_repair_count), dtype=np.intp)
    repair_counts = np.zeros(num_pixels, dtype=np.intp)
    all_indices = np.arange(3**len(sigshape), dtype=np.intp)
    coord_shape = np.full(len(sigshape), 3, dtype=np.intp)
    coord_offsets = numba_unravel_index_multi(all_indices, coord_shape) - 1
    for i in range(num_pixels):
        repair_count = 0
        for position in range(coord_offsets.shape[1]):
            select = False
            for dim in range(coord_offsets.shape[0]):
                coord = coord_offsets[dim, position] + excluded_pixels[dim, i]
                # Any of the coordinates is different
                select += (coord != excluded_pixels[dim, i])
            for dim in range(coord_offsets.shape[0]):
                coord = coord_offsets[dim, position] + excluded_pixels[dim, i]
                # All of the coordinates are within bounds
                select *= (coord >= 0)
                select *= (coord < sigshape[dim])
            if select:
                for dim in range(coord_offsets.shape[0]):
                    coord = coord_offsets[dim, position] + excluded_pixels[dim, i]
                    repairs[i, dim, repair_count] = coord
                repair_count += 1
        repair_counts[i] = repair_count

    return repairs, repair_counts


class RepairValueError(ValueError):
    pass


@numba.njit(cache=True, nogil=True)
def flatten_filter(excluded_pixels, repairs, repair_counts, sig_shape):
    '''
    Flatten excluded pixels and repair environments and filter for collisions

    Ravel indices to flattened signal dimension and
    removed damaged pixels from all repair environments, i.e. only use
    "good" pixels.
    '''
    excluded_flat = numba_ravel_multi_index_multi(excluded_pixels, sig_shape)

    max_repair_count = 3**len(sig_shape) - 1
    new_repair_counts = np.zeros_like(repair_counts)
    repair_flat = np.zeros((len(excluded_flat), max_repair_count), dtype=np.intp)

    excluded_dict = {}
    for i in excluded_flat:
        excluded_dict[i] = True

    for i in range(len(excluded_flat)):
        a = numba_ravel_multi_index_multi(repairs[i, ..., :repair_counts[i]], sig_shape)
        nonzero_index = 0
        for j in range(repair_counts[i]):
            if a[j] not in excluded_dict:
                repair_flat[i, nonzero_index] = a[j]
                nonzero_index += 1
        new_repair_counts[i] = nonzero_index
        if new_repair_counts[i] == 0:
            pass
            # TODO fix for Numba
            # raise RepairValueError("Repair environment for pixel %i is empty" % i)

    return (excluded_flat, repair_flat, new_repair_counts)


def correct(
        buffer, dark_image=None, gain_map=None, excluded_pixels=None, repair_descriptor=None,
        inplace=False, sig_shape=None, allow_empty=False):
    '''
    Function to perform detector corrections

    This function delegates the processing to a function written with numba that is
    about 4x faster than a naive numpy implementation.

    Parameters
    ----------

    buffer:
        shape (*nav, *sig) with data. It is modified in-place if inplace==True.

    dark_image:
        shape (*sig) with dark frame to be subtracted first

    gain_map:
        shape (*sig) with gain map to multiply the data with after subtraction

    exclude_pixels:
        int(sigs, k) array of indices in the signal dimension to patch.
        The first dimension is the number of signal dimensions, the second the number of pixels

    repair_descriptor : RepairDescriptor
        This allows to re-use the calculation and filtering of repair environments when
        specified instead of exclude_pixels. This is particularly advantageous for tiled processing.

    inplace:
        If True, modify the input buffer in-place.
        If False (default), copy the input buffer before correcting.

    Returns
    -------
    shape (*nav, *sig) If inplace==True, this is :code:`buffer` modified in-place.
    '''
    s = buffer.shape

    if dark_image is not None:
        sig_shape = dark_image.shape
        dark_image = dark_image.flatten()
    if gain_map is not None:
        sig_shape = gain_map.shape
        gain_map = gain_map.flatten()
    if sig_shape is None:
        raise ValueError("need either `dark_image`, `gain_map`, or `sig_shape`")
    nav_shape = s[0:-len(sig_shape)]

    if inplace:
        if buffer.dtype.kind not in ('f', 'c'):
            raise TypeError("In-place correction only supported for floating point data.")
        out = buffer
    else:
        # astype() is always a copy even if it is the same dtype
        out = buffer.astype(np.result_type(np.float32, buffer))

    if repair_descriptor is None:
        repair_descriptor = RepairDescriptor(
            sig_shape=sig_shape,
            excluded_pixels=excluded_pixels,
            allow_empty=allow_empty
        )
    else:
        repair_descriptor.check_empty_repairs(allow_empty=allow_empty)
        if excluded_pixels is not None:
            raise ValueError("Invalid arguments: Bot repair_descriptor and excluded_pixels set")

    _correct_numba_inplace(
        buffer=out.reshape((prod(nav_shape), prod(sig_shape))),
        dark_image=dark_image,
        gain_map=gain_map,
        exclude_pixels=repair_descriptor.exclude_flat,
        repair_environments=repair_descriptor.repair_flat,
        repair_counts=repair_descriptor.repair_counts,
    )
    return out


class RepairDescriptor:
    def __init__(self, sig_shape, excluded_pixels=None, allow_empty=False):
        if excluded_pixels is None:
            excluded_pixels = np.zeros((len(sig_shape), 0), dtype=np.intp)
        else:
            excluded_pixels = np.array(excluded_pixels)

        repairs, repair_counts = environments(excluded_pixels, np.array(sig_shape))

        self.exclude_flat, self.repair_flat, self.repair_counts = flatten_filter(
            excluded_pixels, repairs, repair_counts, sig_shape
        )
        self.check_empty_repairs(allow_empty=allow_empty)

    def empty_repairs(self):
        return np.argwhere(self.repair_counts == 0)

    def check_empty_repairs(self, allow_empty):
        if not allow_empty:
            empty = self.empty_repairs()
            if len(empty) > 0:
                raise RepairValueError(
                    f"Empty repair environments for pixel(s) number {empty}."
                )


def correct_dot_masks(masks, gain_map, excluded_pixels=None, allow_empty=False):
    mask_shape = masks.shape
    sig_shape = gain_map.shape
    masks = masks.reshape((-1, prod(sig_shape)))

    if excluded_pixels is not None:
        if is_sparse(masks):
            result = sparse.DOK(masks)
        else:
            result = masks.copy()
        desc = RepairDescriptor(sig_shape, excluded_pixels=excluded_pixels, allow_empty=allow_empty)
        for e, r, c in zip(desc.exclude_flat, desc.repair_flat, desc.repair_counts):
            result[:, e] = 0
            rep = masks[:, e] / c
            # We have to loop because of sparse.pydata limitations
            for m in range(result.shape[0]):
                for rr in r[:c]:
                    result[m, rr] = result[m, rr] + rep[m]
        if is_sparse(result):
            result = sparseconverter.for_backend(result, sparseconverter.SPARSE_COO)
    else:
        result = masks
    result = result * gain_map.flatten()
    return result.reshape(mask_shape)
