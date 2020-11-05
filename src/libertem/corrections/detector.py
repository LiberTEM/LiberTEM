import numpy as np
import numba
import sparse

from libertem.masks import is_sparse


@numba.njit
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
            acc = 0
            for index in repair_environments[i, :repair_counts[i]]:
                acc += buffer[nav, index]
            buffer[nav, p] = acc / repair_counts[i]

    return buffer


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
    sigshape = np.array(sigshape)
    excluded_pixels = np.array(excluded_pixels).T
    max_repair_count = 3**len(sigshape) - 1
    repairs = np.zeros((len(excluded_pixels), len(sigshape), max_repair_count), dtype=np.intp)
    repair_counts = np.zeros(len(excluded_pixels), dtype=np.intp)
    for i, p in enumerate(excluded_pixels):
        coords = np.mgrid[tuple(slice(pp-1, pp+2) for pp in p)]
        coords = coords.reshape((len(p), -1))
        select = np.any(coords != p[:, np.newaxis], axis=0)
        select *= np.all(coords >= 0, axis=0)
        select *= np.all(coords < sigshape[:, np.newaxis], axis=0)
        repair_count = np.count_nonzero(select)
        repair_counts[i] = repair_count
        repairs[i, ..., :repair_count] = coords[..., select]

    return repairs, repair_counts


class RepairValueError(ValueError):
    pass


def flatten_filter(excluded_pixels, repairs, repair_counts, sig_shape):
    '''
    Flatten excluded pixels and repair environments and filter for collisions

    Ravel indices to flattened signal dimension and
    removed damaged pixels from all repair environments, i.e. only use
    "good" pixels.
    '''
    excluded_flat = np.ravel_multi_index(excluded_pixels, sig_shape)

    max_repair_count = 3**len(sig_shape) - 1
    new_repair_counts = np.zeros_like(repair_counts)
    repair_flat = np.zeros((len(excluded_flat), max_repair_count), dtype=np.intp)

    for i in range(len(excluded_flat)):
        a = np.ravel_multi_index(repairs[i, ..., :repair_counts[i]], sig_shape)
        rep = np.extract(np.invert(np.isin(a, excluded_flat)), a)
        if len(rep) == 0:
            raise RepairValueError("Repair environment for pixel %i is empty" % i)
        new_repair_counts[i] = len(rep)
        repair_flat[i, :len(rep)] = rep

    return (excluded_flat, repair_flat, new_repair_counts)


def correct(
        buffer, dark_image=None, gain_map=None, excluded_pixels=None,
        inplace=False, sig_shape=None):
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

    if excluded_pixels is None:
        excluded_pixels = np.zeros((len(sig_shape), 0), dtype=np.int)

    repairs, repair_counts = environments(excluded_pixels, sig_shape)

    exclude_flat, repair_flat, repair_counts = flatten_filter(
        excluded_pixels, repairs, repair_counts, sig_shape
    )

    _correct_numba_inplace(
        buffer=out.reshape((np.prod(nav_shape), np.prod(sig_shape))),
        dark_image=dark_image,
        gain_map=gain_map,
        exclude_pixels=exclude_flat,
        repair_environments=repair_flat,
        repair_counts=repair_counts,
    )
    return out


def correct_dot_masks(masks, gain_map, excluded_pixels=None):
    mask_shape = masks.shape
    sig_shape = gain_map.shape
    masks = masks.reshape((-1, np.prod(sig_shape)))

    if excluded_pixels is not None:
        if is_sparse(masks):
            result = sparse.DOK(masks)
        else:
            result = masks.copy()
        repairs, repair_counts = environments(excluded_pixels, sig_shape)
        for e, r, c in zip(*flatten_filter(excluded_pixels, repairs, repair_counts, sig_shape)):
            result[:, e] = 0
            rep = masks[:, e] / c
            # We have to loop because of sparse.pydata limitations
            for m in range(result.shape[0]):
                for rr in r[:c]:
                    result[m, rr] = result[m, rr] + rep[m]
        if is_sparse(result):
            result = sparse.COO(result)
    else:
        result = masks
    result = result * gain_map.flatten()
    return result.reshape(mask_shape)
