import numpy as np
import numba


@numba.njit
def _correct_numba_inplace(buffer, dark_image, gain_map, exclude_pixels, repair_environments):
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
        Ragged array with environments for each pixel to use as a reference
        List with k entries, where each entry is an array of indices in the flattened
        signal dimension to use as a source

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

    for nav_block in range(nav_blocks):
        # Dark and gain blocked in sig
        for sig_block in range(sig_blocks):
            for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
                for sig in range(sig_block * sig_blocksize, (sig_block + 1) * sig_blocksize):
                    buffer[nav, sig] = (buffer[nav, sig] - dark_image[sig]) * gain_map[sig]

        # Dark and gain remainder of sig blocks
        for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
            for sig in range(sig_blocks*sig_blocksize, sig_blocks*sig_blocksize + sig_remainder):
                buffer[nav, sig] = (buffer[nav, sig] - dark_image[sig]) * gain_map[sig]

        # Hole repair blocked in nav
        for i, p in enumerate(exclude_pixels):
            buf = np.zeros((len(repair_environments[i])))
            for nav in range(nav_block * nav_blocksize, (nav_block + 1) * nav_blocksize):
                for offset, index in enumerate(repair_environments[i]):
                    buf[offset] = buffer[nav, index]
                buffer[nav, p] = np.median(buf)

    # Processing unblocked nav remainder
    for nav in range(nav_blocks * nav_blocksize, nav_blocks * nav_blocksize + nav_remainder):
        # Dark and gain unblocked in sig
        for sig in range(buffer.shape[1]):
            buffer[nav, sig] = (buffer[nav, sig] - dark_image[sig]) * gain_map[sig]

        # Hole repair
        for i, p in enumerate(exclude_pixels):
            buf = np.zeros((len(repair_environments[i])))
            for offset, index in enumerate(repair_environments[i]):
                buf[offset] = buffer[nav, index]
            buffer[nav, p] = np.median(buf)

    return buffer


def environment(excluded_pixel, sigshape):
    '''
    Calculate a hypercube surface around a pixel, excluding frame boundaries
    '''
    excluded_pixel = np.array(excluded_pixel)
    sigshape = np.array(sigshape)
    coords = np.mgrid[tuple(slice(p-1, p+2) for p in excluded_pixel)]
    coords = coords.reshape((len(excluded_pixel), -1))
    select = np.any(coords != excluded_pixel[:, np.newaxis], axis=0)
    select *= np.all(coords >= 0, axis=0)
    select *= np.all(coords < sigshape[:, np.newaxis], axis=0)
    return coords[..., select]


class RepairValueError(ValueError):
    pass


def correct(buffer, dark_image, gain_map, exclude_pixels, inplace=False):
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

    sig_shape = dark_image.shape
    nav_shape = s[0:-len(sig_shape)]

    if inplace:
        out = buffer
    else:
        out = buffer.copy()

    exclude_flat = np.ravel_multi_index(exclude_pixels, sig_shape)
    repairs = [environment(p, sig_shape) for p in exclude_pixels.T]

    repair_flat = tuple(
        [
            np.extract(np.invert(np.isin(a, exclude_flat)), a)
            for a in [np.ravel_multi_index(r, sig_shape) for r in repairs]
        ]
    )

    for i, r in enumerate(exclude_flat):
        if len(repair_flat[i]) == 0:
            raise RepairValueError("Calculated repair environment for pixel %i is empty" % i)

    # Patch to help Numba determine the type in case of an empty list
    if len(repair_flat) == 0:
        # That is an odd case that is hit during fuzzing
        repair_flat = (np.array([], dtype=np.int), )

    _correct_numba_inplace(
        buffer=out.reshape((np.prod(nav_shape), np.prod(sig_shape))),
        dark_image=dark_image.flatten(),
        gain_map=gain_map.flatten(),
        exclude_pixels=exclude_flat,
        repair_environments=repair_flat,
    )
    return out
