import numpy as np


# FIXME we might want to use an external physics or coordinate system package
# Getting this right and consistent is extremely important for correct
# interpretation of physical measurements. In particular 4D STEM data
# and ptychography can get confusing.

def make_cartesian(polar):
    '''
    Accept list of polar vectors, return list of cartesian vectors

    Parameters
    ----------
    polars : numpy.ndarray of tuples [(r1, phi1), (r2, phi2), ...]
        Polar vectors

    Returns
    -------
    numpy.ndarray of tuples [(y, x), (y, x), ...]
    '''
    xes = np.cos(polar[..., 1]) * polar[..., 0]
    yes = np.sin(polar[..., 1]) * polar[..., 0]
    return np.array((yes.T, xes.T)).T


def make_polar(cartesian):
    '''
    Accept list of cartesian vectors, return list of polar vectors

    Parameters
    ----------
    cartesian : numpy.ndarray of tuples [(y, x), (y, x), ...]
        Cartesian vectors

    Returns
    -------

    Polar vector as numpy.ndarray of tuples [(r1, phi1), (r2, phi2), ...]
    '''
    ds = np.linalg.norm(cartesian, axis=-1)
    # (y, x)
    alphas = np.arctan2(cartesian[..., 0], cartesian[..., 1])
    return np.array((ds.T, alphas.T)).T


def rotate_precalc(y, x, cos_angle, sin_angle):
    '''
    Rotate with pre-calculated entries of
    the rotation matrix. This is useful when rotating within a loop,
    in particular when combined with numba.njit of this function for inlining.

    The rotation follows https://en.wikipedia.org/wiki/Rotation_matrix

    In pixel coordinates with y pointing down and x pointing right,
    the rotation is clockwise. In physical coordinates with y pointing up instead,
    it is counter-clockwise.

    Parameters
    ==========
    y, x : float or numpy.ndarray
        Y and X components of the vector(s)

    cos_angle, sin_angle : float or numpy.ndarray
        cos(phi) and sin(phi) of rotation angle phi

    Returns
    =======
    y, x : float or numpy.ndarray
        Y and X components of rotated vector(s)

    .. versionadded:: 0.6.0
    '''
    r_x = cos_angle * x - sin_angle * y
    r_y = sin_angle * x + cos_angle * y
    return r_y, r_x


def rotate_deg(y, x, degrees):
    '''
    Rotate by angle in degrees

    The rotation follows https://en.wikipedia.org/wiki/Rotation_matrix

    In pixel coordinates with y pointing down and x pointing right,
    the rotation is clockwise. In physical coordinates with y pointing up instead,
    it is counter-clockwise.

    Parameters
    ==========
    y, x : float or numpy.ndarray
        Y and X components of the vector(s)

    degrees : float or numpy.ndarray
        Rotation angle in degrees

    Returns
    =======
    y, x : float or numpy.ndarray
        Y and X components of rotated vector(s)

    .. versionadded:: 0.6.0
    '''
    return rotate_rad(y, x, np.pi/180*degrees)


def rotate_rad(y, x, radians):
    '''
    Rotate by angle in radians

    The rotation follows https://en.wikipedia.org/wiki/Rotation_matrix

    In pixel coordinates with y pointing down and x pointing right,
    the rotation is clockwise. In physical coordinates with y pointing up instead,
    it is counter-clockwise.

    Parameters
    ==========
    y, x : float or numpy.ndarray
        Y and X components of the vector(s)

    radians : float or numpy.ndarray
        Rotation angle in radians

    Returns
    =======
    y, x : float or numpy.ndarray
        Y and X components of rotated vector(s)

    .. versionadded:: 0.6.0
    '''
    return rotate_precalc(y, x, cos_angle=np.cos(radians), sin_angle=np.sin(radians))


def regularize_indices(indices):
    s = indices.shape
    # Output of mgrid
    if (len(s) == 3) and (s[0] == 2):
        result = np.concatenate(indices.T)
    # List of (i, j) pairs
    elif (len(s) == 2) and (s[1] == 2):
        result = indices
    else:
        raise ValueError(
            "Shape of indices is %s, expected (n, 2) or (2, n, m)" % str(indices.shape))
    return result


def frame_peaks(fy, fx, zero, a, b, r, indices):
    indices = regularize_indices(indices)
    peaks = calc_coords(zero, a, b, indices)
    selector = within_frame(peaks, r, fy, fx)
    return indices[selector], peaks[selector]


def calc_coords(zero, a, b, indices):
    '''
    Calculate coordinates from lattice vectors a, b and indices
    '''
    coefficients = np.array((a, b))
    return zero + np.dot(indices, coefficients)


def within_frame(peaks, r, fy, fx):
    '''
    Return a boolean vector indicating peaks that are within (r, r) and (fy - r, fx - r)
    '''
    selector = (peaks >= (r, r)) * (peaks < (fy - r, fx - r))
    return selector.all(axis=-1)
