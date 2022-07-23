import numpy as np
from scipy.ndimage import gaussian_filter

from libertem.common.math import prod
from libertem.utils import make_cartesian, make_polar, frame_peaks
import libertem.masks as m


def cbed_frame(
        fy=128, fx=128, zero=None, a=None, b=None, indices=None,
        radius=4, all_equal=False, margin=None):
    if zero is None:
        zero = (fy//2, fx//2)
    zero = np.array(zero)
    if a is None:
        a = (fy//8, 0)
    a = np.array(a)
    if b is None:
        b = make_cartesian(make_polar(a) - (0, np.pi/2))
    b = np.array(b)
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]
    if margin is None:
        margin = radius
    indices, peaks = frame_peaks(fy=fy, fx=fx, zero=zero, a=a, b=b, r=margin, indices=indices)

    data = np.zeros((1, fy, fx), dtype=np.float32)

    dists = np.linalg.norm(peaks - zero, axis=-1)
    max_val = max(dists.max() + 1, len(peaks) + 1)

    for i, p in enumerate(peaks):
        data += m.circular(
            centerX=p[1],
            centerY=p[0],
            imageSizeX=fx,
            imageSizeY=fy,
            radius=radius,
            antialiased=True,
        ) * (1 if all_equal else max(1, max_val - dists[i] + i))

    return (data, indices, peaks)


def hologram_frame(amp, phi,
                   counts=1000.,
                   sampling=5.,
                   visibility=1.,
                   f_angle=30.,
                   gaussian_noise=None,
                   poisson_noise=None):
    """
    Generates holograms using phase and amplitude as an input

    See :ref:`holography app` for detailed application example

    .. versionadded:: 0.3.0

    Notes
    -----
    Theoretical basis for hologram simulations see in:
    Lichte, H., and M. Lehmann. Rep. Prog. Phys. 71 (2008): 016102.
    doi:10.1088/0034-4885/71/1/016102
    :cite:`Lichte2008`

    Parameters
    ----------
    amp, phi: np.ndarray, 2d
        normalized amplitude and phase images of the same shape

    counts: float, default: 1000.
        Number of electron counts in vacuum

    sampling: float, default: 5.
        Hologram fringe sampling (number of pixels per fringe)

    visibility: float, default: 1.
        Hologram fringe visibility (aka fringe contrast)

    f_angle: float, default: 30.
        Angle in degrees of hologram fringes with respect to X-axis

    gaussian_noise: float or int or None, default: None.
        Amount of Gaussian smoothing determined by sigma parameter
        applied to the hologram simulating effect of focus spread or
        PSF of the detector.

    poisson_noise: float or int or None, default: None.
        Amount of Poisson applied to the hologram.

    Returns
    -------
    holo: np.ndarray, 2d
        hologram image
    """
    if not amp.shape == phi.shape:
        raise ValueError('Amplitude and phase should be 2d arrays of the same shape.')
    sy, sx = phi.shape
    x, y = np.meshgrid(np.arange(sx), np.arange(sy))
    f_angle = f_angle / 180. * np.pi

    holo = counts / 2 * (1. + amp ** 2 + 2. * amp * visibility
                         * np.cos(2. * np.pi * y / sampling * np.cos(f_angle)
                                  + 2. * np.pi * x / sampling * np.sin(f_angle)
                                  - phi))

    if poisson_noise:
        if not isinstance(poisson_noise, (float, int)):
            raise ValueError("poisson_noise parameter should be float or int or None.")
        noise_scale = poisson_noise * counts
        holo = noise_scale * np.random.poisson(holo / noise_scale)

    if gaussian_noise:
        if not isinstance(gaussian_noise, (float, int)):
            raise ValueError("gaussian_noise parameter should be float or int or None.")
        holo = gaussian_filter(holo, gaussian_noise)

    return holo


def gradient_data(nav_dims, sig_dims):
    data = np.linspace(
        start=5, stop=30, num=prod(nav_dims) * prod(sig_dims), dtype=np.float32
    )
    return data.reshape(nav_dims + sig_dims)


def exclude_pixels(sig_dims, num_excluded):
    '''
    Generate a list of excluded pixels that
    can be reconstructed faithfully from their neighbors
    in a linear gradient dataset
    '''
    if num_excluded == 0:
        return None
    # Map of pixels that can be reconstructed faithfully from neighbors in a linear gradient
    free_map = np.ones(sig_dims, dtype=bool)

    # Exclude all border pixels
    for dim in range(len(sig_dims)):
        selector = tuple(slice(None) if i != dim else (0, -1) for i in range(len(sig_dims)))
        free_map[selector] = False

    exclude = []

    while len(exclude) < num_excluded:
        exclude_item = tuple(np.random.randint(low=1, high=s-1) for s in sig_dims)
        if free_map[exclude_item]:
            exclude.append(exclude_item)
            knock_out = tuple(slice(e - 1, e + 2) for e in exclude_item)
            # Remove the neighbors of a bad pixel
            # since that can't be reconstructed faithfully from a linear gradient
            free_map[knock_out] = False

    # Transform from list of tuples with length of number of dimensions
    # to array of indices per dimension
    return np.array(exclude).T
