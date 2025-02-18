from typing import Union, Callable
from collections.abc import Iterable

import numpy as np
import scipy.sparse as sp
import sparse

from libertem.utils import make_polar

# Import here for backwards compatibility, refs #1031
from libertem.common.sparse import to_dense, to_sparse, is_sparse  # NOQA: F401

MaskArrayType = Union[np.ndarray, sp.coo_matrix, sp.dok_matrix]
MaskFactoriesType = Union[Callable[[], MaskArrayType], Iterable[Callable[[], MaskArrayType]]]


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius, antialiased=False):
    """
    Make a circular mask in a bool array for masking a region in an image.

    Parameters
    ----------
    centerX, centerY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    Examples
    --------

    >>> image = np.ones((9, 9))
    >>> mask = _make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    if antialiased:
        mask = radial_bins(
            centerX, centerY, imageSizeX, imageSizeY, radius, n_bins=1, use_sparse=False
        )[0]
    else:
        x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
        mask = x*x + y*y <= radius*radius
    return mask


def sparse_template_multi_stack(mask_index, offsetX, offsetY, template, imageSizeX, imageSizeY):
    '''
    Stamp the template in a multi-mask 3D stack at the positions indicated by
    mask_index, offsetY, offsetX. The function clips the bounding box as necessary.
    '''
    num_templates = len(mask_index)
    fy, fx = template.shape
    area = fy * fx
    total_index_size = num_templates * area
    y, x = np.mgrid[0:fy, 0:fx]

    data = np.zeros(total_index_size, dtype=template.dtype)
    coord_mask = np.zeros(total_index_size, dtype=int)
    coord_y = np.zeros(total_index_size, dtype=int)
    coord_x = np.zeros(total_index_size, dtype=int)

    for i in range(len(mask_index)):
        start = i * area
        stop = (i + 1) * area
        data[start:stop] = template.flatten()
        coord_mask[start:stop] = mask_index[i]
        coord_y[start:stop] = y.flatten() + offsetY[i]
        coord_x[start:stop] = x.flatten() + offsetX[i]

    selector = (coord_y >= 0) * (coord_y < imageSizeY) * (coord_x >= 0) * (coord_x < imageSizeX)

    return sparse.COO(
        data=data[selector],
        coords=np.stack((coord_mask[selector], coord_y[selector], coord_x[selector]), axis=0),
        shape=(int(max(mask_index) + 1), imageSizeY, imageSizeX)
    )


def sparse_circular_multi_stack(mask_index, centerX, centerY, imageSizeX, imageSizeY, radius):
    # we make sure it is odd
    bbox = int(2*np.ceil(radius) + 1)
    bbox_center = int((bbox - 1) // 2)
    template = circular(
        centerX=bbox_center,
        centerY=bbox_center,
        imageSizeX=bbox,
        imageSizeY=bbox,
        radius=radius)
    return sparse_template_multi_stack(
        mask_index=mask_index,
        offsetX=np.array(centerX, dtype=int) - bbox_center,
        offsetY=np.array(centerY, dtype=int) - bbox_center,
        template=template,
        imageSizeX=imageSizeX,
        imageSizeY=imageSizeY,
    )


def circular(centerX, centerY, imageSizeX, imageSizeY, radius, antialiased=False):
    """
    Make a circular mask as a 2D array

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Radius of the mask.

    Returns
    -------
    Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.
    """
    mask = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius, antialiased)
    return mask


def ring(centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner, antialiased=False):
    """
    Make a ring mask as a double array.

    Parameters
    ----------
    centreX, centreY : float
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    radius : float
        Outer radius of the ring.
    radius_inner : float
        Inner radius of the ring.

    Returns
    -------
    Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.
    """
    if antialiased:
        mask = radial_bins(
            centerX, centerY, imageSizeX, imageSizeY,
            radius=radius, radius_inner=radius_inner, n_bins=1, use_sparse=False
        )[0]
    else:
        outer = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius)
        inner = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius_inner)
        mask = outer & ~inner
    return mask


def radial_gradient(centerX, centerY, imageSizeX, imageSizeY, radius, antialiased=False):
    '''
    Generate a linear radial gradient from 0 to 1 within radius
    '''
    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    if antialiased:
        r = np.sqrt(x**2 + y**2)
        mask = radial_gradient_background_subtraction(
            r=r, r0=radius, r_outer=0
        )
    else:
        mask = (x*x + y*y <= radius*radius) * (np.sqrt(x*x + y*y) / radius)
    return mask


def radial_gradient_background_subtraction(r, r0, r_outer, delta=1):
    '''
    Generate a template with a linear radial gradient from 0 to 1 inside r0,
    linear transition region for antialiasing between [r0 - delta/2, r0 + delta/2[,
    and a negative ring with value -1 in [r0 + delta/2, r_outer].

    The function accepts the radius for each pixel as a parameter so that a distorted version can
    be generated with the stretchY and angle parameters of :meth:`~libertem.masks.polar_map`.

    Parameters
    ----------

    r : numpy.ndarray
        Map of radius for each pixel, typically 2D. This allows to work in distorted coordinate
        systems by assigning arbitrary radius values to each pixel.
        :meth:`~libertem.masks.polar_map` can generate elliptical maps as an example.
    r0 : float
        Inner radius to fill with a linear gradient in units of r
    r_outer : float
        Outer radius of ring from r0 to fill with -1 in units of r
    delta : float, optional
        Width of transition region between inner and outer in units of r
        with linear gradient for antialiasing or smoothening. Defaults to 1.

    Returns
    -------

    numpy.ndarray
        NumPy numpy.ndarray with the same shape and type of r with mask values assigned as
        described in the description.

    '''
    result = np.zeros_like(r)
    within = r < r0 - delta/2
    result[within] = r[within] / r0

    transition = (r >= r0 - delta/2) * (r < r0 + delta/2)
    result[transition] = (r0 - r[transition]) / (delta/2)

    without = (r >= r0 + delta/2) * (r <= r_outer)
    result[without] = -1

    return result


def polar_map(centerX, centerY, imageSizeX, imageSizeY, stretchY=1., angle=0.):
    '''
    Return a map of radius and angle.

    The optional parameters stretchY and angle allow to stretch and rotate the coordinate system
    into an elliptical form. This is useful to generate modified input data for functions that
    generate a template as a function of radius and angle.

    Parameters
    ----------

    centerX,centerY : float
        Center of the coordinate system in pixel coordinates
    imageSizeX,imageSizeY : int
        Size of the map to generate in px
    stretchY,angle : float, optional
        Stretch the radius elliptically by amount :code:`stretchY` in direction
        :code:`angle` in radians. :code:`angle = 0` means in Y direction.

    Returns
    -------

    Tuple[numpy.ndarray, numpy.ndarray]
        Map of radius and angle of shape :code:`(imageSizeY, imageSizeX)`
    '''
    y, x = np.mgrid[0:imageSizeY, 0:imageSizeX]
    dy = y - centerY
    dx = x - centerX
    if stretchY != 1.0 or angle != 0.:
        (dy, dx) = (
            (dy*np.cos(angle) - dx*np.sin(angle)) / stretchY,
            dx*np.cos(angle) + dy*np.sin(angle),
        )

    dy = dy.flatten()
    dx = dx.flatten()
    cartesians = np.stack((dy, dx)).T
    polars = make_polar(cartesians)
    return (
        polars[:, 0].reshape((imageSizeY, imageSizeX)),
        polars[:, 1].reshape((imageSizeY, imageSizeX))
    )


def balance(template):
    '''
    Accept a template with both positive and negative values and scale the negative
    part in such a way that the sum is zero.

    This is useful to generate masks that return zero when applied to a
    uniform background or linear gradient.
    '''
    result = template.copy()
    above = template > 0
    below = template < 0
    result[below] *= template[above].sum() / template[below].sum() * -1
    return result


def bounding_radius(centerX, centerY, imageSizeX, imageSizeY):
    '''
    Calculate a radius around centerX, centerY that covers the whole frame
    '''
    dy = max(centerY, imageSizeY - centerY)
    dx = max(centerX, imageSizeX - centerX)
    return int(np.ceil(np.sqrt(dy**2 + dx**2))) + 1


def radial_bins(centerX, centerY, imageSizeX, imageSizeY,
        radius=None, radius_inner=0, n_bins=None, normalize=False, use_sparse=None, dtype=None):
    '''
    Generate antialiased rings
    '''
    if radius is None:
        radius = bounding_radius(centerX, centerY, imageSizeX, imageSizeY)

    if n_bins is None:
        n_bins = int(np.round(radius - radius_inner))

    r, phi = polar_map(centerX, centerY, imageSizeX, imageSizeY)
    r = r.flatten()

    width = (radius - radius_inner) / n_bins
    bin_area = np.pi * (radius**2 - (radius - width)**2)

    if use_sparse is None:
        use_sparse = bin_area / (imageSizeX * imageSizeY) < 0.1

    if use_sparse:
        jjs = np.arange(len(r), dtype=np.int64)

    slices = []
    for r0 in np.linspace(radius_inner, radius - width, n_bins) + width/2:
        diff = np.abs(r - r0)
        # The "0.5" ensures that the bins overlap and sum up to exactly 1
        vals = np.maximum(0, np.minimum(1, width/2 + 0.5 - diff))
        if use_sparse:
            select = vals != 0
            vals = vals[select]
            if normalize:  # Make sure each bin has a sum of 1
                s = vals.sum()
                if not np.isclose(s, 0):
                    vals /= s
            slices.append(
                sparse.COO(
                    shape=(len(r), ),
                    data=vals.astype(dtype),
                    coords=(jjs[select])
                )
            )
        else:
            if normalize:  # Make sure each bin has a sum of 1
                s = vals.sum()
                if not np.isclose(s, 0):
                    vals /= s
            slices.append(vals.reshape((imageSizeY, imageSizeX)).astype(dtype))
    # Patch a singularity at the center
    if radius_inner < 0.5:
        yy = int(np.round(centerY))
        xx = int(np.round(centerX))
        if yy >= 0 and yy < imageSizeY and xx >= 0 and xx < imageSizeX:
            if use_sparse:
                index = yy * imageSizeX + xx
                diff = 1 - slices[0][index] - radius_inner
                patch = sparse.COO(shape=len(r), data=np.array([diff]), coords=np.array([index]))
                slices[0] += patch
            else:
                slices[0][yy, xx] = 1 - radius_inner
    if use_sparse:
        return sparse.stack(slices).reshape((-1, imageSizeY, imageSizeX))
    else:
        return np.stack(slices)


def background_subtraction(centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner,
        antialiased=False):
    mask_1 = circular(
        centerX, centerY, imageSizeX, imageSizeY, radius_inner, antialiased=antialiased
    )
    sum_1 = np.sum(mask_1)
    mask_2 = ring(
        centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner, antialiased=antialiased
    )
    sum_2 = np.sum(mask_2)
    mask = mask_1 - mask_2*sum_1/sum_2
    return mask


def rectangular(X, Y, Width, Height, imageSizeX, imageSizeY):
    """
    Make a rectangular mask as a 2D array of bool.
    Parameters
    ----------
    X, Y : Corner coordinates
        Centre point of the mask.
    imageSizeX, imageSizeY : int
        Size of the image to be masked.
    Width, Height : Width and Height of the rectangle
    Returns
    -------
    Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.
    """
    bool_mask = np.zeros([imageSizeY, imageSizeX], dtype="bool")
    if Height*Width > 0:
        ymin = min(Y, Y+Height)
        xmin = min(X, X+Width)
        ymax = max(Y, Y+Height)
        xmax = max(X, X+Width)
    elif Height > 0 and Width < 0:
        ymin = Y
        xmin = X+Width
        ymax = Y+Height
        xmax = X
    elif Height < 0 and Width > 0:
        ymin = Y+Height
        xmin = X
        ymax = Y
        xmax = X+Width
    else:
        ymin = 0
        xmin = 0
        ymax = -1
        xmax = -1
    ymin = int(ymin)
    xmin = int(xmin)
    ymax = int(ymax)
    xmax = int(xmax)
    bool_mask[max(0, ymin):min(ymax+1, imageSizeY), max(0, xmin):min(xmax+1, imageSizeX)] = 1
    return bool_mask


# TODO: dtype parameter? consistency with ring/circular above
def gradient_x(imageSizeX, imageSizeY, dtype=np.float32):
    return np.tile(
        np.ogrid[slice(0, imageSizeX)].astype(dtype), imageSizeY
    ).reshape(imageSizeY, imageSizeX)


def gradient_y(imageSizeX, imageSizeY, dtype=np.float32):
    return gradient_x(imageSizeY, imageSizeX, dtype).transpose()
