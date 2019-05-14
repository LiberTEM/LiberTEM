import numpy as np
import scipy.sparse as sp
import sparse


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask in a bool array for masking a region in an image.

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
    Boolean Numpy 2D Array
        Array with the shape (imageSizeX, imageSizeY) with the mask.

    Examples
    --------
    >>> import numpy as np
    >>> image = np.ones((9, 9))
    >>> mask = make_circular_mask(4, 4, 9, 9, 2)
    >>> image_masked = image*mask
    >>> import matplotlib.pyplot as plt
    >>> cax = plt.imshow(image_masked)
    """
    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = x*x + y*y <= radius*radius
    return(mask)


def circular(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask as a double array.

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
    bool_mask = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius)
    return bool_mask


def ring(centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner):
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
    outer = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius)
    inner = _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius_inner)
    bool_mask = outer & ~inner
    return bool_mask


def radial_gradient(centerX, centerY, imageSizeX, imageSizeY, radius):
    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = (x*x + y*y <= radius*radius) * (np.sqrt(x*x + y*y) / radius)
    return mask


def background_substraction(centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner):
    mask_1 = circular(centerX, centerY, imageSizeX, imageSizeY, radius_inner)
    sum_1 = np.sum(mask_1)
    mask_2 = ring(centerX, centerY, imageSizeX, imageSizeY, radius, radius_inner)
    sum_2 = np.sum(mask_2)
    mask = mask_1 - mask_2*sum_1/sum_2
    return mask


# TODO: dtype parameter? consistency with ring/circular above
def gradient_x(imageSizeX, imageSizeY, dtype=np.float32):
    return np.tile(
        np.ogrid[slice(0, imageSizeX)].astype(dtype), imageSizeY
    ).reshape(imageSizeY, imageSizeX)


def gradient_y(imageSizeX, imageSizeY, dtype=np.float32):
    return gradient_x(imageSizeY, imageSizeX, dtype).transpose()


def to_dense(a):
    if isinstance(a, sparse.SparseArray):
        return a.todense()
    elif sp.issparse(a):
        return a.toarray()
    else:
        return np.array(a)


def to_sparse(a):
    if isinstance(a, sparse.COO):
        return a
    elif isinstance(a, sparse.SparseArray):
        return sparse.COO(a)
    elif sp.issparse(a):
        return sparse.COO.from_scipy_sparse(a)
    else:
        return sparse.COO.from_numpy(np.array(a))


def is_sparse(a):
    return isinstance(a, sparse.SparseArray) or sp.issparse(a)
