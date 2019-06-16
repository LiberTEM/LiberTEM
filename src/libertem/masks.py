import numpy as np
import scipy.sparse as sp
import sparse

from libertem.utils import make_polar


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):
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
        coords=(coord_mask[selector], coord_y[selector], coord_x[selector]),
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
        offsetX=np.array(centerX, dtype=np.int) - bbox_center,
        offsetY=np.array(centerY, dtype=np.int) - bbox_center,
        template=template,
        imageSizeX=imageSizeX,
        imageSizeY=imageSizeY,
    )


def circular(centerX, centerY, imageSizeX, imageSizeY, radius):
    """
    Make a circular mask as a 2D array of bool.

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


def polar_map(centerX, centerY, imageSizeX, imageSizeY):
    y, x = np.mgrid[0:imageSizeY, 0:imageSizeX]
    dy = y - centerY
    dx = x - centerX
    dy = dy.flatten()
    dx = dx.flatten()
    cartesians = np.stack((dy, dx)).T
    polars = make_polar(cartesians)
    return (
        polars[:, 0].reshape((imageSizeY, imageSizeX)),
        polars[:, 1].reshape((imageSizeY, imageSizeX))
    )


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

    y, x = np.ogrid[0:imageSizeY, 0:imageSizeX]
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
            slices.append(sparse.COO(shape=len(r), data=vals.astype(dtype), coords=(jjs[select],)))
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
        if use_sparse:
            index = yy * imageSizeX + xx
            diff = 1 - slices[0][index] - radius_inner
            patch = sparse.COO(shape=len(r), data=[diff], coords=[index])
            slices[0] += patch
        else:
            slices[0][yy, xx] = 1 - radius_inner
    if use_sparse:
        return sparse.stack(slices).reshape((-1, imageSizeY, imageSizeX))
    else:
        return np.stack(slices)


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
