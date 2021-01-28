import logging
from io import BytesIO

from matplotlib import colors, cm
import numpy as np
from PIL import Image
from empyre.vis.colors import ColormapCubehelix, ColormapPerception, ColormapHLS, ColormapClassic


__all__ = ['ColormapCubehelix', 'ColormapPerception', 'ColormapHLS',
           'ColormapClassic', 'cmaps', 'CMAP_CIRCULAR_DEFAULT',
           'visualize_simple', 'encode_image']
_log = logging.getLogger(__name__)


def _get_norm(result, norm_cls=colors.Normalize, vmin=None, vmax=None):
    # TODO: only normalize across the area where we already have values
    # can be accomplished by calculating min/max over are that was
    # affected by the result tiles. for now, ignoring 0 works fine
    result = result.astype(np.float32)
    if vmin is None:
        vmin = 0
        result_ne_zero = result[result != 0]
        if len(result_ne_zero) > 0:
            vmin = np.min(result_ne_zero)
    if vmax is None:
        vmax = np.max(result)
    return norm_cls(vmin=vmin, vmax=vmax)


def encode_image(result, save_kwargs=None):
    """
    Save the RGBA data in ``result`` to an image with parameters ``save_kwargs``
    passed to ``PIL.Image.save``.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    save_kwargs : dict or None
        dict of kwargs passed to Pillow when saving the image, can be used to set
        the file format, quality, ...

    Returns
    -------

    BytesIO
        a buffer containing the result image (as PNG/JPG/... depending on save_kwargs)
    """
    if save_kwargs is None:
        save_kwargs = {'format': 'png'}
    # see also: https://stackoverflow.com/a/10967471/540644
    im = Image.fromarray(result)
    buf = BytesIO()
    im = im.convert(mode="RGB")
    im.save(buf, **save_kwargs)
    buf.seek(0)
    return buf


def visualize_simple(result, colormap=None, logarithmic=False, vmin=None, vmax=None):
    """
    Normalize and visualize ``result`` with ``colormap`` and return the
    resulting RGBA data as an array.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    colormap : matplotlib colormap or None
        colormap used for visualizing intensity values, defaults to ColormapCubehelix()

    Returns
    -------

    np.array
        A numpy array of shape (Y, X, 4) containing RGBA data, suitable for
        passing to `Image.fromarray` in PIL.
    """
    if logarithmic:
        cnorm = colors.LogNorm
        result = result - np.min(result) + 1
    else:
        cnorm = colors.Normalize
    if colormap is None:
        colormap = cm.gist_earth
    norm = _get_norm(result, norm_cls=cnorm, vmin=vmin, vmax=vmax)
    normalized = norm(result)
    colored = colormap(normalized, bytes=True)
    return colored


cmaps = {'cubehelix_standard': ColormapCubehelix(),
         'cubehelix_reverse': ColormapCubehelix(reverse=True),
         'cubehelix_circular': ColormapCubehelix(start=1, rot=1,
                                                 minLight=0.5, maxLight=0.5, sat=2),
         'perception_circular': ColormapPerception(),
         'hls_circular': ColormapHLS(),
         'classic_circular': ColormapClassic()}

CMAP_CIRCULAR_DEFAULT = cmaps['cubehelix_circular']
