import abc
import colorsys
import logging
from io import BytesIO

from matplotlib import colors, cm
import numpy as np
from PIL import Image


__all__ = ['Colormap2D', 'ColormapCubehelix', 'ColormapPerception', 'ColormapHLS',
           'ColormapClassic', 'interpolate_color', 'cmaps', 'CMAP_CIRCULAR_DEFAULT',
           'visualize_simple', 'encode_image']
_log = logging.getLogger(__name__)


class Colormap2D(colors.Colormap, metaclass=abc.ABCMeta):
    """Colormap subclass for encoding directions with colors.

    This abstract class is used as a superclass/interface for 3D vector plotting capabilities.
    In general, a circular colormap should be used to encode the in-plane angle (hue). The
    perpendicular angle is encoded via luminance variation (up: white, down: black). Finally,
    the length of a vector is encoded via saturation. Decreasing vector length causes a desaturated
    color. Subclassing colormaps get access to routines to plot a colorwheel (which should
    ideally be located in the 50% luminance plane, which depends strongly on the underlying map),
    a convenience function to interpolate color tuples and a function to return rgb triples for a
    given vector. The :class:`~.Colormap3D` class itself subclasses the matplotlib base colormap.

    """

    _log = logging.getLogger(__name__ + '.Colormap3D')

    def rgb_from_vector(self, vector, vmax=None):
        """Construct a hls tuple from three coordinates representing a 3D direction.

        Parameters
        ----------
        vector: tuple (N=2) or :class:`~numpy.ndarray`
            Vector containing the x and y component, or a numpy array encompassing the
            components as two lists.

        Returns
        -------
        rgb:  :class:`~numpy.ndarray`
            Numpy array containing the calculated color tuples.

        """
        self._log.debug('Calling rgb_from_vector')
        x, y = np.asarray(vector)
        # Calculate spherical coordinates:
        rho = np.hypot(x, y)
        phi = np.asarray(np.arctan2(y, x))
        phi[phi < 0] += 2 * np.pi
        # Calculate color deterministics:
        hue = phi / (2 * np.pi)
        if vmax is None:
            vmax = rho.max()
        sat = rho / (vmax + 1E-30)
        # Calculate RGB from hue with colormap:
        rgba = np.asarray(self(hue))
        r, g, b = rgba[..., 0], rgba[..., 1], rgba[..., 2]
        # Interpolate saturation:
        r, g, b = interpolate_color(sat, (0.5, 0.5, 0.5), np.stack((r, g, b), axis=-1))
        # Return RGB:
        return np.asarray(255 * np.stack((r, g, b), axis=-1), dtype=np.uint8)


class ColormapCubehelix(colors.LinearSegmentedColormap, Colormap2D):
    """A full implementation of Dave Green's "cubehelix" for Matplotlib.

    Based on the FORTRAN 77 code provided in D.A. Green, 2011, BASI, 39, 289.
    http://adsabs.harvard.edu/abs/2011arXiv1108.5083G
    Also see:
    http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
    http://davidjohnstone.net/pages/cubehelix-gradient-picker
    User can adjust all parameters of the cubehelix algorithm. This enables much greater
    flexibility in choosing color maps. Default color map settings produce the standard cubehelix.
    Create color map in only blues by setting rot=0 and start=0. Create reverse (white to black)
    backwards through the rainbow once by setting rot=1 and reverse=True, etc. Furthermore, the
    algorithm was tuned, so that constant luminance values can be used (e.g. to create a truly
    isoluminant colorwheel). The `rot` parameter is also tuned to hold true for these cases.
    Of the here presented colorwheels, only this one manages to solely navigate through the L*=50
    plane, which can be seen here:
    https://upload.wikimedia.org/wikipedia/commons/2/21/Lab_color_space.png

    Parameters
    ----------
    start : scalar, optional
        Sets the starting position in the color space. 0=blue, 1=red,
        2=green. Defaults to 0.5.
    rot : scalar, optional
        The number of rotations through the rainbow. Can be positive
        or negative, indicating direction of rainbow. Negative values
        correspond to Blue->Red direction. Defaults to -1.5.
    gamma : scalar, optional
        The gamma correction for intensity. Defaults to 1.0.
    reverse : boolean, optional
        Set to True to reverse the color map. Will go from black to
        white. Good for density plots where shade~density. Defaults to False.
    nlev : scalar, optional
        Defines the number of discrete levels to render colors at.
        Defaults to 256.
    sat : scalar, optional
        The saturation intensity factor. Defaults to 1.2
        NOTE: this was formerly known as `hue` parameter
    minSat : scalar, optional
        Sets the minimum-level saturation. Defaults to 1.2.
    maxSat : scalar, optional
        Sets the maximum-level saturation. Defaults to 1.2.
    startHue : scalar, optional
        Sets the starting color, ranging from [0, 360], as in
        D3 version by @mbostock.
        NOTE: overrides values in start parameter.
    endHue : scalar, optional
        Sets the ending color, ranging from [0, 360], as in
        D3 version by @mbostock
        NOTE: overrides values in rot parameter.
    minLight : scalar, optional
        Sets the minimum lightness value. Defaults to 0.
    maxLight : scalar, optional
        Sets the maximum lightness value. Defaults to 1.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap object

    Revisions
    ---------
    2014-04 (@jradavenport) Ported from IDL version
    2014-04 (@jradavenport) Added kwargs to enable similar to D3 version,
                            changed name of `hue` parameter to `sat`.
    2016-11 (@jan.caron) Added support for isoluminant cubehelices while making sure
                         `rot` works as intended. Decoded the plane-vectors a bit.
    """

    _log = logging.getLogger(__name__ + '.ColormapCubehelix')

    def __init__(self, start=0.5, rot=-1.5, gamma=1.0, reverse=False, nlev=256,
                 minSat=1.2, maxSat=1.2, minLight=0., maxLight=1., **kwargs):
        self._log.debug('Calling __init__')
        # Override start and rot if startHue and endHue are set:
        if kwargs is not None:
            if 'startHue' in kwargs:
                start = (kwargs.get('startHue') / 360. - 1.) * 3.
            if 'endHue' in kwargs:
                rot = kwargs.get('endHue') / 360. - start / 3. - 1.
            if 'sat' in kwargs:
                minSat = kwargs.get('sat')
                maxSat = kwargs.get('sat')
        self.nlev = nlev
        # Set up the parameters:
        self.fract = np.linspace(minLight, maxLight, nlev)
        angle = 2.0 * np.pi * (start / 3.0 + rot * np.linspace(0, 1, nlev))
        self.fract = self.fract**gamma
        satar = np.linspace(minSat, maxSat, nlev)
        amp = np.asarray(satar * self.fract * (1. - self.fract) / 2)
        # Set RGB color coefficients (Luma is calculated in RGB Rec.601, so choose those),
        # the original version of Dave green used (0.30, 0.59, 0.11) and REc.709 is
        # c709 = (0.2126, 0.7152, 0.0722) but would not produce correct YPbPr Luma.
        c601 = (0.299, 0.587, 0.114)
        cr, cg, cb = c601
        cw = -0.90649  # Chosen to comply with Dave Greens implementation.
        k = -1.6158 / cr / cw  # k has to balance out cw so nothing gets out of RGB gamut (> 1).
        # Calculate the vectors v and w spanning the plane of constant perceived intensity.
        # v and w have to solve v x w = k(cr, cg, cb) (normal vector of the described plane) and
        # v * w = 0 (scalar product, v and w have to be perpendicular).
        # 6 unknown and 4 equations --> Chose wb = 0 and wg = cw (constant).
        v = np.array((k * cr ** 2 * cb / (cw * (cr ** 2 + cg ** 2)),
                      k * cr * cg * cb / (cw * (cr ** 2 + cg ** 2)), -k * cr / cw))
        w = np.array((-cw * cg / cr, cw, 0))
        # Calculate components:
        self.red = self.fract + amp * (v[0] * np.cos(angle) + w[0] * np.sin(angle))
        self.grn = self.fract + amp * (v[1] * np.cos(angle) + w[1] * np.sin(angle))
        self.blu = self.fract + amp * (v[2] * np.cos(angle) + w[2] * np.sin(angle))
        # Original formulas with original v and w:
        # self.red = self.fract + amp * (-0.14861 * np.cos(angle) + 1.78277 * np.sin(angle))
        # self.grn = self.fract + amp * (-0.29227 * np.cos(angle) - 0.90649 * np.sin(angle))
        # self.blu = self.fract + amp * (1.97294 * np.cos(angle))
        # Find where RBG are outside the range [0,1], clip:
        self.red = np.clip(self.red, 0, 1)
        self.grn = np.clip(self.grn, 0, 1)
        self.blu = np.clip(self.blu, 0, 1)
        # Optional color reverse:
        if reverse is True:
            self.red = self.red[::-1]
            self.blu = self.blu[::-1]
            self.grn = self.grn[::-1]
        # Put in to tuple & dictionary structures needed:
        rr, bb, gg = [], [], []
        for k in range(0, int(nlev)):
            rr.append((float(k) / (nlev - 1), self.red[k], self.red[k]))
            bb.append((float(k) / (nlev - 1), self.blu[k], self.blu[k]))
            gg.append((float(k) / (nlev - 1), self.grn[k], self.grn[k]))
        cdict = {'red': rr, 'blue': bb, 'green': gg}
        super().__init__('cubehelix', cdict, N=256)
        self._log.debug('Created ' + str(self))


class ColormapPerception(colors.LinearSegmentedColormap, Colormap2D):
    """A perceptual colormap based on face-based luminance matching.

    Based on a publication by Kindlmann et. al.
    http://www.cs.utah.edu/~gk/papers/vis02/FaceLumin.pdf
    This colormap tries to achieve an isoluminant perception by using a list of colors acquired
    through face recognition studies. It is a lot better than the HLS colormap, but still not
    completely isoluminant (despite its name). Also it appears a bit dark.

    """

    _log = logging.getLogger(__name__ + '.ColormapPerception')

    CDICT = {'red': [(0/6, 0.847, 0.847),
                     (1/6, 0.527, 0.527),
                     (2/6, 0.000, 0.000),
                     (3/6, 0.000, 0.000),
                     (4/6, 0.316, 0.316),
                     (5/6, 0.718, 0.718),
                     (6/6, 0.847, 0.847)],

             'green': [(0/6, 0.057, 0.057),
                       (1/6, 0.527, 0.527),
                       (2/6, 0.592, 0.592),
                       (3/6, 0.559, 0.559),
                       (4/6, 0.316, 0.316),
                       (5/6, 0.000, 0.000),
                       (6/6, 0.057, 0.057)],

             'blue': [(0/6, 0.057, 0.057),
                      (1/6, 0.000, 0.000),
                      (2/6, 0.000, 0.000),
                      (3/6, 0.559, 0.559),
                      (4/6, 0.991, 0.991),
                      (5/6, 0.718, 0.718),
                      (6/6, 0.057, 0.057)]}

    def __init__(self):
        self._log.debug('Calling __init__')
        super().__init__('perception', self.CDICT, N=256)
        self._log.debug('Created ' + str(self))


class ColormapHLS(colors.ListedColormap, Colormap2D):
    """Colormap subclass for encoding directions with colors.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.ListedColormap`
    class. The class follows the HSL ('hue', 'saturation', 'lightness') 'Double Hexcone' Model
    with the saturation always set to 1 (moving on the surface of the color
    cylinder) with a lightness of 0.5 (full color). The three prime colors (`rgb`) are spaced
    equidistant with 120° space in between, according to a triadic arrangement.
    Even though the lightness is constant in the plane, the luminance (which is a weighted sum
    of the RGB components which encompasses human perception) is not, which can lead to
    artifacts like reliefs. Converting the map to a grayscale show spokes at the secondary colors.
    For more information see:
    https://vis4.net/blog/posts/avoid-equidistant-hsv-colors/
    http://www.workwithcolor.com/color-luminance-2233.htm
    http://blog.asmartbear.com/color-wheels.html

    """

    _log = logging.getLogger(__name__ + '.ColormapHLS')

    def __init__(self):
        self._log.debug('Calling __init__')
        h = np.linspace(0, 1, 256)
        l = 0.5 * np.ones_like(h)  # noqa: E741
        s = np.ones_like(h)
        r, g, b = np.vectorize(colorsys.hls_to_rgb)(h, l, s)
        colors = [(r[i], g[i], b[i]) for i in range(len(r))]
        super().__init__(colors, 'hls', N=256)
        self._log.debug('Created ' + str(self))


class ColormapClassic(colors.LinearSegmentedColormap, Colormap2D):
    """Colormap subclass for encoding directions with colors.

    This class is a subclass of the :class:`~matplotlib.pyplot.colors.LinearSegmentedColormap`
    class. The class follows the HSL ('hue', 'saturation', 'lightness') 'Double
    Hexcone' Model with the saturation always set to 1 (moving on the surface of the color
    cylinder) with a luminance of 0.5 (full color). The colors follow a tetradic arrangement with
    four colors (red, green, blue and yellow) arranged with 90° spacing in between.

    """

    _log = logging.getLogger(__name__ + '.ColormapClassic')

    CDICT = {'red': [(0/4, 1.0, 1.0),
                     (1/4, 0.0, 0.0),
                     (2/4, 0.0, 0.0),
                     (3/4, 1.0, 1.0),
                     (4/4, 1.0, 1.0)],

             'green': [(0/4, 0.0, 0.0),
                       (1/4, 0.0, 0.0),
                       (2/4, 1.0, 1.0),
                       (3/4, 1.0, 1.0),
                       (4/4, 0.0, 0.0)],

             'blue': [(0/4, 0.0, 0.0),
                      (1/4, 1.0, 1.0),
                      (2/4, 0.0, 0.0),
                      (3/4, 0.0, 0.0),
                      (4/4, 0.0, 0.0)]}

    def __init__(self):
        self._log.debug('Calling __init__')
        super().__init__('classic', self.CDICT, N=256)
        self._log.debug('Created ' + str(self))


def interpolate_color(fraction, start, end):
    """Interpolate linearly between two color tuples (e.g. RGB).

    Parameters
    ----------
    fraction: float or :class:`~numpy.ndarray`
        Interpolation fraction between 0 and 1, which determines the position of the
        interpolation between `start` and `end`.
    start: tuple (N=3) or :class:`~numpy.ndarray`
        Start of the interpolation as a tuple of three numbers or a numpy array, where the last
        dimension should have length 3 and contain the color tuples.
    end: tuple (N=3) or :class:`~numpy.ndarray`
        End of the interpolation as a tuple of three numbers or a numpy array, where the last
        dimension should have length 3 and contain the color tuples.

    Returns
    -------
    result: tuple (N=3) or :class:`~numpy.ndarray`
        Result of the interpolation as a tuple of three numbers or a numpy array, where the
        last dimension should has length 3 and contains the color tuples.

    """
    _log.debug('Calling interpolate_color')
    start, end = np.asarray(start), np.asarray(end)
    r1 = start[..., 0] + (end[..., 0] - start[..., 0]) * fraction
    r2 = start[..., 1] + (end[..., 1] - start[..., 1]) * fraction
    r3 = start[..., 2] + (end[..., 2] - start[..., 2]) * fraction
    return r1, r2, r3


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
    Normalize and visualize ``result`` with ``colormap`` and save it to an image
    with parameters ``save_kwargs``.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    colormap : matplotlib colormap or None
        colormap used for visualizing intensity values, defaults to ColormapCubehelix()

    save_kwargs : dict or None
        dict of kwargs passed to Pillow when saving the image, can be used to set
        the file format, quality, ...

    Returns
    -------

    BytesIO
        a buffer containing the result image (as PNG/JPG/... depending on save_kwargs)
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
