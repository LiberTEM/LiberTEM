import numpy as np

import libertem.masks as masks


# FIXME There's work on flexible FFT backends in scipy
# https://github.com/scipy/scipy/wiki/GSoC-2019-project-ideas#revamp-scipyfftpack
# and discussions about pyfftw performance vs other implementations
# https://github.com/pyFFTW/pyFFTW/issues/264
# For that reason we shoud review the state of Python FFT implementations
# regularly and adapt our choices accordingly
try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros


class MatchPattern:
    '''
    Abstract base class for correlation patterns.

    This class provides an API to provide a template for fast correlation-based peak finding.
    '''
    def __init__(self, search):
        '''
        Parameters
        ----------

        search : float
            Range from the center point in px to include in the correlation, defining the size
            of the square correlation pattern.
            Will be ceiled to the next int for performing the correlation.
        '''
        self.search = search

    def get_crop_size(self):
        return int(np.ceil(self.search))

    def get_mask(self, sig_shape):
        raise NotImplementedError

    def get_template(self, sig_shape):
        return fft.rfft2(self.get_mask(sig_shape))


class Circular(MatchPattern):
    '''
    Circular pattern with radius :code:`radius`.

    This pattern is useful for constructing feature vectors using
    :meth:`~libertem.udf.blobfinder.feature_vector`.

    .. versionadded:: 0.3.0
    '''
    def __init__(self, radius, search=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation, 2x radius by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            search = 2*radius
        self.radius = radius
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.circular(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius,
            antialiased=True,
        )


class RadialGradient(MatchPattern):
    '''
    Radial gradient from zero in the center to one at :code:`radius`.

    This pattern rejects the influence of internal intensity variations of the CBED disk.
    '''
    def __init__(self, radius, search=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation, 2x radius by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            search = 2*radius
        self.radius = radius
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.radial_gradient(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius,
            antialiased=True,
        )


class BackgroundSubtraction(MatchPattern):
    '''
    Solid circular disk surrounded with a balancing negative area

    This pattern rejects background and avoids false positives at positions between peaks
    '''
    def __init__(self, radius, search=None, radius_outer=None):
        '''
        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation.
            :code:`max(2*radius, radius_outer)` by default.
            Defining the size of the square correlation pattern.
        radius_outer : float, optional
            Radius of the negative region in px. 1.5x radius by default.
        '''
        if radius_outer is None:
            radius_outer = radius * 1.5
        if search is None:
            search = max(2*radius, radius_outer)
        self.radius = radius
        self.radius_outer = radius_outer
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        return masks.background_subtraction(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius_outer,
            radius_inner=self.radius,
            antialiased=True
        )


class UserTemplate(MatchPattern):
    '''
    User-defined template
    '''
    def __init__(self, template, search=None):
        '''
        Parameters
        ----------

        template : numpy.ndarray
            Correlation template as 2D numpy.ndarray
        search : float, optional
            Range from the center point in px to include in the correlation.
            Half diagonal of the template by default.
            Defining the size of the square correlation pattern.
        '''
        if search is None:
            # Half diagonal
            search = np.sqrt(template.shape[0]**2 + template.shape[1]**2) / 2
        self.template = template
        super().__init__(search=search)

    def get_mask(self, sig_shape):
        result = np.zeros((sig_shape), dtype=self.template.dtype)
        dy, dx = sig_shape
        ty, tx = self.template.shape

        left = dx / 2 - tx / 2
        top = dy / 2 - ty / 2

        r_left = max(0, left)
        r_top = max(0, top)

        t_left = max(0, -left)
        t_top = max(0, -top)

        crop_x = r_left - left
        crop_y = r_top - top

        h = int(ty - 2*crop_y)
        w = int(tx - 2*crop_x)

        r_left = int(r_left)
        r_top = int(r_top)
        t_left = int(t_left)
        t_top = int(t_top)

        result[r_top:r_top + h, r_left:r_left + w] = \
            self.template[t_top:t_top + h, t_left:t_left + w]
        return result


class RadialGradientBackgroundSubtraction(UserTemplate):
    '''
    Combination of radial gradient with background subtraction
    '''
    def __init__(self, radius, search=None, radius_outer=None, delta=1, radial_map=None):
        '''
        See :meth:`~libertem.masks.radial_gradient_background_subtraction` for details.

        Parameters
        ----------

        radius : float
            Radius of the circular pattern in px
        search : float, optional
            Range from the center point in px to include in the correlation.
            :code:`max(2*radius, radius_outer)` by default
            Defining the size of the square correlation pattern.
        radius_outer : float, optional
            Radius of the negative region in px. 1.5x radius by default.
        delta : float, optional
            Width of the transition region between positive and negative in px
        radial_map : numpy.ndarray, optional
            Radius value of each pixel in px. This can be used to distort the shape as needed
            or work in physical coordinates instead of pixels.
            A suitable map can be generated with :meth:`libertem.masks.polar_map`.

        Example
        -------

        >>> import matplotlib.pyplot as plt

        >>> (radius, phi) = libertem.masks.polar_map(
        ...     centerX=64, centerY=64,
        ...     imageSizeX=128, imageSizeY=128,
        ...     stretchY=2., angle=np.pi/4
        ... )

        >>> template = RadialGradientBackgroundSubtraction(
        ...     radius=30, radial_map=radius)

        >>> # This shows an elliptical template that is stretched
        >>> # along the 45 ° bottom-left top-right diagonal
        >>> plt.imshow(template.get_mask(sig_shape=(128, 128)))
        <matplotlib.image.AxesImage object at ...>
        >>> plt.show() # doctest: +SKIP
        '''
        if radius_outer is None:
            radius_outer = radius * 1.5
        if search is None:
            search = max(2*radius, radius_outer)
        if radial_map is None:
            r = max(radius, radius_outer)
            radial_map, _ = masks.polar_map(
                centerX=r + 1,
                centerY=r + 1,
                imageSizeX=int(np.ceil(2*r + 2)),
                imageSizeY=int(np.ceil(2*r + 2)),
            )
        self.radius = radius
        self.radius_outer = radius_outer
        self.delta = delta
        self.radial_map = radial_map
        template = masks.radial_gradient_background_subtraction(
            r=self.radial_map,
            r0=self.radius,
            r_outer=self.radius_outer,
            delta=self.delta
        )
        super().__init__(template=template, search=search)

    def get_mask(self, sig_shape):
        # Recalculate in case someone has changed parameters
        self.template = masks.radial_gradient_background_subtraction(
            r=self.radial_map,
            r0=self.radius,
            r_outer=self.radius_outer,
            delta=self.delta
        )
        return super().get_mask(sig_shape)
