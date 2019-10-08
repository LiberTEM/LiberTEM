import functools

import numpy as np
from skimage.feature import peak_local_max
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from libertem.udf import UDF
import libertem.masks as masks
from libertem.job.masks import MaskContainer
from libertem.utils import frame_peaks

import libertem.analysis.gridmatching as grm

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


def get_peaks(sum_result, match_pattern: MatchPattern, num_peaks):
    '''
    Find peaks of the correlation between :code:`sum_result` and :code:`match_pattern`

    This can then be used in :meth:`~libertem.analysis.fullmatch.FullMatcher.full_match`
    to extract grid parameters, :meth:`~libertem.udf.blobfinder.run_fastcorrelation` to find
    the position in each frame or to construct a mask to extract feature vectors with
    :meth:`~libertem.udf.blobfinder.feature_vector`.

    Parameters
    ----------

    sum_result: numpy.ndarray
        2D result frame as correlation input
    match_pattern : MatchPattern
        Instance of :class:`~libertem.udf.blobfinder.MatchPattern` to correlate
        :code:`sum_result` with
    num_peaks : int
        Number of peaks to find

    Example
    -------

    >>> frame, _, _ = libertem.utils.generate.cbed_frame(radius=4)
    >>> pattern = RadialGradient(radius=4)
    >>> peaks = get_peaks(frame[0], pattern, 7)
    >>> print(peaks)
    [[64 64]
     [64 80]
     [80 80]
     [80 64]
     [48 80]
     [48 64]
     [64 96]]
    '''
    spec_mask = match_pattern.get_template(sig_shape=sum_result.shape)
    spec_sum = fft.rfft2(sum_result)
    corrspec = spec_mask * spec_sum
    corr = fft.fftshift(fft.irfft2(corrspec))
    peaks = peak_local_max(corr, num_peaks=num_peaks)
    return peaks


def refine_center(center, r, corrmap):
    (y, x) = center
    s = corrmap.shape
    r = min(r, y, x, s[0] - y - 1, s[1] - x - 1)
    if r <= 0:
        return (y, x)
    else:
        # FIXME See and compare with Extension of Phase Correlation to Subpixel Registration
        # Hassan Foroosh
        # That one or a close/similar/cited one
        cutout = corrmap[y-r:y+r+1, x-r:x+r+1]
        m = np.min(cutout)
        ry, rx = nd.measurements.center_of_mass(cutout - m)
        refined_y = y + ry - r
        refined_x = x + rx - r
        # print(y, x, refined_y, refined_x, "\n", cutout)
        return np.array((refined_y, refined_x))


def peak_elevation(center, corrmap, height, r_min=1.5, r_max=np.float('inf')):
    '''
    Return the slope of the tightest cone around center with height height
        that touches corrmap between r_min and r_max.

    The correlation of two disks -- mask and perfect diffraction spot -- has the shape of a cone.

    The height is provided as a parameter since center can be float values from refinement
    and the height value is conveniently available from the calling function.

    The function's return value correlates with the quality of a correlation. Higher slope
    means a strong peak and
    no side maxima, while weak signal or side maxima lead to a flatter slope.

    r_min masks out a small local plateau around the peak that would distort and dominate
    the calculation.

    r_max can mask out neighboring peaks if a large area with several legitimate peaks is
    corellated.
    '''
    peak_y, peak_x = center
    (size_y, size_x) = corrmap.shape
    y, x = np.mgrid[0:size_y, 0:size_x]

    dist = np.sqrt((y - peak_y)**2 + (x - peak_x)**2)
    select = (dist >= r_min) * (dist < r_max)
    diff = height - corrmap[select]

    return max(0, np.min(diff / dist[select]))


def do_correlation(template, crop_part):
    spec_part = fft.rfft2(crop_part)
    corrspec = template * spec_part
    corr = fft.fftshift(fft.irfft2(corrspec))
    return evaluate_correlation(corr)


def evaluate_correlation(corr):
    center = np.unravel_index(np.argmax(corr), corr.shape)
    refined = np.array(refine_center(center, 2, corr), dtype='float32')
    height = np.float32(corr[center])
    elevation = np.float32(peak_elevation(refined, corr, height))
    center = np.array(center, dtype='u2')
    return center, refined, height, elevation


def log_scale(data, out):
    return np.log(data - np.min(data) + 1, out=out)


def crop_disks_from_frame(peaks, frame, match_pattern: MatchPattern):
    crop_size = match_pattern.get_crop_size()
    for peak in peaks:
        slice_ = (
            slice(max(peak[0] - crop_size, 0), min(peak[0] + crop_size, frame.shape[0])),
            slice(max(peak[1] - crop_size, 0), min(peak[1] + crop_size, frame.shape[1])),
        )

        # also calculate the slice into the crop buffer, which may be smaller
        # than the buffer if we are operating on peaks near edges:
        size = (
            slice_[0].stop - slice_[0].start,
            slice_[1].stop - slice_[1].start,
        )
        crop_buf_slice = (
            slice(
                max(crop_size - peak[0], 0),
                max(crop_size - peak[0], 0) + size[0]
            ),
            slice(
                max(crop_size - peak[1], 0),
                max(crop_size - peak[1], 0) + size[1]
            ),
        )
        yield frame[slice_], crop_buf_slice


def _shift(relative_center, anchor, crop_size):
    return relative_center + anchor - [crop_size, crop_size]


class CorrelationUDF(UDF):
    '''
    Abstract base class for peak correlation implementations
    '''
    def get_result_buffers(self):
        """
        we 'declare' what kind of result buffers we need, without concrete shapes

        concrete shapes come later, either for partition or whole dataset
        """
        num_disks = len(self.params.peaks)

        return {
            'centers': self.buffer(
                kind="nav", extra_shape=(num_disks, 2), dtype="u2"
            ),
            'refineds': self.buffer(
                kind="nav", extra_shape=(num_disks, 2), dtype="float32"
            ),
            'peak_values': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="float32",
            ),
            'peak_elevations': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="float32",
            ),
        }


class FastCorrelationUDF(CorrelationUDF):
    '''
    Fourier-based fast correlation-based refinement of peak positions within a search frame
    for each peak.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        match_pattern : MatchPattern
            Instance of :class:`~libertem.udf.blobfinder.MatchPattern`
        '''
        super().__init__(*args, **kwargs)

    def get_task_data(self):
        mask = self.params.match_pattern
        crop_size = mask.get_crop_size()
        template = mask.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
        crop_buf = zeros((2 * crop_size, 2 * crop_size), dtype="float32")
        kwargs = {
            'crop_buf': crop_buf,
            'template': template,
        }
        return kwargs

    def process_frame(self, frame):
        match_pattern = self.params.match_pattern
        peaks = self.params.peaks
        crop_buf = self.task_data.crop_buf
        crop_size = match_pattern.get_crop_size()
        r = self.results
        for disk_idx, (crop_part, crop_buf_slice) in enumerate(
                crop_disks_from_frame(peaks=peaks, frame=frame, match_pattern=match_pattern)):

            crop_buf[:] = 0  # FIXME: we need to do this only for edge cases
            log_scale(crop_part, out=crop_buf[crop_buf_slice])
            center, refined, peak_value, peak_elevation = do_correlation(
                self.task_data.template, crop_buf)
            abs_center = _shift(center, peaks[disk_idx], crop_size).astype('u2')
            abs_refined = _shift(refined, peaks[disk_idx], crop_size).astype('float32')
            r.centers[disk_idx] = abs_center
            r.refineds[disk_idx] = abs_refined
            r.peak_values[disk_idx] = peak_value
            r.peak_elevations[disk_idx] = peak_elevation

    def postprocess(self):
        pass


class SparseCorrelationUDF(CorrelationUDF):
    '''
    Direct correlation using sparse matrices

    This method allows to adjust the number of correlation steps independent of the template size.
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        match_pattern : MatchPattern
            Instance of :class:`~libertem.udf.blobfinder.MatchPattern`
        steps : int
            The template is correlated with 2 * steps + 1 symmetrically around the peak position
            in x and y direction. This defines the maximum shift that can be
            detected. The number of calculations grows with the square of this value, that means
            keeping this as small as the data allows speeds up the calculation.
        '''
        super().__init__(*args, **kwargs)

    def get_result_buffers(self):
        super_buffers = super().get_result_buffers()
        num_disks = len(self.params.peaks)
        steps = self.params.steps * 2 + 1
        my_buffers = {
            'corr': self.buffer(
                kind="nav", extra_shape=(num_disks * steps**2,), dtype="float32"
            ),
        }
        super_buffers.update(my_buffers)
        return super_buffers

    def get_task_data(self):
        match_pattern = self.params.match_pattern
        crop_size = match_pattern.get_crop_size()
        size = (2 * crop_size + 1, 2 * crop_size + 1)
        template = match_pattern.get_mask(sig_shape=size)
        steps = self.params.steps
        peak_offsetY, peak_offsetX = np.mgrid[-steps:steps + 1, -steps:steps + 1]

        offsetY = self.params.peaks[:, 0, np.newaxis, np.newaxis] + peak_offsetY - crop_size
        offsetX = self.params.peaks[:, 1, np.newaxis, np.newaxis] + peak_offsetX - crop_size

        offsetY = offsetY.flatten()
        offsetX = offsetX.flatten()

        stack = functools.partial(
            masks.sparse_template_multi_stack,
            mask_index=range(len(offsetY)),
            offsetX=offsetX,
            offsetY=offsetY,
            template=template,
            imageSizeX=self.meta.dataset_shape.sig[1],
            imageSizeY=self.meta.dataset_shape.sig[0]
        )
        # CSC matrices in combination with transposed data are fastest
        container = MaskContainer(mask_factories=stack, dtype=np.float32,
            use_sparse='scipy.sparse.csc')

        kwargs = {
            'mask_container': container,
            'crop_size': crop_size,
        }
        return kwargs

    def process_tile(self, tile):
        tile_slice = self.meta.slice
        c = self.task_data.mask_container
        tile_t = np.zeros(
            (np.prod(tile.shape[1:]), tile.shape[0]),
            dtype=tile.dtype
        )
        log_scale(tile.reshape((tile.shape[0], -1)).T, out=tile_t)

        sl = c.get(key=tile_slice, transpose=False)
        self.results.corr[:] += sl.dot(tile_t).T

    def postprocess(self):
        steps = 2 * self.params.steps + 1
        corrmaps = self.results.corr.reshape((
            -1,  # frames
            len(self.params.peaks),  # peaks
            steps,  # Y steps
            steps,  # X steps
        ))
        peaks = self.params.peaks
        r = self.results
        for f in range(corrmaps.shape[0]):
            for p in range(len(self.params.peaks)):
                corr = corrmaps[f, p]
                center, refined, peak_value, peak_elevation = evaluate_correlation(corr)
                abs_center = _shift(center, peaks[p], self.params.steps).astype('u2')
                abs_refined = _shift(refined, peaks[p], self.params.steps).astype('float32')
                r.centers[f, p] = abs_center
                r.refineds[f, p] = abs_refined
                r.peak_values[f, p] = peak_value
                r.peak_elevations[f, p] = peak_elevation


def run_fastcorrelation(ctx, dataset, peaks, match_pattern: MatchPattern, roi=None):
    peaks = peaks.astype(np.int)
    udf = FastCorrelationUDF(peaks=peaks, match_pattern=match_pattern)
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)


def run_blobfinder(ctx, dataset, match_pattern: MatchPattern, num_peaks, roi=None):
    sum_analysis = ctx.create_sum_analysis(dataset=dataset)
    sum_result = ctx.run(sum_analysis)

    sum_result = log_scale(sum_result.intensity.raw_data, out=None)
    peaks = get_peaks(
        sum_result=sum_result,
        match_pattern=match_pattern,
        num_peaks=num_peaks,
    )

    pass_2_results = run_fastcorrelation(
        ctx=ctx,
        dataset=dataset,
        peaks=peaks,
        match_pattern=match_pattern,
        roi=roi
    )

    return (sum_result, pass_2_results['centers'],
        pass_2_results['refineds'], pass_2_results['peak_values'],
        pass_2_results['peak_elevations'], peaks)


class RefinementMixin():
    '''
    To be combined with a CorrelationUDF using multiple inheritance.

    The mixin must come before the UDF in the inheritance list.

    It adds buffers zero, a, b, selector. A subclass of this mixin implements a
    process_frame() method that will first call the superclass process_frame(), likely the
    CorrelationUDF's, and expects that this will populate the CorrelationUDF result buffers.
    It then calculates a refinement of start_zero, start_a and start_b based on the
    correlation result and populates its own result buffers with this refinement result.

    This allows combining arbitrary implementations of correlation-based matching with
    arbitrary implementations of the refinement by declaring an ad-hoc class that inherits from one
    subclass of RefinementMixin and one subclass of CorrelationUDF.
    '''
    def get_result_buffers(self):
        super_buffers = super().get_result_buffers()
        num_disks = len(self.params.peaks)
        my_buffers = {
            'zero': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'a': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'b': self.buffer(
                kind="nav", extra_shape=(2,), dtype="float32"
            ),
            'selector': self.buffer(
                kind="nav", extra_shape=(num_disks,), dtype="bool"
            ),
            'error': self.buffer(
                kind="nav", dtype="float32",
            ),
        }
        super_buffers.update(my_buffers)
        return super_buffers

    def apply_match(self, index,  match):
        r = self.results
        # We cast from float64 to float32 here
        r.zero[index] = match.zero
        r.a[index] = match.a
        r.b[index] = match.b
        r.selector[index] = match.selector
        r.error[index] = match.error


class FastmatchMixin(RefinementMixin):
    '''
    Refinement using :meth:`~libertem.analysis.gridmatching.Matcher.fastmatch`
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        matcher : libertem.analysis.gridmatching.Matcher
            Instance of :class:`~libertem.analysis.gridmatching.Matcher`
        start_zero : numpy.ndarray
            Approximate value (y, x) in px for "zero" point (origin, zero order peak)
        start_a : numpy.ndarray
            Approximate value (y, x) in px for "a" vector.
        start_b : numpy.ndarray
            Approximate value (y, x) in px for "b" vector.
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
        super().postprocess()
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = p.matcher.fastmatch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                zero=p.start_zero,
                a=p.start_a,
                b=p.start_b,
            )
            self.apply_match(index, match)


class AffineMixin(RefinementMixin):
    '''
    Refinement using :meth:`~libertem.analysis.gridmatching.Matcher.affinematch`
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------

        matcher : libertem.analysis.gridmatching.Matcher
            Instance of :class:`~libertem.analysis.gridmatching.Matcher`
        indices : numpy.ndarray
            List of indices [(h1, k1), (h2, k2), ...] of all peaks. The indices can be
            non-integer and relative to any base vectors, including virtual ones like
            (1, 0); (0, 1). See documentation of
            :meth:`~libertem.analysis.gridmatching.Matcher.affinematch` for details.
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
        super().postprocess()
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = p.matcher.affinematch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                indices=p.indices,
            )
            self.apply_match(index, match)


def run_refine(
        ctx, dataset, zero, a, b, match_pattern: MatchPattern, matcher: grm.Matcher,
        correlation='fast', match='fast', indices=None, steps=5, roi=None):
    '''
    Refine the given lattice for each frame by calculating approximate peak positions and refining
    them for each frame by using the blobcorrelation and methods of
    :class:`~libertem.analysis.gridmatching.Matcher`.

    Parameters
    ----------

    ctx : libertem.api.Context
        Instance of a LiberTEM :class:`~libertem.api.Context`
    dataset : libertem.io.dataset.base.DataSet
        Instance of a :class:`~libertem.io.dataset.base.DataSet`
    zero : numpy.ndarray
        Approximate value for "zero" point (y, x) in px (origin, zero order peak)
    a : numpy.ndarray
        Approximate value for "a" vector (y, x) in px.
    b : numpy.ndarray
        Approximate value for "b" vector (y, x) in px.
    match_pattern : MatchPattern
        Instance of :class:`~MatchPattern`
    matcher : libertem.analysis.gridmatching.Matcher
        Instance of :class:`~libertem.analysis.gridmatching.Matcher` to perform the matching
    correlation : {'fast', 'sparse'}, optional
        'fast' or 'sparse' to select :class:`~FastCorrelationUDF` or :class:`~SparseCorrelationUDF`
    match : {'fast', 'affine'}, optional
        'fast' or 'affine' to select :class:`~FastmatchMixin` or :class:`~AffineMixin`
    indices : numpy.ndarray, optional
        Indices to refine. This is trimmed down to positions within the frame.
        As a convenience, for the indices parameter this function accepts both shape
        (n, 2) and (2, n, m) so that numpy.mgrid[h:k, i:j] works directly to specify indices.
        This saves boilerplate code when using this function. Default: numpy.mgrid[-10:10, -10:10].
    steps : int, optional
        Only for correlation == 'sparse': Correlation steps.
        See :meth:`~SparseCorelationUDF.__init__` for details.
    roi : numpy.ndarray, optional
        ROI for :meth:`~libertem.api.Context.run_udf`

    Returns
    -------

    Tuple[Dict[str, BufferWrapper], numpy.ndarray]
        :code:`(result, used_indices)` where :code:`result` is a :code:`dict`
        mapping buffer names to result buffers based on

        .. code-block:: python

            {
                'centers': BufferWrapper(
                    kind="nav", extra_shape=(num_disks, 2), dtype="u2"
                ),
                'refineds': BufferWrapper(
                    kind="nav", extra_shape=(num_disks, 2), dtype="float32"
                ),
                'peak_values': BufferWrapper(
                    kind="nav", extra_shape=(num_disks,), dtype="float32"
                ),
                'peak_elevations': BufferWrapper(
                    kind="nav", extra_shape=(num_disks,), dtype="float32"
                ),
                'zero': BufferWrapper(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'a': BufferWrapper(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'b': BufferWrapper(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'selector': BufferWrapper(
                    kind="nav", extra_shape=(num_disks,), dtype="bool"
                ),
                'error': self.buffer(
                    kind="nav", dtype="float32",
                ),
            }

        and :code:`used_indices` are the indices that were within the frame.

    Examples
    --------

    >>> dataset = ctx.load(
    ...     filetype="memory",
    ...     data=np.zeros(shape=(2, 2, 128, 128), dtype=np.float32)
    ... )
    >>> (result, used_indices) = run_refine(
    ...     ctx, dataset,
    ...     zero=(64, 64), a=(1, 0), b=(0, 1),
    ...     match_pattern=RadialGradient(radius=4),
    ...     matcher=grm.Matcher()
    ... )
    >>> result['centers'].data  #doctest: +ELLIPSIS
    array(...)

    '''
    if indices is None:
        indices = np.mgrid[-10:11, -10:11]

    (fy, fx) = tuple(dataset.shape.sig)

    indices, peaks = frame_peaks(
        fy=fy, fx=fx, zero=zero, a=a, b=b,
        r=match_pattern.search, indices=indices
    )
    peaks = peaks.astype('int')

    if correlation == 'fast':
        method = FastCorrelationUDF
    elif correlation == 'sparse':
        method = SparseCorrelationUDF
    else:
        raise ValueError(
            "Unknown correlation method %s. Supported are 'fast' and 'sparse'" % correlation
        )

    if match == 'affine':
        mixin = AffineMixin
    elif match == 'fast':
        mixin = FastmatchMixin
    else:
        raise ValueError(
            "Unknown match method %s. Supported are 'fast' and 'affine'" % match
        )

    # The inheritance order matters: FIRST the mixin, which calls
    # the super class methods.
    class MyUDF(mixin, method):
        pass

    udf = MyUDF(
        peaks=peaks,
        indices=indices,
        start_zero=zero,
        start_a=a,
        start_b=b,
        match_pattern=match_pattern,
        matcher=matcher,
        steps=steps
    )

    result = ctx.run_udf(
        dataset=dataset,
        udf=udf,
        roi=roi,
    )
    return (result, indices)


def feature_vector(imageSizeX, imageSizeY, peaks, match_pattern: MatchPattern):
    '''
    This function generates a sparse mask stack to extract a feature vector.

    A match template based on the parameters in :code:`parameters` is placed at
    each peak position in an individual mask layer. This mask stack can then
    be used in :meth:`~libertem.api.Context.create_mask_job` to generate a feature vector for each
    frame.

    Summing up the mask stack along the first axis generates a mask that can be used for virtual
    darkfield imaging of all peaks together.

    Parameters
    ----------

    imageSizeX,imageSizeY : int
        Frame size in px
    peaks : numpy.ndarray
        Peak positions in px as numpy.ndarray of shape (n, 2) with integer type
    match_pattern : MatchPattern
        Instance of :class:`~MatchPattern`
    '''
    crop_size = match_pattern.get_crop_size()
    return masks.sparse_template_multi_stack(
        mask_index=range(len(peaks)),
        offsetX=peaks[:, 1] - crop_size,
        offsetY=peaks[:, 0] - crop_size,
        template=match_pattern.get_mask((2*crop_size + 1, 2*crop_size + 1)),
        imageSizeX=imageSizeX,
        imageSizeY=imageSizeY,
    )


def visualize_frame(ctx, ds, result, indices, r, y, x, axes, colors=None, stretch=10):
    '''
    Visualize the refinement of a specific frame in matplotlib axes
    '''
    # Get the frame from the dataset
    get_sample_frame = ctx.create_pick_analysis(dataset=ds, y=y, x=x)
    sample_frame = ctx.run(get_sample_frame)

    d = sample_frame[0].raw_data

    pcm = axes.imshow(np.log(d - np.min(d) + 1))

    refined = result['refineds'].data[y, x]
    elevations = result['peak_elevations'].data[y, x]
    selector = result['selector'].data[y, x]

    max_elevation = np.max(elevations)

    # Calclate the best fit positions to compare with the
    # individual peak positions.
    # A difference between best fit and individual peaks highlights outliers.
    calculated = grm.calc_coords(
        zero=result['zero'].data[y, x],
        a=result['a'].data[y, x],
        b=result['b'].data[y, x],
        indices=indices
    )

    paint_markers(
        axes=axes,
        r=r,
        refined=refined,
        normalized_elevations=elevations/max_elevation,
        calculated=calculated,
        selector=selector,
        zero=result['zero'].data[y, x],
        a=result['a'].data[y, x],
        b=result['b'].data[y, x],
        colors=colors,
        stretch=stretch,
    )
    return pcm


def paint_markers(axes, r, refined, normalized_elevations, calculated, selector, zero, a, b,
        colors=None, stretch=10):
    if colors is None:
        colors = {
            'marker': 'w',
            'arrow': 'r',
            'missing': 'r',
            'a': 'b',
            'b': 'g',
        }

    axes.arrow(*np.flip(zero), *(np.flip(a)), color=colors['a'])
    axes.arrow(*np.flip(zero), *(np.flip(b)), color=colors['b'])

    # Plot markers for the individual peak positions.
    # The alpha channel represents the peak elevation, which is used as a weight in the fit.
    for i in range(len(refined)):
        p = np.flip(refined[i])
        a = max(0, normalized_elevations[i])
        p0 = np.flip(calculated[i])
        if selector[i]:
            axes.add_artist(plt.Circle(p, r, color=colors['marker'], fill=False, alpha=a))
            axes.add_artist(plt.Circle(p0, 1, color=colors['arrow'], fill=True, alpha=a))
            axes.arrow(*p0, *(p-p0)*stretch, color=colors['arrow'], alpha=a)
        else:
            (yy, xx) = calculated[i]
            xy = (xx - r, yy - r)
            axes.add_artist(plt.Rectangle(xy, 2*r, 2*r, color=colors['missing'], fill=False))
