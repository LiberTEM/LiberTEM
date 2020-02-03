import functools
import os

import numpy as np
from skimage.feature import peak_local_max
import numba

from libertem.udf import UDF
import libertem.masks as masks
from libertem.common.container import MaskContainer

from .patterns import MatchPattern

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

# Necessary to work with JIT disabled for coverage and testing purposes
# https://github.com/LiberTEM/LiberTEM/issues/539
if os.getenv('NUMBA_DISABLE_JIT'):
    def to_fixed_tuple(arr, l):
        return tuple(arr)
else:
    from numba.unsafe.ndarray import to_fixed_tuple


def get_correlation(sum_result, match_pattern: MatchPattern):
    '''
    Calculate the correlation between :code:`sum_result` and :code:`match_pattern`.

    .. versionadded:: 0.4.0.dev0

    Parameters
    ----------

    sum_result: numpy.ndarray
        2D result frame as correlation input
    match_pattern : MatchPattern
        Instance of :class:`~libertem.udf.blobfinder.MatchPattern` to correlate
        :code:`sum_result` with
    '''
    spec_mask = match_pattern.get_template(sig_shape=sum_result.shape)
    spec_sum = fft.rfft2(sum_result)
    corrspec = spec_mask * spec_sum
    return fft.fftshift(fft.irfft2(corrspec))


def get_peaks(sum_result, match_pattern: MatchPattern, num_peaks):
    '''
    Find peaks of the correlation between :code:`sum_result` and :code:`match_pattern`.

    The result  can then be used as input to
    :meth:`~libertem.analysis.fullmatch.FullMatcher.full_match`
    to extract grid parameters, :meth:`~libertem.udf.blobfinder.correlation.run_fastcorrelation`
    to find the position in each frame or to construct a mask to extract feature vectors with
    :meth:`~libertem.udf.blobfinder.utils.feature_vector`.

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
    >>> pattern = libertem.udf.blobfinder.RadialGradient(radius=4)
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
    corr = get_correlation(sum_result, match_pattern)
    peaks = peak_local_max(corr, num_peaks=num_peaks)
    return peaks


@numba.njit
def center_of_mass(arr):
    r_y = r_x = np.float32(0)
    for y in range(arr.shape[0]):
        for x in range(arr.shape[1]):
            r_y += np.float32(arr[y, x]*y)
            r_x += np.float32(arr[y, x]*x)
    s = arr.sum()
    return (np.float32(r_y/s), np.float32(r_x/s))


@numba.njit
def refine_center(center, r, corrmap):
    (y, x) = center
    s = corrmap.shape
    r = min(r, y, x, s[0] - y - 1, s[1] - x - 1)
    if r <= 0:
        return (np.float32(y), np.float32(x))
    else:
        # FIXME See and compare with Extension of Phase Correlation to Subpixel Registration
        # Hassan Foroosh
        # That one or a close/similar/cited one
        cutout = corrmap[y-r:y+r+1, x-r:x+r+1]
        m = np.min(cutout)
        ry, rx = center_of_mass(cutout - m)
        refined_y = y + ry - r
        refined_x = x + rx - r
        # print(y, x, refined_y, refined_x, "\n", cutout)
        return (np.float32(refined_y), np.float32(refined_x))


@numba.njit
def peak_elevation(center, corrmap, height, r_min=1.5, r_max=np.float('inf')):
    '''
    Return the slope of the tightest cone around :code:`center` with height :code:`height`
    that touches :code:`corrmap` between :code:`r_min` and :code:`r_max`.

    The correlation of two disks -- mask and perfect diffraction spot -- has the shape of a cone.
    The function's return value correlates with the quality of a correlation. Higher slope
    means a strong peak and
    no side maxima, while weak signal or side maxima lead to a flatter slope.

    Parameters
    ----------
    center : numpy.ndarray
        (y, x) coordinates of the center within the :code:`corrmap`
    corrmap : numpy.ndarray
        Correlation map
    height : float
        The height is provided as a parameter since center can be float values from refinement
        and the height value is conveniently available from the calling function.
    r_min : float, optional
        Masks out a small local plateau around the peak that would distort and dominate
        the calculation.
    r_max : float, optional
        Mask out neighboring peaks if a large area with several legitimate peaks is
        correlated.

    Returns
    -------
    elevation : float
        Elevation of the tightest cone that fits the correlation map within the given
        parameter range.
    '''
    peak_y, peak_x = center
    (size_y, size_x) = corrmap.shape
    result = np.float32(np.inf)

    for y in range(size_y):
        for x in range(size_x):
            dist = np.sqrt((y - peak_y)**2 + (x - peak_x)**2)
            if (dist >= r_min) and (dist < r_max):
                result = min((result, np.float32((height - corrmap[y, x]) / dist)))
    return max(0, result)


def do_correlations(template, crop_parts):
    spec_parts = fft.rfft2(crop_parts)
    corrspecs = template * spec_parts
    corrs = fft.fftshift(fft.irfft2(corrspecs), axes=(-1, -2))
    return corrs


@numba.njit
def unravel_index(index, shape):
    sizes = np.zeros(len(shape), dtype=np.int64)
    result = np.zeros(len(shape), dtype=np.int64)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return to_fixed_tuple(result, len(shape))


@numba.njit
def evaluate_correlations(corrs, peaks, crop_size,
        out_centers, out_refineds, out_heights, out_elevations):
    for i in range(len(corrs)):
        corr = corrs[i]
        center = unravel_index(np.argmax(corr), corr.shape)
        refined = np.array(refine_center(center, 2, corr), dtype=np.float32)
        height = np.float32(corr[center])
        out_centers[i] = _shift(np.array(center), peaks[i], crop_size)
        out_refineds[i] = _shift(refined, peaks[i], crop_size)
        out_heights[i] = height
        out_elevations[i] = np.float32(peak_elevation(refined, corr, height))


def log_scale(data, out):
    return np.log(data - np.min(data) + 1, out=out)


def log_scale_cropbufs_inplace(crop_bufs):
    m = np.min(crop_bufs, axis=(-1, -2)) - 1
    np.log(crop_bufs - m[:, np.newaxis, np.newaxis], out=crop_bufs)


@numba.njit
def crop_disks_from_frame(peaks, frame, crop_size, out_crop_bufs):

    def frame_coord_y(peak, y):
        return y + peak[0] - crop_size

    def frame_coord_x(peak, x):
        return x + peak[1] - crop_size

    fy, fx = frame.shape
    for i in range(len(peaks)):
        peak = peaks[i]
        for y in range(out_crop_bufs.shape[1]):
            yy = frame_coord_y(peak, y)
            y_outside = yy < 0 or yy >= fy
            for x in range(out_crop_bufs.shape[2]):
                xx = frame_coord_x(peak, x)
                x_outside = xx < 0 or xx >= fx
                if y_outside or x_outside:
                    out_crop_bufs[i, y, x] = 0
                else:
                    out_crop_bufs[i, y, x] = frame[yy, xx]


@numba.njit
def _shift(relative_center, anchor, crop_size):
    return relative_center + anchor - np.array((crop_size, crop_size))


class CorrelationUDF(UDF):
    '''
    Abstract base class for peak correlation implementations
    '''
    def __init__(self, peaks, *args, **kwargs):
        '''
        Parameters
        ----------

        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions in px to correlate
        '''
        super().__init__(peaks=np.round(peaks).astype(int), *args, **kwargs)

    def get_result_buffers(self):
        '''
        The common buffers for all correlation methods.

        :code:`centers`:
            (y, x) integer positions.
        :code:`refineds`:
            (y, x) positions with subpixel refinement.
        :code:`peak_values`:
            Peak height in the log scaled frame.
        :code:`peak_elevations`:
            Peak quality (result of :meth:`peak_elevation`).

        See source code for details of the buffer declaration.
        '''
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

    def output_buffers(self):
        '''
        This function allows abstraction of the result buffers from
        the default implementation in :meth:`get_result_buffers`.

        Override this function if you wish to redirect the results to different
        buffers, for example ragged arrays or binned processing.
        '''
        r = self.results
        return (r.centers, r.refineds, r.peak_values, r.peak_elevations)

    def postprocess(self):
        pass


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
        # For testing purposes, allow to inject a different limit via
        # an internal kwarg
        # It has to come through kwarg because of how UDFs are run
        self.limit = kwargs.get('__limit', 2**19)  # 1/2 MB
        super().__init__(*args, **kwargs)

    def get_task_data(self):
        ""
        n_peaks = len(self.params.peaks)
        mask = self.get_pattern()
        crop_size = mask.get_crop_size()
        template = mask.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
        dtype = np.float32
        full_size = (2 * crop_size)**2 * dtype(1).nbytes
        buf_count = min(max(1, self.limit // full_size), n_peaks)
        crop_bufs = zeros((buf_count, 2 * crop_size, 2 * crop_size), dtype=dtype)
        kwargs = {
            'crop_bufs': crop_bufs,
            'template': template,
        }
        return kwargs

    def get_peaks(self):
        return self.params.peaks

    def get_pattern(self):
        return self.params.match_pattern

    def get_template(self):
        return self.task_data.template

    def process_frame(self, frame):
        match_pattern = self.get_pattern()
        peaks = self.get_peaks()
        crop_bufs = self.task_data.crop_bufs
        crop_size = match_pattern.get_crop_size()
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        template = self.get_template()
        buf_count = len(crop_bufs)
        block_count = (len(peaks) - 1) // buf_count + 1
        for block in range(block_count):
            start = block * buf_count
            stop = min((block + 1) * buf_count, len(peaks))
            size = stop - start
            crop_disks_from_frame(
                peaks=peaks[start:stop], frame=frame, crop_size=crop_size,
                out_crop_bufs=crop_bufs[:size]
            )
            log_scale_cropbufs_inplace(crop_bufs[:size])
            corrs = do_correlations(template, crop_bufs[:size])
            evaluate_correlations(
                corrs=corrs, peaks=peaks[start:stop], crop_size=crop_size,
                out_centers=centers[start:stop], out_refineds=refineds[start:stop],
                out_heights=peak_values[start:stop], out_elevations=peak_elevations[start:stop]
            )


class FullFrameCorrelationUDF(CorrelationUDF):
    '''
    Fourier-based correlation-based refinement of peak positions within a search
    frame for each peak using a single correlation step. This can be faster for
    correlating a large number of peaks in small frames in comparison to
    :class:`FastCorrelationUDF`. However, it is more sensitive to interference
    from strong peaks next to the peak of interest.

    .. versionadded:: 0.3.0
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
        # For testing purposes, allow to inject a different limit via
        # an internal kwarg
        # It has to come through kwarg because of how UDFs are run
        self.limit = kwargs.get('__limit', 2**19)  # 1/2 MB

        super().__init__(*args, **kwargs)

    def get_task_data(self):
        ""
        mask = self.get_pattern()
        n_peaks = len(self.params.peaks)
        template = mask.get_template(sig_shape=self.meta.dataset_shape.sig)
        dtype = np.float32
        frame_buf = zeros(shape=self.meta.dataset_shape.sig, dtype=dtype)
        crop_size = mask.get_crop_size()
        full_size = (2 * crop_size)**2 * dtype(1).nbytes
        buf_count = min(max(1, self.limit // full_size), n_peaks)
        kwargs = {
            'template': template,
            'frame_buf': frame_buf,
            'buf_count': buf_count,
        }
        return kwargs

    def get_peaks(self):
        return self.params.peaks

    def get_pattern(self):
        return self.params.match_pattern

    def get_template(self):
        return self.task_data.template

    def process_frame(self, frame):
        match_pattern = self.get_pattern()
        peaks = self.get_peaks()
        crop_size = match_pattern.get_crop_size()
        template = self.get_template()
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        frame_buf = self.task_data.frame_buf
        log_scale(frame, out=frame_buf)
        spec_part = fft.rfft2(frame_buf)
        corrspec = template * spec_part
        corr = fft.fftshift(fft.irfft2(corrspec))
        buf_count = self.task_data.buf_count
        crop_bufs = np.zeros((buf_count, 2 * crop_size, 2 * crop_size), dtype=np.float32)
        block_count = (len(peaks) - 1) // buf_count + 1
        for block in range(block_count):
            start = block * buf_count
            stop = min(len(peaks), (block + 1) * buf_count)
            size = stop - start
            crop_disks_from_frame(
                peaks=peaks[start:stop], frame=corr, crop_size=crop_size,
                out_crop_bufs=crop_bufs[:size]
            )
            evaluate_correlations(
                corrs=crop_bufs[:size], peaks=peaks[start:stop], crop_size=crop_size,
                out_centers=centers[start:stop], out_refineds=refineds[start:stop],
                out_heights=peak_values[start:stop], out_elevations=peak_elevations[start:stop]
            )


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
        """
        This method adds the :code:`corr` buffer to the result of
        :meth:`CorrelationUDF.get_result_buffers`. See source code for the
        exact buffer declaration.
        """
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
        ""
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
        """
        The correlation results are evaluated during postprocessing since this
        implementation uses tiled processing where the correlations are
        incomplete in :meth:`process_tile`.
        """
        steps = 2 * self.params.steps + 1
        corrmaps = self.results.corr.reshape((
            -1,  # frames
            len(self.params.peaks),  # peaks
            steps,  # Y steps
            steps,  # X steps
        ))
        peaks = self.params.peaks
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        for f in range(corrmaps.shape[0]):
            evaluate_correlations(
                corrs=corrmaps[f], peaks=peaks, crop_size=self.params.steps,
                out_centers=centers[f], out_refineds=refineds[f],
                out_heights=peak_values[f], out_elevations=peak_elevations[f]
            )


def run_fastcorrelation(ctx, dataset, peaks, match_pattern: MatchPattern, roi=None):
    """
    Wrapper function to construct and run a :class:`FastCorrelationUDF`

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
    peaks : numpy.ndarray
        List of peaks with (y, x) coordinates
    match_pattern : libertem.udf.blobfinder.patterns.MatchPattern
    roi : numpy.ndarray, optional
        Boolean mask of the navigation dimension to select region of interest (ROI)

    Returns
    -------
    buffers : Dict[libertem.common.buffers.BufferWrapper]
        See :meth:`CorrelationUDF.get_result_buffers` for details.
    """
    peaks = peaks.astype(np.int)
    udf = FastCorrelationUDF(peaks=peaks, match_pattern=match_pattern)
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)


def run_blobfinder(ctx, dataset, match_pattern: MatchPattern, num_peaks, roi=None):
    """
    Wrapper function to find peaks in a dataset and refine their position using
    :class:`FastCorrelationUDF`

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.base.DataSet
    match_pattern : libertem.udf.blobfinder.patterns.MatchPattern
    num_peaks : int
        Number of peaks to look for
    roi : numpy.ndarray, optional
        Boolean mask of the navigation dimension to select region of interest (ROI)

    Returns
    -------
    sum_result : numpy.ndarray
        Log-scaled sum frame of the dataset/ROI
    centers, refineds, peak_values, peak_elevations : libertem.common.buffers.BufferWrapper
        See :meth:`CorrelationUDF.get_result_buffers` for details.
    peaks : numpy.ndarray
        List of found peaks with (y, x) coordinates
    """
    sum_analysis = ctx.create_sum_analysis(dataset=dataset)
    sum_result = ctx.run(sum_analysis, roi=roi)

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
