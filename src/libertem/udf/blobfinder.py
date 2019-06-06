import functools

import numpy as np
from skimage.feature import peak_local_max
import scipy.ndimage as nd
import matplotlib.pyplot as plt

from libertem.udf import UDF
from libertem.masks import radial_gradient, background_substraction, sparse_template_multi_stack
from libertem.job.sum import SumFramesJob
from libertem.job.masks import MaskContainer

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
    def __init__(self, parameters):
        pass

    def get_crop_size(self):
        raise NotImplementedError

    def get_mask(self, sig_shape):
        raise NotImplementedError

    def get_template(self, sig_shape):
        return fft.rfft2(self.get_mask(sig_shape))


class RadialGradient(MatchPattern):
    def __init__(self, parameters):
        self.radius = parameters['radius']
        self.padding = parameters['padding']

    def get_crop_size(self):
        return int(np.ceil(self.radius * (1 + self.padding)))

    def get_mask(self, sig_shape):
        return radial_gradient(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius,
        )


class BackgroundSubstraction(MatchPattern):
    def __init__(self, parameters):
        self.radius = parameters['radius']
        self.padding = parameters['padding']
        self.radius_outer = parameters.get('radius_outer', self.radius*1.5)

    def get_crop_size(self):
        return int(np.ceil(max(self.radius, self.radius_outer) * (1 + self.padding)))

    def get_mask(self, sig_shape):
        return background_substraction(
            centerY=sig_shape[0] // 2,
            centerX=sig_shape[1] // 2,
            imageSizeY=sig_shape[0],
            imageSizeX=sig_shape[1],
            radius=self.radius_outer,
            radius_inner=self.radius
        )


def mask_maker(parameters):
    if parameters['mask_type'] == 'radial_gradient':
        return RadialGradient(parameters)
    elif parameters['mask_type'] == 'background_substraction':
        return BackgroundSubstraction(parameters)
    else:
        raise ValueError("unknown mask type: %s" % parameters['mask_type'])


def get_peaks(parameters, sum_result):
    """
    executed on master node, calculate crop rects from average image

    padding : float
        to prevent very close disks from interfering with another,
        we add only a small fraction of radius to area that will be cropped
    """
    mask = mask_maker(parameters)
    num_disks = parameters['num_disks']
    spec_mask = mask.get_template(sig_shape=sum_result.shape)
    spec_sum = fft.rfft2(sum_result)
    corrspec = spec_mask * spec_sum
    corr = fft.fftshift(fft.irfft2(corrspec))
    peaks = peak_local_max(corr, num_peaks=num_disks)
    return peaks


def refine_center(center, r, corrmap):
    (y, x) = center
    s = corrmap.shape
    r = min(r, y, x, s[0] - y - 1, s[1] - x - 1)
    if r <= 0:
        return (y, x)
    else:
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

    return np.min(diff / dist[select])


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


def crop_disks_from_frame(peaks, frame, mask):
    crop_size = mask.get_crop_size()
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
    def __init__(self, *args, **kwargs):
        '''
        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions to correlate
        mask_type : str
            Mask to use for correlation. Currently one of 'radial_gradient' and
            'background_substraction'
        radius:
            Radius of the CBED disks in px
        padding:
            Extra space around radius as a fraction of radius, which defines the search
            area around a peak
        radius_outer:
            Only with 'background_substraction': Radius of outer region with negative values.
            For calculating the padding, the maximum of radius and radius_outer is used.
        '''
        super().__init__(*args, **kwargs)

    def get_task_data(self, meta):
        mask = mask_maker(self.params)
        crop_size = mask.get_crop_size()
        template = mask.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
        crop_buf = zeros((2 * crop_size, 2 * crop_size), dtype="float32")
        kwargs = {
            'mask': mask,
            'crop_buf': crop_buf,
            'template': template,
        }
        return kwargs

    def process_frame(self, frame):
        mask = self.task_data.mask
        peaks = self.params.peaks
        crop_buf = self.task_data.crop_buf
        crop_size = mask.get_crop_size()
        r = self.results
        for disk_idx, (crop_part, crop_buf_slice) in enumerate(
                crop_disks_from_frame(peaks=peaks, frame=frame, mask=mask)):

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
        peaks : numpy.ndarray
            Numpy array of (y, x) coordinates with peak positions to correlate
        mask_type : str
            Mask to use for correlation. Currently one of 'radial_gradient' and
            'background_substraction'
        radius:
            Radius of the CBED disks in px
        padding:
            Extra space around radius as a fraction of radius. Can be zero for this method
            since the shifting is performed independently of the template size.
        radius_outer:
            Only with 'background_substraction': Radius of outer region with negative values.
            For calculating the padding, the maximum of radius and radius_outer is used.
        steps:
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

    def get_task_data(self, meta):
        mask = mask_maker(self.params)
        crop_size = mask.get_crop_size()
        size = (2 * crop_size + 1, 2 * crop_size + 1)
        template = mask.get_mask(sig_shape=size)
        steps = self.params.steps
        peak_offsetY, peak_offsetX = np.mgrid[-steps:steps + 1, -steps:steps + 1]

        offsetY = self.params.peaks[:, 0, np.newaxis, np.newaxis] + peak_offsetY - crop_size
        offsetX = self.params.peaks[:, 1, np.newaxis, np.newaxis] + peak_offsetX - crop_size

        offsetY = offsetY.flatten()
        offsetX = offsetX.flatten()

        stack = functools.partial(
            sparse_template_multi_stack,
            mask_index=range(len(offsetY)),
            offsetX=offsetX,
            offsetY=offsetY,
            template=template,
            imageSizeX=meta.dataset_shape.sig[1],
            imageSizeY=meta.dataset_shape.sig[0]
        )
        # CSC matrices in combination with transposed data are fastest
        container = MaskContainer(mask_factories=stack, dtype=np.float32,
            use_sparse='scipy.sparse.csc')

        kwargs = {
            'mask_container': container,
            'crop_size': crop_size,
        }
        return kwargs

    def process_tile(self, tile, tile_slice):
        c = self.task_data['mask_container']
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


def run_fastcorrelation(ctx, dataset, peaks, parameters, roi=None):
    peaks = peaks.astype(np.int)
    udf = FastCorrelationUDF(peaks=peaks, **parameters)
    return ctx.run_udf(dataset=dataset, udf=udf, roi=roi)


def run_blobfinder(ctx, dataset, parameters, roi=None):
    # FIXME implement ROI for SumFramesJob
    sum_job = SumFramesJob(dataset=dataset)
    sum_result = ctx.run(sum_job)

    sum_result = log_scale(sum_result, out=None)

    peaks = get_peaks(
        parameters=parameters,
        sum_result=sum_result,
    )

    pass_2_results = run_fastcorrelation(
        ctx=ctx,
        dataset=dataset,
        peaks=peaks,
        parameters=parameters,
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


class FastmatchMixin(RefinementMixin):
    '''
    Refinement using gridmatching.fastmatch()
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters for gridmatching.fastmatch():

        start_a : (y, x)
            Approximate value for "a" vector.
        start_b : (y, x)
            Approximate value for "b" vector.
        start_zero : (y, x)
            Approximate value for "zero" point (origin, zero order peak)
        tolerance (optional):
            Relative position tolerance for peaks to be considered matches
        min_delta (optional):
            Minimum length of a potential grid vector
        max_delta:
            Maximum length of a potential grid vector
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
<<<<<<< HEAD
        super().postprocess()
=======
>>>>>>> Postprocessing interface for UDFs
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = grm.fastmatch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                zero=p.start_zero,
                a=p.start_a,
                b=p.start_b,
                parameters=p,
            )
<<<<<<< HEAD
            self.apply_match(index, match)
=======
            self.apply_match(index, match)        
>>>>>>> Postprocessing interface for UDFs


class AffineMixin(RefinementMixin):
    '''
    Refinement using gridmatching.affinematch()
    '''
    def __init__(self, *args, **kwargs):
        '''
        Parameters for gridmatching.affinematch():

        indices : numpy.ndarray
            List of indices [(h1, k1), (h2, k2), ...] of all peaks. The indices can be
            non-integer and relative to any base vectors, including virtual ones like
            (1, 0); (0, 1). See documentation of gridmatching.affinematch() for details.
        '''
        super().__init__(*args, **kwargs)

    def postprocess(self):
<<<<<<< HEAD
        super().postprocess()
=======
>>>>>>> Postprocessing interface for UDFs
        p = self.params
        r = self.results
        for index in range(len(self.results.centers)):
            match = grm.affinematch(
                centers=r.centers[index],
                refineds=r.refineds[index],
                peak_values=r.peak_values[index],
                peak_elevations=r.peak_elevations[index],
                indices=p.indices,
            )
            self.apply_match(index, match)


def run_refine(ctx, dataset, zero, a, b, params, indices=None, roi=None):
    '''
    Refine the given lattice for each frame by calculating approximate peak positions and refining
    them for each frame by using the blobcorrelation and gridmatching.fastmatch().

    indices:
        Indices to refine. This is trimmed down to positions within the frame.
        As a convenience, for the indices parameter this function accepts both shape
        (n, 2) and (2, n, m) so that numpy.mgrid[h:k, i:j] works directly to specify indices.
        This saves boilerplate code when using this function. Default: numpy.mgrid[-10:10, -10:10].
    params['affine']:
        If True, use affine transformation matching. This is robust against a
        distorted field of view, but doesn't exclude outliers and requires the indices of the
        peaks to be known. See documentation of gridmatching.affinematch() for details.

    returns:
        (result, used_indices) where result is
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
        }
        and used_indices are the indices that were within the frame.
    '''
    if indices is None:
        indices = np.mgrid[-10:10, -10:10]
    s = indices.shape
    # Output of mgrid
    if (len(s) == 3) and (s[0] == 2):
        indices = np.concatenate(indices.T)
    # List of (i, j) pairs
    elif (len(s) == 2) and (s[1] == 2):
        pass
    else:
        raise ValueError(
            "Shape of indices is %s, expected (n, 2) or (2, n, m)" % str(indices.shape))

    (fy, fx) = tuple(dataset.shape.sig)

    peaks = grm.calc_coords(zero, a, b, indices).astype('int')

    selector = grm.within_frame(peaks, params['radius'], fy, fx)

    peaks = peaks[selector]
    indices = indices[selector]

    if params.get('method', 'fastcorrelation') == 'fastcorrelation':
        method = FastCorrelationUDF
    elif params['method'] == 'sparse':
        method = SparseCorrelationUDF

    if params.get('affine', False):
        mixin = AffineMixin
    else:
        mixin = FastmatchMixin

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
        **params
    )

    result = ctx.run_udf(
        dataset=dataset,
        udf=udf,
        roi=roi,
    )
    return (result, indices)


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
