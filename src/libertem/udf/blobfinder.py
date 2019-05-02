import functools

import numpy as np
from skimage.feature import peak_local_max
import scipy.ndimage as nd

from libertem.udf import check_cast
from libertem.common.buffers import BufferWrapper
from libertem.masks import radial_gradient, background_substraction
from libertem.job.sum import SumFramesJob

import libertem.analysis.gridmatching as grm

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


def get_peaks(parameters, framesize, sum_result):
    """
    executed on master node, calculate crop rects from average image

    padding : float
        to prevent very close disks from interfering with another,
        we add only a small fraction of radius to area that will be cropped
    """
    mask = mask_maker(parameters)
    num_disks = parameters['num_disks']
    spec_mask = mask.get_template(sig_shape=framesize)
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
    center = np.unravel_index(np.argmax(corr), corr.shape)
    refined = np.array(refine_center(center, 2, corr), dtype='float32')
    height = np.float32(corr[center])
    elevation = np.float32(peak_elevation(refined, corr, height))
    return np.array(center, dtype='u2'), refined, height, elevation


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


def get_result_buffers_pass_2(num_disks):
    """
    we 'declare' what kind of result buffers we need, without concrete shapes

    concrete shapes come later, either for partition or whole dataset
    """
    return {
        'centers': BufferWrapper(
            kind="nav", extra_shape=(num_disks, 2), dtype="u2"
        ),
        'refineds': BufferWrapper(
            kind="nav", extra_shape=(num_disks, 2), dtype="float32"
        ),
        'peak_values': BufferWrapper(
            kind="nav", extra_shape=(num_disks,), dtype="float32",
        ),
        'peak_elevations': BufferWrapper(
            kind="nav", extra_shape=(num_disks,), dtype="float32",
        ),
    }


def init_pass_2(partition, peaks, parameters):
    mask = mask_maker(parameters)
    crop_size = mask.get_crop_size()
    template = mask.get_template(sig_shape=(2 * crop_size, 2 * crop_size))
    crop_buf = zeros((2 * crop_size, 2 * crop_size), dtype="float32")
    kwargs = {
        'peaks': peaks,
        'mask': mask,
        'crop_buf': crop_buf,
        'template': template,
    }
    return kwargs


def _shift(relative_center, anchor, crop_size):
    return relative_center + anchor - [crop_size, crop_size]


def pass_2(frame, template, crop_buf, peaks, mask,
           centers, refineds, peak_values, peak_elevations):
    crop_size = mask.get_crop_size()
    for disk_idx, (crop_part, crop_buf_slice) in enumerate(
            crop_disks_from_frame(peaks=peaks, frame=frame, mask=mask)):

        crop_buf[:] = 0  # FIXME: we need to do this only for edge cases
        log_scale(crop_part, out=crop_buf[crop_buf_slice])
        center, refined, peak_value, peak_elevation = do_correlation(template, crop_buf)
        abs_center = _shift(center, peaks[disk_idx], crop_size).astype('u2')
        abs_refined = _shift(refined, peaks[disk_idx], crop_size).astype('float32')
        check_cast(abs_center, centers)
        check_cast(abs_refined, refineds)
        check_cast(peak_value, peak_values)
        check_cast(peak_elevation, peak_elevations)
        centers[disk_idx] = abs_center
        refineds[disk_idx] = abs_refined
        peak_values[disk_idx] = peak_value
        peak_elevations[disk_idx] = peak_elevation


def run_blobcorrelation(ctx, dataset, peaks, parameters):
    peaks = peaks.astype(np.int)
    return ctx.run_udf(
        dataset=dataset,
        fn=pass_2,
        init=functools.partial(init_pass_2, peaks=peaks, parameters=parameters),
        make_buffers=functools.partial(
            get_result_buffers_pass_2,
            num_disks=len(peaks),
        ),
    )


def run_blobfinder(ctx, dataset, parameters):
    sum_job = SumFramesJob(dataset=dataset)
    sum_result = ctx.run(sum_job)
    sum_result = np.log(sum_result - np.min(sum_result) + 1)

    peaks = get_peaks(
        parameters=parameters,
        framesize=tuple(dataset.shape.sig),
        sum_result=sum_result,
    )

    pass_2_results = run_blobcorrelation(ctx, dataset, peaks, parameters)

    return (sum_result, pass_2_results['centers'],
        pass_2_results['refineds'], pass_2_results['peak_values'],
        pass_2_results['peak_elevations'], peaks)


def get_result_buffers_refine(num_disks):
    return {
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


def refine(frame, template, start_zero, start_a, start_b, crop_buf, peaks, mask,
           centers, refineds, peak_values, peak_elevations, zero, a, b, selector, match_params):
    pass_2(
        frame=frame,
        template=template,
        crop_buf=crop_buf,
        peaks=peaks,
        mask=mask,
        centers=centers,
        refineds=refineds,
        peak_values=peak_values,
        peak_elevations=peak_elevations
    )
    match = grm.fastmatch(
        centers=centers,
        refineds=refineds,
        peak_values=peak_values,
        peak_elevations=peak_elevations,
        zero=start_zero,
        a=start_a,
        b=start_b,
        parameters=match_params
    )
    # We don't check the cast since we cast from float64 to float32 here
    # and avoid a lot of boilerplate
    zero[:] = match.zero
    a[:] = match.a
    b[:] = match.b
    selector[:] = match.selector


def run_refine(ctx, dataset, zero, a, b, indices, corr_params, match_params):
    peaks = grm.calc_coords(zero, a, b, indices).astype('int')

    return ctx.run_udf(
        dataset=dataset,
        fn=functools.partial(
            refine,
            start_zero=zero,
            start_a=a,
            start_b=b,
            match_params=match_params
        ),
        init=functools.partial(init_pass_2, peaks=peaks, parameters=corr_params),
        make_buffers=functools.partial(
            get_result_buffers_refine,
            num_disks=len(peaks),
        ),
    )
