import functools

import numpy as np
from skimage.feature import peak_local_max
import scipy.ndimage as nd

from libertem.udf import ResultBuffer, map_frames
from libertem.masks import radial_gradient, background_substraction
from libertem.job.sum import SumFramesJob

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


def get_crop_size(radius, padding):
    return int(radius + radius * padding)


def crop_disks_from_frame(peaks, frame, mask):
    crop_size = mask.get_crop_size()
    for peak in peaks:
        yield frame[
            peak[0] - crop_size:peak[0] + crop_size,
            peak[1] - crop_size:peak[1] + crop_size,
        ]


def get_result_buffers_pass_2(num_disks):
    """
    we 'declare' what kind of result buffers we need, without concrete shapes

    concrete shapes come later, either for partition or whole dataset
    """
    return {
        'centers': ResultBuffer(
            kind="nav", extra_shape=(num_disks, 2), dtype="u2"
        ),
        'refineds': ResultBuffer(
            kind="nav", extra_shape=(num_disks, 2), dtype="float32"
        ),
        'peak_values': ResultBuffer(
            kind="nav", extra_shape=(num_disks,), dtype="float32",
        ),
        'peak_elevations': ResultBuffer(
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


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


def pass_2(frame, template, crop_buf, peaks, mask,
           centers, refineds, peak_values, peak_elevations):
    crop_size = mask.get_crop_size()
    for disk_idx, crop_part in enumerate(
            crop_disks_from_frame(peaks=peaks, frame=frame, mask=mask)):
        scaled = log_scale(crop_part, out=crop_buf)
        center, refined, peak_value, peak_elevation = do_correlation(template, scaled)
        crop_origin = np.array(peaks[disk_idx] - [crop_size, crop_size], dtype='u2')
        abs_center = np.array(center + crop_origin, dtype='u2')
        abs_refined = np.array(refined + crop_origin, dtype='float32')
        check_cast(abs_center, centers)
        check_cast(abs_refined, refineds)
        check_cast(peak_value, peak_values)
        check_cast(peak_elevation, peak_elevations)
        centers[disk_idx] = abs_center
        refineds[disk_idx] = abs_refined
        peak_values[disk_idx] = peak_value
        peak_elevations[disk_idx] = peak_elevation


def pass_2_merge(partition_result_buffers, centers, refineds, peak_values, peak_elevations):
    c = partition_result_buffers['centers'].data
    r = partition_result_buffers['refineds'].data
    p = partition_result_buffers['peak_values'].data
    e = partition_result_buffers['peak_elevations'].data
    check_cast(c, centers)
    check_cast(r, refineds)
    check_cast(p, peak_values)
    check_cast(e, peak_elevations)
    centers[:] = c
    refineds[:] = r
    peak_values[:] = p
    peak_elevations[:] = e


def run_analysis(ctx, dataset, parameters):
    sum_job = SumFramesJob(dataset=dataset)
    sum_result = ctx.run(sum_job)
    sum_result = np.log(sum_result - np.min(sum_result) + 1)

    peaks = get_peaks(
        parameters=parameters,
        framesize=tuple(dataset.shape.sig),
        sum_result=sum_result,
    )

    pass_2_results = map_frames(
        ctx=ctx,
        dataset=dataset,
        make_result_buffers=functools.partial(
            get_result_buffers_pass_2,
            num_disks=parameters['num_disks'],
        ),
        merge=pass_2_merge,
        init_fn=functools.partial(init_pass_2, peaks=peaks, parameters=parameters),
        frame_fn=pass_2,
    )

    return (sum_result, pass_2_results['centers'],
        pass_2_results['refineds'], pass_2_results['peak_values'],
        pass_2_results['peak_elevations'], peaks)
