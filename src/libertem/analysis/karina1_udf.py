import functools

import numpy as np
from skimage.feature import peak_local_max

from libertem.udf import ResultBuffer, map_frames
from libertem.masks import radial_gradient
from libertem.job.sum import SumFramesJob

try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros


def get_peaks(parameters, framesize, sum_result):
    """
    executed on master node, calculate crop rects from average image

    padding : float
        to prevent very close disks from interfering with another,
        we add only a small fraction of radius to area that will be cropped
    """
    radius = parameters['radius']
    num_disks = parameters['num_disks']
    spec_mask = get_template(sig_shape=framesize, radius=radius, mask_type=parameters['mask_type'])
    spec_sum = fft.rfft2(sum_result)
    corrspec = spec_mask * spec_sum
    corr = fft.fftshift(fft.irfft2(corrspec))
    peaks = peak_local_max(corr, num_peaks=num_disks)
    return peaks


def do_correlation(template, crop_part):
    spec_part = fft.rfft2(crop_part)
    corrspec = template * spec_part
    corr = fft.fftshift(fft.irfft2(corrspec))
    center = np.unravel_index(np.argmax(corr), corr.shape)
    return center, corr[center]


def get_template(sig_shape, radius, mask_type):
    if mask_type != "radial_gradient":
        raise ValueError("unknown mask type: %s" % mask_type)
    mask = radial_gradient(
        centerY=sig_shape[0] // 2,
        centerX=sig_shape[1] // 2,
        imageSizeY=sig_shape[0],
        imageSizeX=sig_shape[1],
        radius=radius,
    )
    spec_mask = fft.rfft2(mask)
    return spec_mask


def log_scale(data, out):
    return np.log(data - np.min(data) + 1, out=out)


def get_crop_size(radius, padding):
    return int(radius + radius * padding)


def crop_disks_from_frame(peaks, frame, padding, radius):
    crop_size = get_crop_size(radius, padding)
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
        'peak_values': ResultBuffer(
            kind="nav", extra_shape=(num_disks,), dtype="float32",
        ),
    }


def init_pass_2(partition, peaks, parameters):
    radius, padding = parameters['radius'], parameters['padding']
    crop_size = get_crop_size(radius, padding)
    template = get_template(
        sig_shape=(2 * crop_size, 2 * crop_size),
        radius=radius,
        mask_type=parameters['mask_type'],
    )
    crop_buf = zeros((2 * crop_size, 2 * crop_size), dtype="float32")
    kwargs = {
        'peaks': peaks,
        'padding': parameters['padding'],
        'radius': parameters['radius'],
        'crop_buf': crop_buf,
        'template': template,
    }
    return kwargs


def pass_2(frame, template, crop_buf, peaks, padding, radius,
           centers, peak_values):
    for disk_idx, crop_part in enumerate(crop_disks_from_frame(peaks=peaks,
                                                               frame=frame,
                                                               padding=padding,
                                                               radius=radius)):
        scaled = log_scale(crop_part, out=crop_buf)
        center, peak_value = do_correlation(template, scaled)
        centers[disk_idx] = center
        peak_values[disk_idx] = peak_value


def pass_2_merge(partition_result_buffers, centers, peak_values):
    centers[:] = partition_result_buffers['centers'].data
    peak_values[:] = partition_result_buffers['peak_values'].data


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

    return sum_result, pass_2_results['centers'], pass_2_results['peak_values'], peaks
