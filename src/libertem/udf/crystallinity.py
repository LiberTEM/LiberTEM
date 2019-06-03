import functools

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.masks import _make_circular_mask


def make_result_buffers():
    return {
        'intensity': BufferWrapper(
            kind="nav", dtype="float32"
        ),
    }


def init_fft(partition, rad_in, rad_out, real_center, real_rad):
    sigshape = partition.shape.sig
    if not (real_center is None or real_rad is None):
        real_mask = 1-1*_make_circular_mask(
            real_center[1], real_center[0], sigshape[1], sigshape[0], real_rad
        )
    else:
        real_mask = None
    fourier_mask_out = 1*_make_circular_mask(
        sigshape[1]*0.5, sigshape[0]*0.5, sigshape[1], sigshape[0], rad_out
    )
    fourier_mask_in = 1*_make_circular_mask(
        sigshape[1]*0.5, sigshape[0]*0.5, sigshape[1], sigshape[0], rad_in
    )
    fourier_mask = fourier_mask_out - fourier_mask_in

    kwargs = {
        'real_mask': real_mask,
        'fourier_mask': fourier_mask,
    }
    return kwargs


def fft(frame, real_mask, fourier_mask, intensity):
    if not (real_mask is None):
        intensity[:] = np.sum(np.fft.fftshift(abs(np.fft.fft2(frame*real_mask)))*fourier_mask)
    else:
        intensity[:] = np.sum(np.fft.fftshift(abs(np.fft.fft2(frame)))*fourier_mask)
    return


def run_analysis_crystall(ctx, dataset, rad_in, rad_out, real_center=None, real_rad=None):
    """
    Return a value after integration of Fourier spectrum for each frame over ring.
    Parameters
    ----------
    ctx: Context
        Context class that contains methods for loading datasets,
        creating jobs on them and running them

    dataset: DataSet
        A dataset with 1- or 2-D scan dimensions and 2-D frame dimensions
    rad_in: int
        Inner radius of a ring mask for the integration in Fourier space

    rad_out: int
        Outer radius of a ring mask for the integration in Fourier space

    real_center: tuple
        (x,y) - coordinates of a center of a circle for a masking out zero-order peak in real space

    real_rad: int
        Radius of circle for a masking out zero-order peak in real space

    Returns
    -------
    pass_results: dict
        Returns a "crystallinity" value for each frame.
        To return 2-D array use pass_results['intensity'].data

    """

    results = ctx.run_udf(
        dataset=dataset,
        make_buffers=make_result_buffers,
        init=functools.partial(
            init_fft, rad_in=rad_in, rad_out=rad_out, real_center=real_center, real_rad=real_rad),
        fn=fft,
    )

    return results
