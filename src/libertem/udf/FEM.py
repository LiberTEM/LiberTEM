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


def init(partition, center, rad_in, rad_out):
    mask_out = 1*_make_circular_mask(
        center[1], center[0],
        partition.shape.sig[1], partition.shape.sig[0],
        rad_out
    )
    mask_in = 1*_make_circular_mask(
        center[1], center[0],
        partition.shape.sig[1], partition.shape.sig[0],
        rad_in
    )
    mask = mask_out - mask_in

    kwargs = {
        'mask': mask,
    }
    return kwargs


def masked_std(frame, mask, intensity):
    intensity[:] = np.std(frame[mask == 1])
    return


def run_fem(ctx, dataset, center, rad_in, rad_out):
    """
    Return a standard deviation(SD) value for each frame of pixels which belong to ring mask.
    Parameters
    ----------
    ctx: Context
        Context class that contains methods for loading datasets,
        creating jobs on them and running them

    dataset: DataSet
        A dataset with 1- or 2-D scan dimensions and 2-D frame dimensions

    center: tuple
        (x,y) - coordinates of a center of a ring for a masking region of interest to calculate SD

    rad_in: int
        Inner radius of a ring mask

    rad_out: int
        Outer radius of a ring mask

    Returns
    -------
    pass_results: dict
        Returns a standard deviation(SD) value for each frame of pixels which belong to ring mask.
        To return 2-D array use pass_results['intensity'].data

    """

    pass_results = ctx.run_udf(
        dataset=dataset,
        make_buffers=make_result_buffers,
        init=functools.partial(init, center=center, rad_in=rad_in, rad_out=rad_out),
        fn=masked_std,
    )

    return (pass_results)
