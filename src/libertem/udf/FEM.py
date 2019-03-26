import functools

import numpy as np
from skimage.feature import peak_local_max
import scipy.ndimage as nd

from libertem.udf import check_cast
from libertem.common.buffers import BufferWrapper
from libertem.masks import radial_gradient, background_substraction
from libertem.job.sum import SumFramesJob

def make_result_buffers():
    return {
        'example_intensity': ResultBuffer(
            kind="nav", dtype="float32"
        ),
    }

def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


def pass_2_merge(partition_result_buffers, example_intensity):
    c = partition_result_buffers['example_intensity'].data
    check_cast(c, example_intensity)
    example_intensity[:] = c


def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):

    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = x*x + y*y <= radius*radius
    return(mask)



def init_pass_2(partition, center, rad_in, rad_out):
    mask_out=1*_make_circular_mask(center[1]*0.5, center[0]*0.5, partition.shape.sig[1], partition.shape.sig[0], rad_out)
    mask_in=1*_make_circular_mask(center[1]*0.5, center[0]*0.5,partition.shape.sig[1], partition.shape.sig[0],rad_in)
    mask=mask_out-mask_in

    kwargs = {
        'mask': mask,
    }
    return kwargs



def pass_2(frame, mask, example_intensity):
    example_intensity[:]=np.std(frame[mask==1])
    return
   
    



def run_analysis(ctx, dataset, center, rad_in, rad_out):
    
    pass_2_results = map_frames(
        ctx=ctx,
        dataset=dataset,
        make_result_buffers=make_result_buffers,
        merge=pass_2_merge,
        init_fn=functools.partial(init_pass_2, center=center, rad_in=rad_in, rad_out=rad_out),
        frame_fn=pass_2,
    )

    return (pass_2_results)