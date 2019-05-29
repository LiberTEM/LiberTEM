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



def _make_circular_mask(centerX, centerY, imageSizeX, imageSizeY, radius):

    x, y = np.ogrid[-centerY:imageSizeY-centerY, -centerX:imageSizeX-centerX]
    mask = x*x + y*y <= radius*radius
    return(mask)



def init_fft(partition, rad_in, rad_out, real_center, real_rad):
    real_mask = 1-1*_make_circular_mask(real_center[1], real_center[0], partition.shape.sig[1], partition.shape.sig[0],real_rad)
    fourier_mask_out=1*_make_circular_mask(partition.shape.sig[1]*0.5, partition.shape.sig[0]*0.5, partition.shape.sig[1], partition.shape.sig[0],rad_out)
    fourier_mask_in=1*_make_circular_mask(partition.shape.sig[1]*0.5, partition.shape.sig[0]*0.5, partition.shape.sig[1], partition.shape.sig[0],rad_in)
    fourier_mask=fourier_mask_out-fourier_mask_in
    kwargs = {
        'real_mask': real_mask,
        'fourier_mask': fourier_mask,
    }
    return kwargs



def fft(frame, real_mask, fourier_mask, intensity):
    intensity[:]=np.sum(np.fft.fftshift(abs(np.fft.fft2(frame*real_mask)))*fourier_mask)
    return
   
    



def run_analysis_crystall(ctx, dataset, rad_in, rad_out, real_center=None, real_rad=None):
    
    results = ctx.run_udf(
        dataset=dataset,
        make_buffers=make_result_buffers,
        init=functools.partial(init_fft, rad_in=rad_in, rad_out=rad_out, real_center=real_center, real_rad=real_rad),
        fn=fft,
    )

    return results