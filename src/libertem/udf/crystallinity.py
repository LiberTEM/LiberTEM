import numpy as np

from libertem.masks import _make_circular_mask
from libertem.udf import UDF


class CrystallinityUDF(UDF):
    def get_result_buffers(self):
        return {
            'intensity': self.buffer(
                kind="nav", dtype="float32"
            ),
        }

    def get_task_data(self):
        sigshape = tuple(self.meta.partition_shape.sig)
        rad_in = self.params.rad_in
        rad_out = self.params.rad_out
        real_center = self.params.real_center
        real_rad = self.params.real_rad
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
        fourier_mask = np.fft.fftshift(fourier_mask_out - fourier_mask_in)
        half_fourier_mask = fourier_mask[:, :int(fourier_mask.shape[1]*0.5)+1]
        kwargs = {
            'real_mask': real_mask,
            'half_fourier_mask': half_fourier_mask,
        }
        return kwargs

    def process_frame(self, frame):
        h_f_mask = self.task_data.half_fourier_mask
        if self.task_data.real_mask is not None:
            maskedframe = frame*self.task_data.real_mask
        else:
            maskedframe = frame
        self.results.intensity[:] = np.sum(abs(np.fft.rfft2(maskedframe))*h_f_mask)


def run_analysis_crystall(ctx, dataset, rad_in, rad_out, real_center=None, real_rad=None, roi=None):
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
        Inner radius in pixels of a ring mask for the integration in Fourier space

    rad_out: int
        Outer radius in pixels of a ring mask for the integration in Fourier space

    real_center: tuple, optional
        (y,x) - pixels, coordinates of a center of a circle for a masking out zero-order peak
        in real space.

    real_rad: int, optional
        Radius in pixels of circle for a masking out zero-order peak in real space.
        If one from real_center or real_rad is missing: the integration will be done without
        masking zero-order peak out.

    Returns
    -------
    pass_results: dict
        Returns a "crystallinity" value for each frame.
        To return 2-D array use pass_results['intensity'].data

    """
    udf = CrystallinityUDF(
        rad_in=rad_in, rad_out=rad_out, real_center=real_center, real_rad=real_rad
        )
    pass_results = ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    return pass_results
