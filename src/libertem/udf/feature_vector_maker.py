from libertem.udf import UDF
from libertem.masks import _make_circular_mask
from libertem.udf.stddev import run_stddev

from skimage.feature import peak_local_max


class FeatureVecMakerUDF(UDF):
    def get_result_buffers(self):
        coordinates = self.params.coordinates
        return {
            'feature_vec': self.buffer(
                kind="nav", extra_shape=(coordinates.shape[0],), dtype="bool"
            ),
        }

    def process_frame(self, frame):
        delta = self.params.delta
        savg = self.params.savg
        coordinates = self.params.coordinates
        ref = savg[coordinates[:, 0], coordinates[:, 1]]
        for j in range(0, coordinates.shape[0]):
            if (frame[coordinates[j, 0], coordinates[j, 1]]-ref[j])/ref[j] > delta:
                self.results.feature_vec[j] = 1
        return


def make_feature_vec(ctx, dataset, num, delta, center=None, rad_in=None, rad_out=None, roi=None):
    """
    Return a value after integration of Fourier spectrum for each frame over ring.
    Parameters
    ----------
    ctx: Context
        Context class that contains methods for loading datasets,
        creating jobs on them and running them

    dataset: DataSet
        A dataset with 1- or 2-D scan dimensions and 2-D frame dimensions

    num: int
        Number of possible peak positions to detect (better put higher value,
        the output is limited to the number of peaks the algorithm could find)

    delta: float
        Relative intensity difference between current frame and reference image for decision making
        for feature vector value (delta = (x-ref)/ref, so, normally, value should be in range [0,1])

    rad_in: int, optional
        Inner radius in pixels of a ring to mask region of interest of SD image to delete outliers
        for peak finding

    rad_out: int, optional
        Outer radius in pixels of a ring to mask region of interest of SD image to delete outliers
        for peak finding

    center: tuple, optional
        (y,x) - pixels, coordinates of a ring to mask region of interest of SD image
        to delete outliers for peak finding

    roi: numpy.ndarray, optional
        boolean array which limits the elements the UDF is working on.
        Has a shape of dataset_shape.nav

    Returns
    -------
    pass_results: dict
        Returns a feature vector for each frame.
        "1" - denotes presence of peak for current frame for given possible peak position,
        "0" - absence of peak for current frame for given possible peak position,
        To return 2-D array use pass_results['feature_vec'].data

    coordinates: numpy array of int
        Returns array of coordinates of possible peaks positions

    """
    res_stat = run_stddev(ctx, dataset)
    savg = res_stat['mean']
    sstd = res_stat['std']
    sshape = sstd.shape
    if not (center is None or rad_in is None or rad_out is None):
        mask_out = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_out)
        mask_in = 1*_make_circular_mask(center[1], center[0], sshape[1], sshape[0], rad_in)
        mask = mask_out - mask_in
        masked_sstd = sstd*mask
    else:
        masked_sstd = sstd
    coordinates = peak_local_max(masked_sstd, num_peaks=num, min_distance=0)
    udf = FeatureVecMakerUDF(
        num=num, delta=delta, center=center, rad_in=rad_in, rad_out=rad_out,
        savg=savg, coordinates=coordinates
        )
    pass_results = ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    return (pass_results, coordinates)
