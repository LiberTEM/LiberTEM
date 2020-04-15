import warnings

from skimage.feature import peak_local_max

from libertem.udf import UDF
from libertem.masks import _make_circular_mask
from libertem.udf.stddev import run_stddev


# FIXME remove this file in 0.6.0 after deprecation period
class FeatureVecMakerUDF(UDF):
    """
    Creates a feature vector for each frame in ROI based on non-zero order diffraction peaks
    positions
    """

    def __init__(self, coordinates, delta, savg):
        """
        .. deprecated:: 0.5.0
            :code:`FeatureVecMakerUDF` is deprecated and will be removed in 0.6.0.
            Use :class:`libertem.udf.masks.ApplyMasksUDF` with a stack of sparse one-pixel masks
            or with a mask stack generated using
            :meth:`libertem_blobfinder.common.patterns.feature_vector` instead.

        Parameters
        ----------

        coordinates: np.array
            Array of frame coordinates (i.e. shape (N, 2) for N coords) which should be
            checked for peak intensity

        savg: np.array
            Reference image, usually the mean of all frames (or mean of all frames in a ROI)

        delta: float
            Relative intensity difference between current frame and reference
            image for decision making for feature vector value
            (delta = (x-ref) / ref, so, normally, value should be in range [0,1])

        Examples
        --------
        >>> coords = np.array([[1, 2], [3, 4], [5, 6]])
        >>> savg = np.random.randn(16, 16)  # in real usage should be mean of all frames
        >>> udf = FeatureVecMakerUDF(delta=0.5, coordinates=coords, savg=savg)
        >>> result = ctx.run_udf(dataset=dataset, udf=udf)
        >>> np.array(result["feature_vec"]).shape
        (16, 16, 3)
        """
        warnings.warn(
            "FeatureVecMakerUDF is deprecated and will be removed in 0.6.0. Use "
            "libertem.udf.masks.ApplyMasksUDF with a sparse stack of one-pixel masks "
            "or with a mask stack generated using "
            "libertem_blobfinder.common.patterns.feature_vector instead.",
            DeprecationWarning
        )
        super().__init__(coordinates=coordinates, delta=delta, savg=savg)

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


def make_feature_vec(ctx, dataset, delta, n_peaks, min_dist=None,
                    center=None, rad_in=None, rad_out=None, roi=None):
    """
    Creates a feature vector for each frame in ROI based on non-zero order diffraction peaks
    positions

    Parameters
    ----------
    ctx : libertem.api.Context
    dataset : libertem.io.dataset.DataSet
        A dataset with 1- or 2-D scan dimensions and 2-D frame dimensions
    num : int
        Number of possible peak positions to detect (better put higher value,
        the output is limited to the number of peaks the algorithm could find)
    delta : float
        Relative intensity difference between current frame and reference image for decision making
        for feature vector value (delta = (x-ref)/ref, so, normally, value should be in range [0,1])
    rad_in : int, optional
        Inner radius in pixels of a ring to mask region of interest of SD image to delete outliers
        for peak finding
    rad_out : int, optional
        Outer radius in pixels of a ring to mask region of interest of SD image to delete outliers
        for peak finding
    center : tuple, optional
        (y,x) - pixels, coordinates of a ring to mask region of interest of SD image
        to delete outliers for peak finding
    roi : numpy.ndarray, optional
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
    res_stat = run_stddev(ctx, dataset, roi)
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
    if min_dist is None:
        min_dist = 1
    coordinates = peak_local_max(masked_sstd, num_peaks=n_peaks, min_distance=min_dist)
    udf = FeatureVecMakerUDF(
        delta=delta, savg=savg, coordinates=coordinates
    )
    pass_results = ctx.run_udf(dataset=dataset, udf=udf, roi=roi)

    return (pass_results, coordinates)
