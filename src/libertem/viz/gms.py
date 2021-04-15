import time

import numpy as np

from .base import Live2DPlot


class GMSLive2DPlot(Live2DPlot):
    """
    Live plot for Gatan Microscopy Suite, Digital Micrograph (experimental).

    This works with Python scripting within GMS
    """
    def __init__(
            self, DM, dataset, udf, roi=None, channel=None, buffers=None, title=None,
            min_delta=0.2
    ):
        """
        Construct a new :class:`GMSLive2DPlot` instance.

        Parameters
        ----------
        DM : DM
            The DM module used for interfacing between GMS and Python
            TODO how to import that in a submodule?
        dataset : DataSet
            The dataset on which the UDf will be run. This allows to determine the
            shape of the plots for initialization.

        udf : UDF
            The UDF instance this plot is associated to. This needs to be
            the same instance that is passed to :meth:`~libertem.api.Context.run_udf`.

        roi : numpy.ndarray or None
            Region of interest (ROI) that the UDF will be run on. This is necessary for UDFs
            where the `extra_shape` parameter of result buffers is a function of the ROI,
            such as :class:`~libertem.udf.raw.PickUDF`.

        channel : str or function udf_result -> ndarray
            The UDF result buffer name that should be plotted, or a function
            that derives a plottable 2D ndarray from the full UDF results.

        buffers : None or udf_result
            UDF result used to initialize the plot data and determine plot shape.
            If None (default), this is determined using
            :meth:`~libertem.udf.base.UDFRunner.dry_run`. This parameter allows re-using
            buffers to avoid unnecessary dry runs.

        cmap : str
            Colormap

        min_delta : float in seconds
            Don't update more frequently than this value.
        """
        super().__init__(dataset, udf, roi, channel, buffers)
        self.DM = DM
        self.image = DM.CreateImage(self.data.copy())
        self.window = None
        self.disp = None
        if title is None:
            title = f"{udf.__class__.__name__}: {self.channel}"
        self.image.SetName(title)
        self.last_update = 0
        self.min_delta = min_delta

    def display(self):
        self.window = self.image.ShowImage()
        self.disp = self.image.GetImageDisplay(0)
        self.disp.SetDoAutoSurvey(False)

    def update(self, damage, force=False):
        t0 = time.time()
        delta = t0 - self.last_update
        if (not force) and delta < self.min_delta:
            return  # don't update plot if we recently updated
        if self.disp is None:
            assert self.window is None
            return
        damage = damage & np.isfinite(self.data)
        valid_data = self.data[damage]
        if len(valid_data) > 0:
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
            self.disp.SetContrastLimits(float(vmin), float(vmax))
        buffer = self.image.GetNumArray()
        buffer[:] = self.data
        self.image.UpdateImage()
        self.last_update = time.time()
