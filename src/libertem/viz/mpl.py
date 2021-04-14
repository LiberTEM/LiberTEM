import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from .base import Live2DPlot


logger = logging.getLogger(__name__)


class MPLLive2DPlot(Live2DPlot):
    """
    Matplotlib-based live plot
    """
    def __init__(
            self, dataset, udf, roi=None, channel=None, buffers=None, title=None,
            cmap='viridis', min_delta=0.2, **kwargs,
    ):
        """
        Construct a new :class:`MPLLivePlot` instance.

        Parameters
        ----------

        dataset : DataSet
            The dataset on which the UDf will be run - needed to have access to
            concrete shapes for the plot results.

        udf : UDF
            The UDF instance this plot is associated to. This needs to be
            the same instance that is passed to :meth:`~libertem.api.Context.run_udf`.

        roi : numpy.ndarray or None
            Region of interest (ROI) that the UDF will be run on. This is necessary for UDFs
            where the `extra_shape` parameter of result buffers is a function of the ROI,
            such as :class:`~libertem.udf.raw.PickUDF`.

        channel : str or function udf_result -> ndarray
            The UDF result buffer name that should be plotted, or a function
            that derives a plottable ndarray from the UDF results.

        buffers : None or UDF result
            UDF result used to initialize the plot data and determine plot shape.
            If None (default), this is determined using
            :meth:`~libertem.udf.base.UDFRunner.dry_run`. This parameter allows re-using
            buffers to avoid unnecessary dry runs.

        cmap : str
            Colormap

        min_delta : float in seconds
            Don't update more frequently than this value.

        **kwargs
            Passed on to :code:`imshow`
        """
        super().__init__(dataset, udf, roi, channel, buffers)
        if title is None:
            title = f"{udf.__class__.__name__}: {self.channel}"
        self.title = title
        self.cmap = cmap
        self.kwargs = kwargs
        self.last_update = 0
        self.min_delta = min_delta
        self.fig = None
        self.axes = None
        self.im_obj = None

    def display(self):
        self.fig, self.axes = plt.subplots()
        self.im_obj = self.axes.imshow(self.data, cmap=self.cmap, **self.kwargs)
        self.axes.set_title(self.title)

    def update(self, force=False):
        """
        Update the plot based on `self.data`.

        Parameters
        ----------
        force : bool
            Force an update, disabling any throttling mechanisms
        """
        # Nothing to draw, not displayed
        if self.im_obj is None:
            assert self.fig is None
            assert self.axes is None
            return
        t0 = time.time()
        delta = t0 - self.last_update
        if (not force) and delta < self.min_delta:
            return  # don't update plot if we recently updated
        i_o = self.im_obj
        i_o.set_data(self.data)
        # Buffer is initialized by the UDF, by default NaN for floats
        valid_data = self.data[np.isfinite(self.data)]
        if len(valid_data) > 0:
            i_o.norm.vmin = np.min(valid_data)
            i_o.norm.vmax = np.max(valid_data)
        i_o.changed()
        self.fig.canvas.draw()
        self.last_update = time.time()
        logger.debug("MPLLivePlot updated in %.3f seconds", self.last_update - t0)
