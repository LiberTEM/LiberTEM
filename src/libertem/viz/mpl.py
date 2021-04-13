import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from .base import LivePlot


logger = logging.getLogger(__name__)


class MPLLivePlot(LivePlot):
    """
    Matplotlib-based live plot
    """
    def __init__(
            self, ds, udf, channel=None, postprocess=None,
            cmap='viridis', min_delta=0.2, **kwargs,
    ):
        """
        Construct a new :class:`MPLLivePlot` instance.

        Parameters
        ----------

        ds : DataSet
            The dataset on which the UDf will be run - needed to have access to
            concrete shapes for the plot results.

        udf : UDF
            The UDF instance this plot is associated to. This needs to be
            the same instance that is passed to :meth:`~libertem.api.Context.run_udf`.

        postprocess : function ndarray -> ndarray
            Optional postprocessing function, identity by default.

        channel : str
            The UDF result buffer name that should be plotted.

        cmap : str
            Colormap

        min_delta : float in seconds
            Don't update more frequently than this value.

        **kwargs
            Passed on to :code:`imshow`
        """
        super().__init__(ds, udf, postprocess, channel)
        self.fig, self.axes = plt.subplots()
        self.im_obj = self.axes.imshow(self.data, cmap=cmap, **kwargs)
        self.axes.set_title(f"{udf.__class__.__name__}: {self.channel}")
        self.last_update = 0
        self.min_delta = min_delta

    def update(self, force=False):
        """
        Update the plot based on `self.data`. This should be implemented by subclasses.

        Parameters
        ----------
        force : bool
            Force an update, disabling any throttling mechanisms
        """
        delta = time.time() - self.last_update
        if delta < self.min_delta or force:
            return  # don't update plot if we recently updated
        t0 = time.time()
        i_o = self.im_obj
        i_o.set_data(self.data)
        nonzero_data = self.data[self.data != 0]
        if len(nonzero_data) > 0:
            i_o.norm.vmin = np.min(nonzero_data)
            i_o.norm.vmax = np.max(np.max(nonzero_data))
        i_o.changed()
        self.fig.canvas.draw()
        logger.debug("MPLLivePlot updated in %.3f seconds", time.time() - t0)
