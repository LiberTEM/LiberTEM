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
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0.5, udfresult=None,
            cmap='viridis', **kwargs,
    ):
        """
        Construct a new :class:`MPLLivePlot` instance.

        Parameters
        ----------
    dataset : DataSet
            The dataset on which the UDf will be run. This allows
            to determine the shape of the plots for initialization.
        udf : UDF
            The UDF instance this plot is associated to. This needs to be
            the same instance that is passed to
            :meth:`~libertem.api.Context.run_udf`.
        roi : numpy.ndarray or None
            Region of interest (ROI) that the UDF will
            be run on. This is necessary for UDFs where the `extra_shape`
            parameter of result buffers is a function of the ROI, such as
            :class:`~libertem.udf.raw.PickUDF`.
        channel : str or function (udf_result, damage) -> (ndarray, damage)
            The UDF result buffer name that should be plotted, or a function that
            derives a plottable 2D ndarray and damage indicator from the full
            UDF results and the processed nav space.

            The function receives the partial result of the UDF together with :code:`damage`, a
            :class:`~libertem.common.buffers.BufferWraper` with :code:`kind='nav'`
            and :code:`dtype=bool` that indicates the area of the nav dimension that
            has been processed by the UDF already.

            If the extracted value is derived from :code:`kind='nav'`buffers,
            the function can just pass through :code:`damage`
            as its return value. If it is unrelated to the navigations space, for example
            :code:`kind='sig'` or :code:`kind='single'`, the function can return :code:`True`
            to indicate that the entire buffer was updated. The damage information
            is currently used to determine the correct plot range by ignoring the
            buffer's initialization value.

            If no channel is given, the first plottable (2D) channel of the UDF
            is chosen.
        title : str
            The plot title. By default UDF class name and channel name.
        min_delta : float
            Minimum time span in seconds between updates to reduce overheads for slow plotting.
        udfresult : None or UDF result
            UDF result used to initialize the plot
            data and determine plot shape. If None (default), this is determined
            using :meth:`~libertem.udf.base.UDFRunner.dry_run` on the dataset, UDF and ROI.
            This parameter allows re-using buffers to avoid unnecessary dry runs.
        cmap : str
            Colormap

        **kwargs
            Passed on to :code:`imshow`
        """
        super().__init__(
            dataset=dataset,
            udf=udf,
            roi=roi,
            channel=channel,
            title=title,
            min_delta=min_delta,
            udfresult=udfresult
        )
        self.cmap = cmap
        self.kwargs = kwargs
        self.fig = None
        self.axes = None
        self.im_obj = None

    def display(self):
        self.fig, self.axes = plt.subplots()
        self.im_obj = self.axes.imshow(self.data, cmap=self.cmap, **self.kwargs)
        self.axes.set_title(self.title)

    def update(self, damage, force=False):
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

        i_o = self.im_obj
        i_o.set_data(self.data)
        valid_data = self.data[damage]
        if len(valid_data) > 0:
            i_o.norm.vmin = np.min(valid_data)
            i_o.norm.vmax = np.max(valid_data)
        i_o.changed()
        self.fig.canvas.draw()
