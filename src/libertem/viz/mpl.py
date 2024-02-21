import logging
import warnings
from contextlib import contextmanager

import numpy as np
import matplotlib.pyplot as plt
from jupyter_ui_poll import ui_events

from .base import Live2DPlot


logger = logging.getLogger(__name__)

try:
    with ui_events() as poll:
        pass
except Exception:
    logger.info(
            'Deactivating Jupyter UI event polling, '
            'possibly not running in IPython?',
            exc_info=True
        )

    # Replacing jupyter_ui_poll with a dummy
    @contextmanager
    def ui_events():
        yield lambda x: None


class MPLLive2DPlot(Live2DPlot):
    """
    Matplotlib-based live plot

    .. versionadded:: 0.7.0

    Parameters
    ----------
    dataset : DataSet
        The dataset on which the UDF will be run. This allows
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
    channel : misc
        Indicate the channel to be plotted.

        - :code:`None`: The first plottable (2D) channel of the UDF is plotted.
        - :code:`str`: The UDF result buffer name that should be plotted.
        - :code:`tuple(str, function(ndarray) -> ndarray)`: The UDF result buffer name that
          should be plotted together with a function that extracts a plottable result
        - :code:`function(udf_result, damage) -> (ndarray, damage)`: Function that derives a
          plottable 2D ndarray and damage indicator from the full
          UDF results and the processed nav space. See :ref:`plotting` for more details!
    title : str
        The plot title. By default UDF class name and channel name.
    min_delta : float
        Minimum time span in seconds between updates to reduce overheads for slow plotting.
    udfresult : UDFResults, optional
        UDF result to initialize the plot data and determine plot shape. If None (default),
        this is determined using :meth:`~libertem.udf.base.UDFRunner.dry_run` on the dataset,
        UDF and ROI. This parameter allows re-using buffers to avoid unnecessary dry runs.

    **kwargs
        Passed on to :code:`imshow`
    """
    def __init__(
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0.5, udfresult=None,
            **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            udf=udf,
            roi=roi,
            channel=channel,
            title=title,
            min_delta=min_delta,
            udfresult=udfresult
        )
        self.kwargs = kwargs
        self.fig = None
        self.axes = None
        self.im_obj = None

    def display(self):
        with ui_events() as poll:
            self.fig, self.axes = plt.subplots()
            self.im_obj = self.axes.imshow(self.data, **self.kwargs)
            # Set values compatible with log norm
            self.im_obj.norm.vmin = 1
            self.im_obj.norm.vmax = 1 + 1e-12
            self.axes.set_title(self.title)
            self.fig.show()
            poll(1000)

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
            warnings.warn(
                "Plot is not displayed, not plotting. "
                "Call display() to display the plot."
            )
            return

        i_o = self.im_obj
        i_o.set_data(self.data)
        valid_data = self.data[damage]
        valid_data = valid_data[np.isfinite(valid_data)]
        if len(valid_data) > 0:
            i_o.norm.vmin = np.min(valid_data)
            i_o.norm.vmax = np.max(valid_data)
        with ui_events() as poll:
            i_o.changed()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            poll(1000)
