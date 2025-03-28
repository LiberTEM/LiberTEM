import warnings

import numpy as np

from .base import Live2DPlot


class GMSLive2DPlot(Live2DPlot):
    """
    Live plot for Gatan Microscopy Suite, Digital Micrograph (experimental).

    This works with Python scripting within GMS

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
    """
    def __init__(
            self, dataset, udf, roi=None, channel=None, title=None,
            min_delta=0.2, udfresult=None,

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
        # Optional dependency, don't import top-level
        import DigitalMicrograph as DM

        self.image = DM.CreateImage(np.array(self.data))
        self.window = None
        self.disp = None
        self.image.SetName(self.title)

    def display(self):
        '''
        Use :meth:`DM.ShowImage` to create an image display.
        '''
        self.window = self.image.ShowImage()
        self.disp = self.image.GetImageDisplay(0)
        self.disp.SetDoAutoSurvey(False)

    def update(self, damage, force=False):
        if self.disp is None:
            assert self.window is None
            warnings.warn(
                "Plot is not displayed, not plotting. "
                "Call display() to display the plot."
            )
            return
        valid_data = self.data[damage]
        if len(valid_data) > 0:
            vmin = np.min(valid_data)
            vmax = np.max(valid_data)
            self.disp.SetContrastLimits(float(vmin), float(vmax))
        buffer = self.image.GetNumArray()
        buffer[:] = self.data
        self.image.UpdateImage()
