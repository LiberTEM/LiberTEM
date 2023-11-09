import logging

import numpy as np

from .base import Live2DPlot


logger = logging.getLogger(__name__)


class BQLive2DPlot(Live2DPlot):
    """
    bqplot-image-gl-based live plot.

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
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=1/60, udfresult=None
    ):
        super().__init__(
            dataset=dataset,
            udf=udf,
            roi=roi,
            channel=channel,
            title=title,
            min_delta=min_delta,
            udfresult=udfresult,
        )
        # keep bqplot and bqplot_image_gl as optional dependencies
        from bqplot import Figure, LinearScale, Axis, ColorScale
        from bqplot_image_gl import ImageGL

        scale_x = LinearScale(min=0, max=1)
        # Make sure y points down
        # See https://libertem.github.io/LiberTEM/concepts.html#coordinate-system
        scale_y = LinearScale(min=1, max=0)
        axis_x = Axis(scale=scale_x, label='x')
        axis_y = Axis(scale=scale_y, label='y', orientation='vertical')

        s = self.data.shape
        aspect = s[1] / s[0]

        figure = Figure(
            axes=[axis_x, axis_y],
            scale_x=scale_x,
            scale_y=scale_y,
            min_aspect_ratio=aspect,
            max_aspect_ratio=aspect,
            title=self.title
        )

        color_scale = ColorScale(min=0, max=1)

        scales_image = {'x': scale_x,
                        'y': scale_y,
                        'image': color_scale}

        dtype = np.result_type(self.data, np.int8)
        image = ImageGL(image=self.data.astype(dtype), scales=scales_image)
        figure.marks = (image,)
        self.figure = figure
        self.image = image
        self.color_scale = color_scale

    def display(self):
        from IPython.display import display
        display(self.figure)

    def update(self, damage, force=False):
        dtype = np.result_type(self.data, np.int8)
        # Map on dtype that supports subtraction
        valid_data = self.data[damage].astype(dtype)
        valid_data = valid_data[np.isfinite(valid_data)]
        if valid_data.size > 0:
            mmin = valid_data.min()
            mmax = valid_data.max()
        else:
            mmin = 1
            mmax = 1 + 1e-12
        delta = mmax - mmin
        if delta <= 0:
            delta = 1
        # Map on color scale range 0..1
        self.image.image = (self.data.astype(dtype) - mmin) / delta
