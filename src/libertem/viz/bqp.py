import logging

import numpy as np

from .base import Live2DPlot


logger = logging.getLogger(__name__)


class BQLive2DPlot(Live2DPlot):
    """
    bqplot-image-gl-based live plot (experimental).
    """
    def __init__(
            self, dataset, udf, roi=None, channel=None, buffers=None
    ):
        """
        Construct a new :class:`BQLive2DPlot` instance.

        Parameters
        ----------
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
        """
        super().__init__(dataset, udf, roi, channel, buffers)
        # keep bqplot and bqplot_image_gl as optional dependencies
        from bqplot import Figure, LinearScale, Axis, ColorScale
        from bqplot_image_gl import ImageGL

        scale_x = LinearScale(min=0, max=1)
        scale_y = LinearScale(min=0, max=1)
        scales = {'x': scale_x,
                  'y': scale_y}
        axis_x = Axis(scale=scale_x, label='x')
        axis_y = Axis(scale=scale_y, label='y', orientation='vertical')

        s = self.data.shape
        aspect = s[1] / s[0]

        figure = Figure(
            scales=scales,
            axes=[axis_x, axis_y],
            scale_x=scale_x,
            scale_y=scale_y,
            min_aspect_ratio=aspect,
            max_aspect_ratio=aspect,
        )

        scales_image = {'x': scale_x,
                        'y': scale_y,
                        'image': ColorScale(min=0, max=1)}

        image = ImageGL(image=self.data, scales=scales_image)
        figure.marks = (image,)
        self.figure = figure
        self.image = image

    def display(self):
        return self.figure

    def update(self, damage, force=False):
        # TODO use damage for min and max
        damage = damage & np.isfinite(self.data)
        valid_data = self.data[damage]
        mmin = valid_data.min()
        mmax = valid_data.max()
        delta = mmax - mmin
        if delta <= 0:
            delta = 1
        # Map on color scale range 0..1
        self.image.image = (self.data - mmin) / delta
