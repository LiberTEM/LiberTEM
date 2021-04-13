import logging

from .base import LivePlot


logger = logging.getLogger(__name__)


class BQLivePlot(LivePlot):
    """
    bqplot-image-gl-based live plot (experimental).

    Note
    ----
    As opposed to the matplotlib-base live plot, you need to explicitly call
    the :meth:`display` method to show the plot in a specific notebook cell.
    """
    def __init__(
            self, ds, udf, channel=None, postprocess=None,
    ):
        """
        Construct a new :class:`BQLivePlot` instance.

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
        """
        super().__init__(ds, udf, postprocess, channel)
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

    def update(self, force=False):
        self.image.image = self.data / self.data.max()
