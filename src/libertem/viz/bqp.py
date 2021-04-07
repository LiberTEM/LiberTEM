import logging

from .base import LivePlot


logger = logging.getLogger(__name__)


class BQLivePlot(LivePlot):
    def __init__(
            self, ds, udf, channel=None, postprocess=None,
    ):
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
