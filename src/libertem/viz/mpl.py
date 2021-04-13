import time
import logging

import numpy as np
import matplotlib.pyplot as plt

from .base import LivePlot


logger = logging.getLogger(__name__)


class MPLLivePlot(LivePlot):
    def __init__(
            self, ds, udf, channel=None, postprocess=None,
            cmap='viridis', min_delta=0.2, **kwargs,
    ):
        super().__init__(ds, udf, postprocess, channel)
        self.fig, self.axes = plt.subplots()
        self.im_obj = self.axes.imshow(self.data, cmap=cmap, **kwargs)
        self.last_update = 0
        self.min_delta = min_delta

    def update(self, force=False):
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
