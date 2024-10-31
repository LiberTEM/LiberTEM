import typing
import logging
import time

import numpy as np
# Makes colormaps available in matplotlib
import colorcet  # noqa:F401
from matplotlib import colors, cm, colormaps
from skimage.color import rgb2luv, luv2rgb

from libertem.udf.base import UDFRunner

# Import here for backwards compatibility, refs #1031
from libertem.common.viz import encode_image  # NOQA: F401

if typing.TYPE_CHECKING:
    from libertem.common.buffers import BufferWrapper

logger = logging.getLogger(__name__)


def _get_norm(result, norm_cls=colors.Normalize, vmin=None, vmax=None, damage=None):
    if (vmin is not None) and (vmax is not None):
        return norm_cls(vmin=vmin, vmax=vmax)

    result = result.astype(np.float32)

    if damage is None:
        damage = (result != 0)

    damage = damage & np.isfinite(result)

    if damage.sum() == 0:
        return norm_cls(vmin=1, vmax=1)  # all-NaN or all-zero

    if vmin is None:
        vmin = np.min(result[damage])
    if vmax is None:
        vmax = np.max(result[damage])

    return norm_cls(vmin=vmin, vmax=vmax)


def visualize_simple(result, colormap=None, logarithmic=False, vmin=None, vmax=None, damage=None):
    """
    Normalize and visualize ``result`` with ``colormap`` and return the
    resulting RGBA data as an array.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    colormap : matplotlib colormap or None
        colormap used for visualizing intensity values, defaults to matplotlib.cm.gist_earth

    Returns
    -------

    np.array
        A numpy array of shape (Y, X, 4) containing RGBA data, suitable for
        passing to `Image.fromarray` in PIL.
    """
    if logarithmic:
        # Convert to the smallest dtype that supports subtractions
        dtype = np.result_type(result, np.int8)
        result = result.astype(dtype)
        cnorm = colors.LogNorm
        result = result - np.min(result) + 1
    else:
        cnorm = colors.Normalize
    if colormap is None:
        colormap = cm.gist_earth
    norm = _get_norm(result, norm_cls=cnorm, vmin=vmin, vmax=vmax, damage=damage)
    shape = result.shape
    normalized = norm(result.reshape((-1,))).reshape(shape)
    colored = colormap(normalized, bytes=True)
    return colored


cet_cyclic_isoluminant = colormaps.get_cmap('cet_cyclic_isoluminant')


def rgb_from_2dvector(y, x, vmax=None):
    y = np.asarray(y)
    x = np.asarray(x)
    absval = np.sqrt(np.abs(y)**2 + np.abs(x)**2)
    if vmax is None:
        vmax = np.max(absval[np.isfinite(absval)])
    if vmax == 0:
        vmax = 1
    scaled_absval = absval / vmax
    scaled_absval = np.minimum(scaled_absval, 1)
    # Transform coordinates to match previous circular color map
    angle = np.arctan2(-y, -x)
    # Get color from isoluminant cyclic map of the angle
    color = np.asarray(visualize_simple(
        angle, colormap=cet_cyclic_isoluminant, vmin=-np.pi, vmax=np.pi
    ))
    # Transform to LUV space for linear axes
    luv = rgb2luv(color[..., :3])
    # Tweak luminance to match previous map
    luv[..., 0] *= 0.8
    # Adjust color intensity to roughly match previous implementation
    color_scale = scaled_absval * 2.2
    # Scale color
    luv[..., 1] *= color_scale
    luv[..., 2] *= color_scale
    rgb = (luv2rgb(luv) * 255).astype(np.uint8)
    return rgb


def make_colormap():
    angles = np.arange(360)
    angles_rad = angles/180*np.pi
    yy, xx = -np.sin(angles_rad), -np.cos(angles_rad)
    cols = rgb_from_2dvector(y=yy, x=xx) / 255
    cmap = colors.LinearSegmentedColormap.from_list('libertem_cyclic', cols)
    return cmap


libertem_cyclic = make_colormap()


def get_plottable_2D_channels(buffers):
    return [
        k
        for k, buf in buffers.items()
        # 2D data, removing axes of size 1
        if len(buf.data.squeeze().shape) == 2
    ]


class Live2DPlot:
    """
    Base plotting class for interactive use. Please see the subclasses for
    concrete details.

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
            self, dataset, udf, roi=None, channel=None, title=None, min_delta=0, udfresult=None
    ):
        if udfresult is None:
            udfresult = UDFRunner.dry_run([udf], dataset, roi)
        eligible_channels = get_plottable_2D_channels(udfresult.buffers[0])
        if channel is None:
            if not eligible_channels:
                raise ValueError(f"No plottable 2D channel found for {udf.__class__.__name__}")
            channel = eligible_channels[0]
            channel_title = channel

        if callable(channel):
            extract = channel
            channel_title = channel.__name__
            channel = None
        elif isinstance(channel, (tuple, list)):
            channel, func = channel
            if channel not in udfresult.buffers[0]:
                raise ValueError(
                    f"channel {channel} not found, have: {udfresult.buffers[0].keys()}"
                )
            kind = udfresult.buffers[0][channel].kind
            if kind == 'nav':
                def extract(udf_results, damage):
                    return (func(udf_results[channel].data), damage)
            else:
                def extract(udf_results, damage):
                    return (func(udf_results[channel].data), True)

            channel_title = f"{func.__name__}({channel})"
        else:
            extract = None
            if channel not in eligible_channels:
                raise ValueError("channel {} not found or not plottable, have: {!r}".format(
                    channel, eligible_channels
                ))
            channel_title = channel

        self._extract = extract
        self.channel = channel
        if title is None:
            title = f"{udf.__class__.__name__}: {channel_title}"
        self.title = title
        self.data, _ = self.extract(udfresult.buffers[0], udfresult.damage)
        self.udf = udf
        self.last_update = 0
        self.min_delta = min_delta

    def get_udf(self):
        """
        Returns the associated UDF instance
        """
        return self.udf

    def extract(
        self, udf_results: dict[str, "BufferWrapper"], damage: "BufferWrapper"
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Extract plotting data from UDF result.


        Parameters
        ----------

        udf_results : UDF result
            (Partial) UDF result

        damage : BufferWrapper
            :class:`~libertem.common.buffers.BufferWraper` with :code:`kind='nav'`
            and :code:`dtype=bool` that indicates the area of the nav dimension that
            has been processed by the UDF already.

        Returns
        -------
        (numpy.ndarray, damage)
            It returns the data and damage of a UDF result buffer indicated by :code:`channel`
            if that is a string, or a numpy.ndarray and damage derived from the UDF results by
            calling :code:`channel` with this method's arguments if it is callable.
        """
        if self._extract is None:
            buffer = udf_results[self.channel]
            squeezed = buffer.masked_data.squeeze()
            res_damage = buffer.valid_mask.squeeze()
            return (squeezed, res_damage)
        else:
            return self._extract(udf_results, damage)

    def new_data(self, udf_results: dict[str, "BufferWrapper"], damage, force: bool = False):
        """
        This method is called with the raw `udf_results` any time a new
        partition has finished processing.

        The :code:`damage` parameter is filtered to only cover finite
        values of :code:`self.data` and passed to :meth:`self.update`,
        which should then be implemented by a subclass.
        """
        t0 = time.time()
        delta = t0 - self.last_update
        if (not force) and delta < self.min_delta:
            return  # don't update plot if we recently updated
        (self.data, damage) = self.extract(udf_results, damage)
        finite = np.isfinite(self.data)
        # duck typing "array-like" since damage can also be bool
        if hasattr(damage, "shape"):
            start_axis = len(damage.shape)
            stop_axis = len(self.data.shape)
            # Reduce surplus axes, for example RGB plotting #1052
            if stop_axis > start_axis:
                finite = np.all(finite, axis=tuple(range(start_axis, stop_axis)))
        damage = damage & finite
        self.update(damage, force=force)
        self.last_update = time.time()
        logger.debug("%s updated in %.3f seconds", self.__class__.__name__, self.last_update - t0)

    def update(self, damage, force=False):
        """
        Update the plot based on `self.data`.

        Parameters
        ----------
        damage : numpy.ndarray Boolean array with the shape of
            :code:`self.data`. It is :code:`True` for all positions in
            :code:`self.data` that contain finite values and have potentially
            been touched by the UDF. This can be used to extract the correct
            plot range by ignoring invalid buffer portions.

        force : bool Force an update, disabling any throttling mechanisms
        """
        raise NotImplementedError()

    def display(self):
        """
        Show the plot, for example in the current Jupyter cell.
        """
        raise NotImplementedError()


class Dummy2DPlot(Live2DPlot):
    '''
    No-op plot. This is useful for test and example code to not
    attempt displaying :code:`matplotlib` plots in a headless environment
    or during batch operation.
    '''
    def update(self, damage, force=False):
        pass

    def display(self):
        pass
