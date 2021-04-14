import logging
from io import BytesIO

import numpy as np
from matplotlib import colors, cm
from PIL import Image

from libertem.udf.base import UDFRunner

logger = logging.getLogger(__name__)


def _get_norm(result, norm_cls=colors.Normalize, vmin=None, vmax=None):
    # TODO: only normalize across the area where we already have values
    # can be accomplished by calculating min/max over are that was
    # affected by the result tiles. for now, ignoring 0 works fine

    result = result.astype(np.float32)

    valid_mask = (result != 0) & ~np.isnan(result)
    if valid_mask.sum() == 0:
        return norm_cls(vmin=1, vmax=1)  # all-NaN or all-zero

    if vmin is None:
        vmin = np.min(result[valid_mask])
    if vmax is None:
        vmax = np.max(result[valid_mask])

    return norm_cls(vmin=vmin, vmax=vmax)


def encode_image(result, save_kwargs=None):
    """
    Save the RGBA data in ``result`` to an image with parameters ``save_kwargs``
    passed to ``PIL.Image.save``.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    save_kwargs : dict or None
        dict of kwargs passed to Pillow when saving the image, can be used to set
        the file format, quality, ...

    Returns
    -------

    BytesIO
        a buffer containing the result image (as PNG/JPG/... depending on save_kwargs)
    """
    if save_kwargs is None:
        save_kwargs = {'format': 'png'}
    # see also: https://stackoverflow.com/a/10967471/540644
    im = Image.fromarray(result)
    buf = BytesIO()
    im = im.convert(mode="RGB")
    im.save(buf, **save_kwargs)
    buf.seek(0)
    return buf


def visualize_simple(result, colormap=None, logarithmic=False, vmin=None, vmax=None):
    """
    Normalize and visualize ``result`` with ``colormap`` and return the
    resulting RGBA data as an array.

    Parameters
    ----------
    result : numpy.ndarray
        2d array of intensity values

    colormap : matplotlib colormap or None
        colormap used for visualizing intensity values, defaults to ColormapCubehelix()

    Returns
    -------

    np.array
        A numpy array of shape (Y, X, 4) containing RGBA data, suitable for
        passing to `Image.fromarray` in PIL.
    """
    if logarithmic:
        cnorm = colors.LogNorm
        result = result - np.min(result) + 1
    else:
        cnorm = colors.Normalize
    if colormap is None:
        colormap = cm.gist_earth
    norm = _get_norm(result, norm_cls=cnorm, vmin=vmin, vmax=vmax)
    shape = result.shape
    normalized = norm(result.reshape((-1,))).reshape(shape)
    colored = colormap(normalized, bytes=True)
    return colored


def get_plottable_2D_channels(buffers):
    return [
        k
        for k, buf in buffers.items()
        # 2D data, removing axes of size 1
        if len(buf.data.squeeze().shape) == 2
    ]


class Live2DPlot:
    """
    Base plotting class for interactive use. Please see the subclasses for concrete details.
    """
    def __init__(
            self, dataset, udf, roi=None, channel=None, buffers=None
    ):
        """
        Construct a new `LivePlot`

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

        buffers : None or UDF result
            UDF result used to initialize the plot data and determine plot shape.
            If None (default), this is determined using
            :meth:`~libertem.udf.base.UDFRunner.dry_run`. This parameter allows re-using
            buffers to avoid unnecessary dry runs.
        """
        if buffers is None:
            buffers = UDFRunner.dry_run([udf], dataset, roi).buffers[0]
        eligible_channels = get_plottable_2D_channels(buffers)
        if channel is None:
            assert len(eligible_channels) > 0, "should have at least one plottable channel"
            channel = eligible_channels[0]

        if callable(channel):
            extract = channel
            channel = channel.__name__
        else:
            extract = None
            if channel not in eligible_channels:
                raise ValueError("channel %s not found or not plottable, have: %r" % (
                    channel, eligible_channels
                ))

        self._extract = extract
        self.channel = channel
        self.data = self.extract(buffers)
        self.udf = udf

    def get_udf(self):
        """
        Returns the associated UDF instance
        """
        return self.udf

    def extract(self, udf_results):
        """
        Extract plotting data from UDF result
        """
        if self._extract is None:
            return udf_results[self.channel].data.squeeze()
        else:
            return self._extract(udf_results)

    def new_data(self, udf_results, damage, force=False):
        """
        This method is called with the raw `udf_results` any time a new
        partition has finished processing.
        """
        self.data = self.extract(udf_results)
        self.update(damage, force=force)

    def update(self, damage, force=False):
        """
        Update the plot based on `self.data`. This should be implemented by subclasses.

        Parameters
        ----------
        force : bool
            Force an update, disabling any throttling mechanisms
        """
        raise NotImplementedError()

    def display(self):
        """
        Show the plot ("bind it to the current jupyter cell")
        """
        raise NotImplementedError()
