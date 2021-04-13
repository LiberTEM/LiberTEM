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


def get_plottable_channels(udf, dataset):
    from libertem.udf.base import UDFRunner

    bufs = UDFRunner.inspect_udf(udf, dataset)

    return [
        k
        for k, buf in bufs.items()
        if (buf.kind in ('sig', 'nav') and buf.extra_shape == ())
        or (buf.kind == 'single' and len(buf.extra_shape) == 2)
    ]


class LivePlot:
    """
    Base plotting class for interactive use. Please see the subclasses for concrete details.
    """
    def __init__(
            self, ds, udf, postprocess=None, channel=None,
    ):
        """
        Construct a new `LivePlot`

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
        eligible_channels = get_plottable_channels(udf, ds)
        if channel is None:
            assert len(eligible_channels) > 0, "should have at least one plottable channel"
            channel = eligible_channels[0]

        if channel not in eligible_channels:
            raise ValueError("channel %s not found or not plottable, have: %r" % (
                channel, eligible_channels
            ))

        buf = UDFRunner.inspect_udf(udf, ds)[channel]
        kind = buf.kind
        if kind == 'sig':
            shape = ds.shape.sig
        elif kind == 'nav':
            shape = ds.shape.nav
        elif kind == 'single':
            shape = buf.extra_shape
        else:
            raise ValueError("unknown plot kind")

        self.shape = shape
        self.data = np.zeros(shape, dtype=np.float32)
        self.channel = channel
        self.pp = postprocess or (lambda x: x)
        self.udf = udf

    def get_udf(self):
        """
        Returns the associated UDF instance
        """
        return self.udf

    def postprocess(self, udf_results):
        """
        Optional post-processing, before the data is visualized
        (useful, for example, for quick-and-dirty custom re-scaling)
        """
        return self.pp(udf_results[self.channel].data)

    def new_data(self, udf_results, force=False):
        """
        This method is called with the raw `udf_results` any time a new
        partition has finished processing.
        """
        self.data[:] = self.postprocess(udf_results)
        self.update(force=force)

    def update(self, force=False):
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
        pass
