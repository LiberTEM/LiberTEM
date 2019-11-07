import numpy as np
import matplotlib.pyplot as plt

import libertem.masks as masks
import libertem.analysis.gridmatching as grm

from .patterns import MatchPattern

# FIXME There's work on flexible FFT backends in scipy
# https://github.com/scipy/scipy/wiki/GSoC-2019-project-ideas#revamp-scipyfftpack
# and discussions about pyfftw performance vs other implementations
# https://github.com/pyFFTW/pyFFTW/issues/264
# For that reason we shoud review the state of Python FFT implementations
# regularly and adapt our choices accordingly
try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros


def feature_vector(imageSizeX, imageSizeY, peaks, match_pattern: MatchPattern):
    '''
    This function generates a sparse mask stack to extract a feature vector.

    A match template based on the parameters in :code:`parameters` is placed at
    each peak position in an individual mask layer. This mask stack can then
    be used in :meth:`~libertem.api.Context.create_mask_job` to generate a feature vector for each
    frame.

    Summing up the mask stack along the first axis generates a mask that can be used for virtual
    darkfield imaging of all peaks together.

    Parameters
    ----------

    imageSizeX,imageSizeY : int
        Frame size in px
    peaks : numpy.ndarray
        Peak positions in px as numpy.ndarray of shape (n, 2) with integer type
    match_pattern : MatchPattern
        Instance of :class:`~MatchPattern`
    '''
    crop_size = match_pattern.get_crop_size()
    return masks.sparse_template_multi_stack(
        mask_index=range(len(peaks)),
        offsetX=peaks[:, 1] - crop_size,
        offsetY=peaks[:, 0] - crop_size,
        template=match_pattern.get_mask((2*crop_size + 1, 2*crop_size + 1)),
        imageSizeX=imageSizeX,
        imageSizeY=imageSizeY,
    )


def visualize_frame(ctx, ds, result, indices, r, y, x, axes, colors=None, stretch=10):
    '''
    Visualize the refinement of a specific frame in matplotlib axes
    '''
    # Get the frame from the dataset
    get_sample_frame = ctx.create_pick_analysis(dataset=ds, y=y, x=x)
    sample_frame = ctx.run(get_sample_frame)

    d = sample_frame[0].raw_data

    pcm = axes.imshow(np.log(d - np.min(d) + 1))

    refined = result['refineds'].data[y, x]
    elevations = result['peak_elevations'].data[y, x]
    selector = result['selector'].data[y, x]

    max_elevation = np.max(elevations)

    # Calclate the best fit positions to compare with the
    # individual peak positions.
    # A difference between best fit and individual peaks highlights outliers.
    calculated = grm.calc_coords(
        zero=result['zero'].data[y, x],
        a=result['a'].data[y, x],
        b=result['b'].data[y, x],
        indices=indices
    )

    paint_markers(
        axes=axes,
        r=r,
        refined=refined,
        normalized_elevations=elevations/max_elevation,
        calculated=calculated,
        selector=selector,
        zero=result['zero'].data[y, x],
        a=result['a'].data[y, x],
        b=result['b'].data[y, x],
        colors=colors,
        stretch=stretch,
    )
    return pcm


def paint_markers(axes, r, refined, normalized_elevations, calculated, selector, zero, a, b,
        colors=None, stretch=10):
    if colors is None:
        colors = {
            'marker': 'w',
            'arrow': 'r',
            'missing': 'r',
            'a': 'b',
            'b': 'g',
        }

    axes.arrow(*np.flip(zero), *(np.flip(a)), color=colors['a'])
    axes.arrow(*np.flip(zero), *(np.flip(b)), color=colors['b'])

    # Plot markers for the individual peak positions.
    # The alpha channel represents the peak elevation, which is used as a weight in the fit.
    for i in range(len(refined)):
        p = np.flip(refined[i])
        a = max(0, normalized_elevations[i])
        p0 = np.flip(calculated[i])
        if selector[i]:
            axes.add_artist(plt.Circle(p, r, color=colors['marker'], fill=False, alpha=a))
            axes.add_artist(plt.Circle(p0, 1, color=colors['arrow'], fill=True, alpha=a))
            axes.arrow(*p0, *(p-p0)*stretch, color=colors['arrow'], alpha=a)
        else:
            (yy, xx) = calculated[i]
            xy = (xx - r, yy - r)
            axes.add_artist(plt.Rectangle(xy, 2*r, 2*r, color=colors['missing'], fill=False))
