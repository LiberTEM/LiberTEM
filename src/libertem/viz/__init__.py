from empyre.vis.colors import ColormapCubehelix, ColormapPerception, ColormapHLS, ColormapClassic

from .base import visualize_simple, rgb_from_2dvector, get_plottable_2D_channels

from libertem.common.viz import encode_image


__all__ = ['ColormapCubehelix', 'ColormapPerception', 'ColormapHLS',
           'ColormapClassic', 'cmaps', 'CMAP_CIRCULAR_DEFAULT',
           'visualize_simple', 'encode_image', 'get_plottable_2D_channels', 'rgb_from_2dvector']


cmaps = {'cubehelix_standard': ColormapCubehelix(),
         'cubehelix_reverse': ColormapCubehelix(reverse=True),
         'cubehelix_circular': ColormapCubehelix(start=1, rot=1,
                                                 minLight=0.5, maxLight=0.5, sat=2),
         'perception_circular': ColormapPerception(),
         'hls_circular': ColormapHLS(),
         'classic_circular': ColormapClassic()}

CMAP_CIRCULAR_DEFAULT = cmaps['cubehelix_circular']
