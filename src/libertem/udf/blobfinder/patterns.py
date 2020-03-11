import warnings

from libertem_blobfinder.common.patterns import MatchPattern, Circular, RadialGradient, BackgroundSubtraction, UserTemplate, RadialGradientBackgroundSubtraction  # noqa: F401,E501


warnings.warn(
    "Blobfinder has moved to its own package LiberTEM-blobfinder with a new structure. "
    "Please see https://libertem.github.io/LiberTEM-blobfinder/index.html for installation "
    "instructions and documentation of the new structure. "
    "Imports from libertem.udf.blobfinder are supported until LiberTEM release 0.6.0.",
    FutureWarning
)
