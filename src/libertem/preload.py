# This module pre-loads modules that can have a long load time.
# To be used to avoid https://github.com/LiberTEM/LiberTEM/issues/218
# where loading modules makes workers unresponsive for too long.
# Usage: dask-worker [...] --preload libertem.preload [...]

# Disable flake8 because we import a lot of modules without using them.

# flake8: noqa

import numpy
import sparse
from matplotlib import colors, cm

try:
    import torch
except ImportError:
    pass

import libertem

try:
    import hdf5plugin
except ImportError:
    pass

import numba.cuda  # see also: #1432

from libertem.common.backend import set_file_limit
set_file_limit()
