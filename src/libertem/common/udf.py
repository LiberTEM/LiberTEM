from enum import Enum
from typing_extensions import Protocol, TypedDict, Literal
from typing import Union
from libertem.common.array_formats import NUMPY, SPARSE_COO, SPARSE_GCXS

import numpy as np


# markers for special values:
class TileDepthEnum(Enum):
    TILE_DEPTH_DEFAULT = object()


class TileSizeEnum(Enum):
    TILE_SIZE_BEST_FIT = object()


class TilingPreferences(TypedDict):
    depth: Union[int, TileDepthEnum]
    total_size: Union[float, int]


class UDFProtocol(Protocol):
    '''
    Parts of the UDF interface required for MIT code in LiberTEM
    '''
    USE_NATIVE_DTYPE = bool
    TILE_SIZE_BEST_FIT = TileSizeEnum.TILE_SIZE_BEST_FIT
    TILE_SIZE_MAX = np.inf
    TILE_DEPTH_DEFAULT = TileDepthEnum.TILE_DEPTH_DEFAULT
    TILE_DEPTH_MAX = np.inf
    BACKEND_NUMPY = 'numpy'
    BACKEND_CUPY = 'cupy'
    BACKEND_CUDA = 'cuda'
    FORMAT_NUMPY = NUMPY
    FORMAT_SPARSE_COO = SPARSE_COO
    FORMAT_SPARSE_GCXS = SPARSE_GCXS

    def get_method() -> Literal['tile', 'frame', 'partition']:
        raise NotImplementedError()

    def get_tiling_preferences() -> TilingPreferences:
        raise NotImplementedError()
