from enum import Enum
from typing_extensions import Protocol, TypedDict, Literal
from typing import Union
from sparseconverter import (
    CUDA, CUPY, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, NUMPY,
    SCIPY_COO, SCIPY_CSC, SCIPY_CSR, SPARSE_COO, SPARSE_DOK, SPARSE_GCXS
)

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
    BACKEND_NUMPY = NUMPY
    BACKEND_CUPY = CUPY
    BACKEND_CUDA = CUDA
    BACKEND_SPARSE_COO = SPARSE_COO
    BACKEND_SPARSE_GCXS = SPARSE_GCXS
    BACKEND_SPARSE_DOK = SPARSE_DOK
    BACKEND_SCIPY_COO = SCIPY_COO
    BACKEND_SCIPY_CSR = SCIPY_CSR
    BACKEND_SCIPY_CSC = SCIPY_CSC
    BACKEND_CUPY_SCIPY_COO = CUPY_SCIPY_COO
    BACKEND_CUPY_SCIPY_CSR = CUPY_SCIPY_CSR
    BACKEND_CUPY_SCIPY_CSC = CUPY_SCIPY_CSC
    # Excludes sparse.DOK and CUDA, prefers scipy.sparse and GPU
    # Deprioritizes sparse.pydata.org due to their high call overhead
    BACKEND_ALL = (
        CUPY_SCIPY_CSR, CUPY_SCIPY_CSC, CUPY_SCIPY_COO,
        SCIPY_CSR, SCIPY_CSC, SCIPY_COO,
        CUPY, NUMPY,
        SPARSE_COO, SPARSE_GCXS,
    )

    def get_method() -> Literal['tile', 'frame', 'partition']:
        raise NotImplementedError()

    def get_tiling_preferences() -> TilingPreferences:
        raise NotImplementedError()
