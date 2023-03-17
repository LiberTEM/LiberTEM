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
    USE_NATIVE_DTYPE = bool  #: Neutral element for type conversion
    TILE_SIZE_BEST_FIT = TileSizeEnum.TILE_SIZE_BEST_FIT  #: Suggest using recommended tile size
    TILE_SIZE_MAX = np.inf  #: Suggest using maximum tile size
    TILE_DEPTH_DEFAULT = TileDepthEnum.TILE_DEPTH_DEFAULT  #: Suggest using recommended tile depth
    TILE_DEPTH_MAX = np.inf  #: Suggest using maximum tile depth
    BACKEND_NUMPY = NUMPY  #: NumPy array
    BACKEND_CUPY = CUPY  #: CuPy array
    BACKEND_CUDA = CUDA  #: NumPy array, but run on CUDA device class
    BACKEND_SPARSE_COO = SPARSE_COO  #: sparse.COO array
    BACKEND_SPARSE_GCXS = SPARSE_GCXS  #: sparse.GCXS array
    BACKEND_SPARSE_DOK = SPARSE_DOK  #: sparse.DOK array -- not recommended since slow!
    BACKEND_SCIPY_COO = SCIPY_COO  #: scipy.sparse.coo_matrix
    BACKEND_SCIPY_CSR = SCIPY_CSR  #: scipy.sparse.csr_matrix
    BACKEND_SCIPY_CSC = SCIPY_CSC  #: scipy.sparse.csc_matrix
    BACKEND_CUPY_SCIPY_COO = CUPY_SCIPY_COO  #: cupyx.scipy.sparse.coo_matrix
    BACKEND_CUPY_SCIPY_CSR = CUPY_SCIPY_CSR  #: cupyx.scipy.sparse.csr_matrix
    BACKEND_CUPY_SCIPY_CSC = CUPY_SCIPY_CSC  #: cupyx.scipy.sparse.csc_matrix
    # Excludes sparse.DOK and CUDA, prefers scipy.sparse and GPU
    # Deprioritizes sparse.pydata.org due to their high call overhead
    #: Tuple with all backends in suggested priority
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
