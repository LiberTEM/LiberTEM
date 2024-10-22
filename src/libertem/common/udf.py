from enum import Enum
from typing_extensions import Protocol, TypedDict
from typing import Union
from sparseconverter import (
    CUDA, CUPY, CUPY_SCIPY_COO, CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, NUMPY,
    SCIPY_COO, SCIPY_CSC, SCIPY_CSR, SCIPY_COO_ARRAY, SCIPY_CSC_ARRAY, SCIPY_CSR_ARRAY,
    SPARSE_COO, SPARSE_DOK, SPARSE_GCXS,
    CPU_BACKENDS, CUDA_BACKENDS, CUPY_BACKENDS, SPARSE_BACKENDS, DENSE_BACKENDS,
    ND_BACKENDS, D2_BACKENDS,
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


class UDFMethod(Enum):
    TILE = 'tile'
    FRAME = 'frame'
    PARTITION = 'partition'


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
    BACKEND_SCIPY_COO_ARRAY = SCIPY_COO_ARRAY  #: scipy.sparse.coo_array
    BACKEND_SCIPY_CSR_ARRAY = SCIPY_CSR_ARRAY  #: scipy.sparse.csr_array
    BACKEND_SCIPY_CSC_ARRAY = SCIPY_CSC_ARRAY  #: scipy.sparse.csc_array
    BACKEND_CUPY_SCIPY_COO = CUPY_SCIPY_COO  #: cupyx.scipy.sparse.coo_matrix
    BACKEND_CUPY_SCIPY_CSR = CUPY_SCIPY_CSR  #: cupyx.scipy.sparse.csr_matrix
    BACKEND_CUPY_SCIPY_CSC = CUPY_SCIPY_CSC  #: cupyx.scipy.sparse.csc_matrix
    # Excludes sparse.DOK, numpy.matrix and CUDA, prefers scipy.sparse and GPU
    # Deprioritizes sparse.pydata.org due to their high call overhead
    #: Tuple with all backends in suggested priority
    BACKEND_ALL = (
        CUPY_SCIPY_CSR, CUPY_SCIPY_CSC, CUPY_SCIPY_COO,
        SCIPY_COO_ARRAY, SCIPY_CSC_ARRAY, SCIPY_CSR_ARRAY,
        SCIPY_CSR, SCIPY_CSC, SCIPY_COO,
        CUPY, NUMPY,
        SPARSE_COO, SPARSE_GCXS,
    )

    CPU_BACKENDS = CPU_BACKENDS  #: Set of backends that run on device class CPU
    CUDA_BACKENDS = CUDA_BACKENDS  #: Set of backends that run on device class CUDA
    CUPY_BACKENDS = CUPY_BACKENDS  #: Set of backends that use CuPy, subset of class CUDA
    SPARSE_BACKENDS = SPARSE_BACKENDS  #: Set of backends that are sparse arrays
    DENSE_BACKENDS = DENSE_BACKENDS  #: Set of backends that are dense arrays
    ND_BACKENDS = ND_BACKENDS  #: Set of backends that support n-dimensional arrays
    D2_BACKENDS = D2_BACKENDS  #: Set of backends that only support two-dimensional arrays

    UDF_METHOD = UDFMethod  #: Enum of process_ methods accepted by the UDF interface

    def get_method() -> UDFMethod:
        raise NotImplementedError()

    def get_tiling_preferences() -> TilingPreferences:
        raise NotImplementedError()
